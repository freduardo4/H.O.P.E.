import os
import sys
import logging
import json
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Optimization objectives for the genetic algorithm."""
    AFR_ACCURACY = "afr_accuracy"  # Minimize AFR deviation from target
    FUEL_ECONOMY = "fuel_economy"  # Maximize fuel efficiency
    POWER_OUTPUT = "power_output"  # Maximize power (richer mixtures)
    EMISSIONS = "emissions"        # Minimize emissions (lambda = 1.0)
    BALANCED = "balanced"          # Multi-objective balance

@dataclass
class CalibrationMap:
    """Represents a 2D calibration map (e.g., VE table, fuel table)."""
    name: str
    rpm_axis: np.ndarray  # RPM breakpoints
    load_axis: np.ndarray  # Load/MAP breakpoints
    values: np.ndarray    # 2D array of values
    min_value: float = 0.0
    max_value: float = 200.0

    def __post_init__(self):
        """Validate map dimensions."""
        assert len(self.rpm_axis) == self.values.shape[0], "RPM axis mismatch"
        assert len(self.load_axis) == self.values.shape[1], "Load axis mismatch"

    def interpolate(self, rpm: float, load: float) -> float:
        """Bilinear interpolation at given RPM and load."""
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (self.rpm_axis, self.load_axis),
            self.values,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        return float(interp([[rpm, load]])[0])

    def copy(self) -> 'CalibrationMap':
        """Create a deep copy of the map."""
        return CalibrationMap(
            name=self.name,
            rpm_axis=self.rpm_axis.copy(),
            load_axis=self.load_axis.copy(),
            values=self.values.copy(),
            min_value=self.min_value,
            max_value=self.max_value
        )

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'rpm_axis': self.rpm_axis.tolist(),
            'load_axis': self.load_axis.tolist(),
            'values': self.values.tolist(),
            'min_value': self.min_value,
            'max_value': self.max_value
        }

@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population."""
    genome: CalibrationMap
    fitness: float = 0.0
    objectives: Dict[str, float] = field(default_factory=dict)
    generation: int = 0

    def copy(self) -> 'Individual':
        return Individual(
            genome=self.genome.copy(),
            fitness=self.fitness,
            objectives=self.objectives.copy(),
            generation=self.generation
        )

@dataclass
class TelemetryPoint:
    """Single telemetry data point for fitness evaluation."""
    rpm: float
    load: float
    actual_afr: float
    target_afr: float
    maf: float = 0.0

class GeneticOptimizer:
    """
    Genetic Algorithm optimizer for ECU calibration maps.
    """

    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_count: int = 5,
        tournament_size: int = 3,
        objective: OptimizationObjective = OptimizationObjective.AFR_ACCURACY,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.objective = objective
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.generation = 0
        self.history: List[Dict] = []
        self.mutation_sigma = 2.0

    def initialize_population(self, baseline_map: CalibrationMap) -> None:
        self.population = [Individual(genome=baseline_map.copy())]
        for _ in range(1, self.population_size):
            new_map = baseline_map.copy()
            noise = np.random.normal(0, self.mutation_sigma, new_map.values.shape)
            new_map.values = np.clip(new_map.values + noise, new_map.min_value, new_map.max_value)
            self.population.append(Individual(genome=new_map, generation=0))

    def evaluate_fitness(self, individual: Individual, telemetry: List[TelemetryPoint]) -> float:
        if not telemetry: return 0.0
        total_error = 0.0
        for point in telemetry:
            ve = individual.genome.interpolate(point.rpm, point.load)
            ve_ratio = ve / 100.0
            estimated_afr = point.actual_afr * (1 + (ve_ratio - 1) * 0.5)
            error = (estimated_afr - point.target_afr) ** 2
            total_error += error
        
        mean_error = total_error / len(telemetry)
        fitness = 1.0 / (1.0 + mean_error)
        individual.fitness = fitness
        individual.objectives = {'afr_error': np.sqrt(mean_error)}
        return fitness

    def evolve(
        self,
        baseline_map: CalibrationMap,
        telemetry: List[TelemetryPoint],
        generations: int = 50,
    ) -> Individual:
        self.initialize_population(baseline_map)
        for gen in range(generations):
            for ind in self.population:
                self.evaluate_fitness(ind, telemetry)
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            self.best_individual = self.population[0].copy()
            
            # Create next gen
            new_pop = [ind.copy() for ind in self.population[:self.elite_count]]
            while len(new_pop) < self.population_size:
                p1 = self._tournament_selection()
                p2 = self._tournament_selection()
                c1, c2 = self._crossover(p1, p2)
                new_pop.extend([self._mutate(c1), self._mutate(c2)])
            self.population = new_pop[:self.population_size]
            self.generation += 1
            
            if (gen + 1) % 10 == 0:
                logger.info(f"Gen {gen+1} - Best Fitness: {self.best_individual.fitness:.4f}")
        
        return self.best_individual

    def _tournament_selection(self) -> Individual:
        candidates = np.random.choice(len(self.population), size=self.tournament_size, replace=False)
        return max([self.population[i] for i in candidates], key=lambda x: x.fitness)

    def _crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        if np.random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        c1, c2 = p1.copy(), p2.copy()
        mask = np.random.random(p1.genome.values.shape) > 0.5
        c1.genome.values = np.where(mask, p1.genome.values, p2.genome.values)
        c2.genome.values = np.where(mask, p2.genome.values, p1.genome.values)
        return c1, c2

    def _mutate(self, ind: Individual) -> Individual:
        mutated = ind.copy()
        mask = np.random.random(mutated.genome.values.shape) < self.mutation_rate
        noise = np.random.normal(0, self.mutation_sigma, mutated.genome.values.shape)
        mutated.genome.values[mask] += noise[mask]
        mutated.genome.values = np.clip(mutated.genome.values, mutated.genome.min_value, mutated.genome.max_value)
        return mutated

class TuneOptimizer:
    """
    Simplified, array-based genetic optimizer used by CLI and legacy tests.
    """
    def __init__(self, population_size=50, mutation_rate=0.1, crossover_rate=0.7, mutation_strength=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_strength = mutation_strength

    def fitness_function(self, individual, mask, target):
        error = np.sum(((individual - target) * mask) ** 2)
        return error / (np.sum(mask) + 1e-6)

    def mutate(self, individual):
        res = individual.copy()
        mask = np.random.rand(*res.shape) < self.mutation_rate
        noise = np.random.randn(*res.shape) * self.mutation_strength
        res[mask] += noise[mask]
        return res

    def crossover(self, parent1, parent2):
        mask = np.random.rand(*parent1.shape) > 0.5
        return np.where(mask, parent1, parent2)

    def _initialize_population(self, start_map, size):
        return [self.mutate(start_map.copy()) for _ in range(size)]

    def _tournament_selection(self, population, fitness_scores):
        candidates = np.random.choice(len(population), size=3, replace=False)
        # Lower fitness is better in the Array-based version (MSE)
        winner_idx = candidates[np.argmin([fitness_scores[i] for i in candidates])]
        return population[winner_idx]

    def evolve(self, start_map, mask, target_map, generations=20):
        population = self._initialize_population(start_map, self.population_size)
        history = []
        
        for gen in range(generations):
            fitness = [self.fitness_function(ind, mask, target_map) for ind in population]
            best_idx = np.argmin(fitness)
            history.append(float(fitness[best_idx]))
            
            new_population = [population[best_idx].copy()] # Elitism
            while len(new_population) < self.population_size:
                p1 = self._tournament_selection(population, fitness)
                p2 = self._tournament_selection(population, fitness)
                
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
            
        final_fitness = [self.fitness_function(ind, mask, target_map) for ind in population]
        best_idx = np.argmin(final_fitness)
        return population[best_idx], history
