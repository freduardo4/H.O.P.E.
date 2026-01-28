import numpy as np
import copy
from typing import List, Tuple, Callable

class TuneOptimizer:
    """
    Genetic Algorithm optimizer for ECU calibration maps.
    Optimizes a 'current_map' (e.g., VE table) to minimize error against a target (e.g., Target AFR).
    """

    def __init__(self, 
                 population_size: int = 50, 
                 mutation_rate: float = 0.05, 
                 crossover_rate: float = 0.7,
                 mutation_strength: float = 0.1): 
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_strength = mutation_strength

    def evolve(self, 
               current_map: np.ndarray, 
               datalog_mask: np.ndarray, 
               target_map: np.ndarray,
               generations: int = 20) -> Tuple[np.ndarray, List[float]]:
        """
        Evolves the current_map to match target_map where datalog_mask indicates valid data.
        
        Args:
            current_map: 2D array representing the starting tune (e.g., VE table).
            datalog_mask: 2D binary/boolean array or weights where 1 indicates we have valid data to tune.
                          (In real usage, this would be derived from log hits). 
                          For simplicity here, we assume we optimize 'current_map' directly against 'target_map' 
                          as if 'target_map' is the "ideal" map we want to find.
                          In a real scenario, fitness would be: |TargetAFR - MeasuredAFR|.
                          Here we simulate: We want 'current_map' to become 'target_map'.
            target_map: The ideal map we want to achieve (Ground Truth).
            generations: Number of generations to run.

        Returns:
            Tuple of (Best Map Found, History of Best Fitness Scores)
        """
        
        # 1. Initialize Population
        population = self._initialize_population(current_map, self.population_size)
        best_fitness_history = []
        
        for generation in range(generations):
            # 2. Fitness Evaluation
            fitness_scores = [self.fitness_function(ind, datalog_mask, target_map) for ind in population]
            
            # Record stats
            best_fitness = min(fitness_scores)
            best_fitness_history.append(best_fitness)
            
            # Elitism: keep the best
            sorted_pop_indices = np.argsort(fitness_scores)
            new_population = [population[sorted_pop_indices[0]]] # Keep best 1
            
            # 3. Selection & Reproduction
            # We need population_size - 1 more individuals
            while len(new_population) < self.population_size:
                # Tournament Selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if np.random.rand() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)
                
                # Mutation
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population

        # Return best of final generation
        fitness_scores = [self.fitness_function(ind, datalog_mask, target_map) for ind in population]
        best_idx = np.argmin(fitness_scores)
        return population[best_idx], best_fitness_history

    def _initialize_population(self, base_map: np.ndarray, size: int) -> List[np.ndarray]:
        """Creates initial population by perturbing the base map."""
        population = [base_map] # Include original
        for _ in range(size - 1):
            # random noise based on mutation strength
            noise = 1.0 + (np.random.rand(*base_map.shape) - 0.5) * (self.mutation_strength * 2)
            individual = base_map * noise
            population.append(individual)
        return population

    def fitness_function(self, individual: np.ndarray, mask: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate fitness (Lower is better).
        Simple MSE (Mean Squared Error) weighted by mask.
        """
        diff = (individual - target) * mask
        mse = np.mean(np.square(diff))
        return mse

    def _tournament_selection(self, population: list, fitness_scores: list, k: int=3) -> np.ndarray:
        """Select best individual from random k individuals."""
        indices = np.random.randint(0, len(population), k)
        best_idx = indices[0]
        for idx in indices[1:]:
            if fitness_scores[idx] < fitness_scores[best_idx]:
                best_idx = idx
        return population[best_idx]

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """2D Crossover: Swap rectangular regions."""
        child = copy.deepcopy(parent1)
        rows, cols = parent1.shape
        
        # Pick random crossover point (row split or col split)
        if np.random.rand() > 0.5:
            # Row split
            split = np.random.randint(0, rows)
            child[split:, :] = parent2[split:, :]
        else:
            # Col split
            split = np.random.randint(0, cols)
            child[:, split:] = parent2[:, split:]
            
        return child

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Randomly perturb cells."""
        rows, cols = individual.shape
        # Create mutation mask
        mask = np.random.rand(rows, cols) < self.mutation_rate
        
        # Apply mutation: +/- mutation_strength range
        # e.g. if strength=0.1, factor is between 0.9 and 1.1
        mutation_factor = 1.0 + (np.random.rand(rows, cols) - 0.5) * (self.mutation_strength * 2)
        
        individual[mask] *= mutation_factor[mask]
        return individual
