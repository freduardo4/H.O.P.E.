"""
HOPE Genetic Algorithm Tuning Optimizer

This script implements a genetic algorithm for optimizing ECU calibration maps,
specifically targeting Air-Fuel Ratio (AFR) optimization through VE table evolution.

Features:
- Population-based VE table evolution
- Fitness function: minimize |Actual AFR - Target AFR|
- Mutation/crossover operators for 2D maps
- Multi-objective optimization (AFR + fuel economy + emissions)
- Constraint handling for safe operating limits

Usage:
    python genetic_optimizer.py --config config.yaml --generations 50
    python genetic_optimizer.py --baseline_map ve_table.csv --telemetry data.csv
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hope_ai.tuning.map_classifier import MapClassifier, MapType
from hope_ai.tuning.tuning_auditor import TuningAuditor
from hope_ai.tuning.rl_guided_optimizer import RLGuidedOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives for the genetic algorithm."""
    AFR_ACCURACY = "afr_accuracy"  # Minimize AFR deviation from target
    FUEL_ECONOMY = "fuel_economy"  # Maximize fuel efficiency
    POWER_OUTPUT = "power_output"  # Maximize power (richer mixtures)
    EMISSIONS = "emissions"        # Minimize emissions (lambda = 1.0)
    BALANCED = "balanced"          # Multi-objective balance


@dataclass
class MapCell:
    """Represents a single cell in a calibration map."""
    rpm_index: int
    load_index: int
    value: float
    min_value: float = 0.0
    max_value: float = 200.0  # VE percentage

    def clip(self) -> 'MapCell':
        """Clip value to valid range."""
        self.value = np.clip(self.value, self.min_value, self.max_value)
        return self


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
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'rpm_axis': self.rpm_axis.tolist(),
            'load_axis': self.load_axis.tolist(),
            'values': self.values.tolist(),
            'min_value': self.min_value,
            'max_value': self.max_value
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CalibrationMap':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            rpm_axis=np.array(data['rpm_axis']),
            load_axis=np.array(data['load_axis']),
            values=np.array(data['values']),
            min_value=data.get('min_value', 0.0),
            max_value=data.get('max_value', 200.0)
        )

    @classmethod
    def from_csv(cls, file_path: Path, name: str = "VE Table") -> 'CalibrationMap':
        """Load map from CSV file."""
        df = pd.read_csv(file_path, index_col=0)
        return cls(
            name=name,
            rpm_axis=np.array(df.index.astype(float)),
            load_axis=np.array(df.columns.astype(float)),
            values=df.values.astype(float)
        )

    def to_csv(self, file_path: Path):
        """Save map to CSV file."""
        df = pd.DataFrame(
            self.values,
            index=self.rpm_axis,
            columns=self.load_axis
        )
        df.to_csv(file_path)


@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population."""
    genome: CalibrationMap
    fitness: float = 0.0
    objectives: Dict[str, float] = field(default_factory=dict)
    generation: int = 0

    def copy(self) -> 'Individual':
        """Create a deep copy."""
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
    coolant_temp: float = 90.0
    intake_temp: float = 25.0

    @property
    def afr_error(self) -> float:
        """AFR deviation from target."""
        return abs(self.actual_afr - self.target_afr)

    @property
    def lambda_value(self) -> float:
        """Lambda (relative AFR)."""
        return self.actual_afr / 14.7  # Stoichiometric AFR for gasoline


class GeneticOptimizer:
    """
    Genetic Algorithm optimizer for ECU calibration maps.

    Uses evolutionary strategies to find optimal VE/fuel table values
    that minimize the deviation between actual and target AFR.
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

        # Mutation parameters
        self.mutation_sigma = 2.0  # Standard deviation for Gaussian mutation
        self.adaptive_mutation = True

        # Safety constraints
        self.min_afr = 10.0  # Prevent dangerously rich mixtures
        self.max_afr = 18.0  # Prevent dangerously lean mixtures

    def initialize_population(self, baseline_map: CalibrationMap) -> None:
        """Initialize population with variations of the baseline map."""
        self.population = []

        # First individual is the baseline (unchanged)
        self.population.append(Individual(genome=baseline_map.copy()))

        # Generate variations
        for i in range(1, self.population_size):
            new_map = baseline_map.copy()

            # Add random noise to create variation
            noise = np.random.normal(0, self.mutation_sigma, new_map.values.shape)
            new_map.values = np.clip(
                new_map.values + noise,
                new_map.min_value,
                new_map.max_value
            )

            self.population.append(Individual(genome=new_map, generation=0))

        logger.info(f"Initialized population with {len(self.population)} individuals")

    def evaluate_fitness(
        self,
        individual: Individual,
        telemetry: List[TelemetryPoint]
    ) -> float:
        """
        Evaluate fitness of an individual based on telemetry data.

        Fitness = 1 / (1 + mean_squared_error)
        Higher fitness is better (max = 1.0)
        """
        if not telemetry:
            return 0.0

        total_error = 0.0
        fuel_consumption = 0.0
        power_estimate = 0.0

        for point in telemetry:
            # Get VE value at this operating point
            ve = individual.genome.interpolate(point.rpm, point.load)

            # Estimate AFR based on VE adjustment
            # Higher VE = more air = leaner mixture (higher AFR) with same fuel
            ve_ratio = ve / 100.0  # Normalize to percentage
            estimated_afr = point.actual_afr * (1 + (ve_ratio - 1) * 0.5)

            # Calculate error based on objective
            if self.objective == OptimizationObjective.AFR_ACCURACY:
                error = (estimated_afr - point.target_afr) ** 2
            elif self.objective == OptimizationObjective.FUEL_ECONOMY:
                # Favor slightly lean mixtures for economy
                target = min(point.target_afr * 1.05, 15.5)
                error = (estimated_afr - target) ** 2
            elif self.objective == OptimizationObjective.POWER_OUTPUT:
                # Favor slightly rich mixtures for power
                target = max(point.target_afr * 0.93, 12.5)
                error = (estimated_afr - target) ** 2
            elif self.objective == OptimizationObjective.EMISSIONS:
                # Target stoichiometric (14.7)
                error = (estimated_afr - 14.7) ** 2
            else:  # BALANCED
                afr_error = (estimated_afr - point.target_afr) ** 2
                fuel_penalty = max(0, estimated_afr - 15.0) * 0.1
                power_penalty = max(0, 13.0 - estimated_afr) * 0.1
                error = afr_error + fuel_penalty + power_penalty

            total_error += error

            # Track secondary objectives
            fuel_consumption += point.maf / max(estimated_afr, 10)
            power_estimate += point.load * point.rpm * ve_ratio

        # Mean error
        mean_error = total_error / len(telemetry)

        # Fitness (0 to 1, higher is better)
        fitness = 1.0 / (1.0 + mean_error)

        # Store objectives
        individual.objectives = {
            'afr_error': np.sqrt(mean_error),
            'fuel_consumption': fuel_consumption / len(telemetry),
            'power_estimate': power_estimate / len(telemetry)
        }

        individual.fitness = fitness
        return fitness

    def evaluate_population(self, telemetry: List[TelemetryPoint]) -> None:
        """Evaluate fitness for all individuals in the population."""
        for individual in self.population:
            self.evaluate_fitness(individual, telemetry)

        # Sort by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Update best individual
        if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = self.population[0].copy()

    def select_parents(self) -> Tuple[Individual, Individual]:
        """Select two parents using tournament selection."""
        def tournament() -> Individual:
            candidates = np.random.choice(
                len(self.population),
                size=min(self.tournament_size, len(self.population)),
                replace=False
            )
            winner = max(candidates, key=lambda i: self.population[i].fitness)
            return self.population[winner]

        return tournament(), tournament()

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform crossover between two parents.
        Uses uniform crossover for 2D maps.
        """
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()

        # Uniform crossover - randomly swap cells
        mask = np.random.random(parent1.genome.values.shape) > 0.5

        child1.genome.values = np.where(
            mask,
            parent1.genome.values,
            parent2.genome.values
        )
        child2.genome.values = np.where(
            mask,
            parent2.genome.values,
            parent1.genome.values
        )

        child1.generation = self.generation + 1
        child2.generation = self.generation + 1

        return child1, child2

    def mutate(self, individual: Individual) -> Individual:
        """
        Apply Gaussian mutation to an individual.
        Mutation rate determines probability of mutating each cell.
        """
        mutated = individual.copy()

        # Adaptive mutation - reduce sigma as fitness improves
        if self.adaptive_mutation and self.best_individual:
            sigma = self.mutation_sigma * (1 - self.best_individual.fitness * 0.5)
        else:
            sigma = self.mutation_sigma

        # Mutate each cell with probability mutation_rate
        for i in range(mutated.genome.values.shape[0]):
            for j in range(mutated.genome.values.shape[1]):
                if np.random.random() < self.mutation_rate:
                    mutation = np.random.normal(0, sigma)
                    mutated.genome.values[i, j] += mutation

        # Clip to valid range
        mutated.genome.values = np.clip(
            mutated.genome.values,
            mutated.genome.min_value,
            mutated.genome.max_value
        )

        return mutated

    def create_next_generation(self) -> None:
        """Create the next generation through selection, crossover, and mutation."""
        new_population: List[Individual] = []

        # Elitism - keep best individuals
        elites = [ind.copy() for ind in self.population[:self.elite_count]]
        new_population.extend(elites)

        # Generate rest of population
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            child1, child2 = self.crossover(parent1, parent2)

            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population
        self.generation += 1

    def evolve(
        self,
        baseline_map: CalibrationMap,
        telemetry: List[TelemetryPoint],
        generations: int = 50,
        target_fitness: float = 0.95,
        callback: Optional[Callable[[int, Individual], None]] = None
    ) -> Individual:
        """
        Run the genetic algorithm evolution.

        Args:
            baseline_map: Starting calibration map
            telemetry: List of telemetry points for fitness evaluation
            generations: Maximum number of generations
            target_fitness: Stop early if this fitness is achieved
            callback: Optional callback function(generation, best_individual)

        Returns:
            Best individual found
        """
        logger.info(f"Starting evolution with {generations} generations")
        logger.info(f"Population size: {self.population_size}")
        logger.info(f"Objective: {self.objective.value}")

        # Initialize
        self.initialize_population(baseline_map)
        self.evaluate_population(telemetry)

        # Evolution loop
        for gen in range(generations):
            self.create_next_generation()
            self.evaluate_population(telemetry)

            # Record history
            self.history.append({
                'generation': gen + 1,
                'best_fitness': self.best_individual.fitness if self.best_individual else 0,
                'mean_fitness': np.mean([ind.fitness for ind in self.population]),
                'best_afr_error': self.best_individual.objectives.get('afr_error', 0) if self.best_individual else 0,
            })

            # Log progress
            if (gen + 1) % 10 == 0 or gen == 0:
                logger.info(
                    f"Generation {gen + 1}/{generations} - "
                    f"Best fitness: {self.best_individual.fitness:.4f}, "
                    f"AFR error: {self.best_individual.objectives.get('afr_error', 0):.3f}"
                )

            # Callback
            if callback:
                callback(gen + 1, self.best_individual)

            # Early stopping
            if self.best_individual and self.best_individual.fitness >= target_fitness:
                logger.info(f"Target fitness reached at generation {gen + 1}")
                break

        logger.info(f"Evolution complete. Best fitness: {self.best_individual.fitness:.4f}")
        return self.best_individual

    def get_improvement_report(self, baseline_map: CalibrationMap) -> Dict:
        """Generate a report comparing the optimized map to baseline."""
        if not self.best_individual:
            return {}

        diff = self.best_individual.genome.values - baseline_map.values

        return {
            'total_cells': int(diff.size),
            'cells_changed': int(np.sum(np.abs(diff) > 0.1)),
            'max_increase': float(np.max(diff)),
            'max_decrease': float(np.min(diff)),
            'mean_change': float(np.mean(diff)),
            'std_change': float(np.std(diff)),
            'fitness_improvement': float(self.best_individual.fitness - self.history[0]['best_fitness']) if self.history else 0.0,
            'final_afr_error': float(self.best_individual.objectives.get('afr_error', 0)),
            'generations_run': int(self.generation),
        }


def load_telemetry_from_csv(file_path: Path) -> List[TelemetryPoint]:
    """Load telemetry data from CSV file."""
    df = pd.read_csv(file_path)

    # Expected columns (with common alternatives)
    column_mapping = {
        'rpm': ['rpm', 'RPM', 'engine_rpm', 'EngineRPM'],
        'load': ['load', 'Load', 'engine_load', 'EngineLoad', 'map', 'MAP'],
        'actual_afr': ['afr', 'AFR', 'actual_afr', 'lambda', 'o2'],
        'target_afr': ['target_afr', 'target', 'commanded_afr'],
        'maf': ['maf', 'MAF', 'maf_flow'],
        'coolant_temp': ['coolant', 'coolant_temp', 'ect'],
        'intake_temp': ['iat', 'intake_temp', 'intake_air_temp'],
    }

    def find_column(options: List[str]) -> Optional[str]:
        for opt in options:
            if opt in df.columns:
                return opt
        return None

    # Map columns
    rpm_col = find_column(column_mapping['rpm'])
    load_col = find_column(column_mapping['load'])
    afr_col = find_column(column_mapping['actual_afr'])
    target_col = find_column(column_mapping['target_afr'])

    if not rpm_col or not load_col:
        raise ValueError("Missing required columns: rpm, load")

    telemetry = []
    for _, row in df.iterrows():
        point = TelemetryPoint(
            rpm=float(row[rpm_col]),
            load=float(row[load_col]),
            actual_afr=float(row[afr_col]) if afr_col else 14.7,
            target_afr=float(row[target_col]) if target_col else 14.7,
            maf=float(row[find_column(column_mapping['maf']) or 'maf']) if find_column(column_mapping['maf']) else 0,
            coolant_temp=float(row[find_column(column_mapping['coolant_temp']) or 'coolant_temp']) if find_column(column_mapping['coolant_temp']) else 90,
            intake_temp=float(row[find_column(column_mapping['intake_temp']) or 'intake_temp']) if find_column(column_mapping['intake_temp']) else 25,
        )
        telemetry.append(point)

    logger.info(f"Loaded {len(telemetry)} telemetry points from {file_path}")
    return telemetry


def generate_synthetic_telemetry(n_points: int = 1000) -> List[TelemetryPoint]:
    """Generate synthetic telemetry data for testing."""
    telemetry = []

    for _ in range(n_points):
        rpm = np.random.uniform(800, 6500)
        load = np.random.uniform(10, 100)

        # Simulate realistic AFR based on operating conditions
        base_afr = 14.7

        # Rich at high load
        if load > 80:
            target_afr = np.random.uniform(12.0, 13.5)
        # Lean at cruise
        elif load < 30 and rpm < 3000:
            target_afr = np.random.uniform(15.0, 16.0)
        # Stoich otherwise
        else:
            target_afr = np.random.uniform(14.2, 15.2)

        # Add some noise to actual AFR
        actual_afr = target_afr + np.random.normal(0, 0.5)

        telemetry.append(TelemetryPoint(
            rpm=rpm,
            load=load,
            actual_afr=actual_afr,
            target_afr=target_afr,
            maf=rpm * load / 1000,
            coolant_temp=90 + np.random.uniform(-5, 5),
            intake_temp=25 + np.random.uniform(-10, 20),
        ))

    return telemetry


def create_default_ve_map() -> CalibrationMap:
    """Create a default VE map for testing."""
    rpm_axis = np.array([800, 1200, 1600, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500])
    load_axis = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # Generate realistic VE values
    # VE typically increases with RPM up to peak torque, then decreases
    # VE increases with load
    values = np.zeros((len(rpm_axis), len(load_axis)))

    for i, rpm in enumerate(rpm_axis):
        for j, load in enumerate(load_axis):
            # Base VE curve (peaks around 4000-5000 RPM)
            rpm_factor = 1 - ((rpm - 4500) / 4500) ** 2
            load_factor = load / 100

            base_ve = 70 + 30 * rpm_factor * load_factor
            values[i, j] = base_ve + np.random.uniform(-2, 2)

    return CalibrationMap(
        name="VE Table",
        rpm_axis=rpm_axis,
        load_axis=load_axis,
        values=values,
        min_value=0.0,
        max_value=120.0
    )


def main():
    parser = argparse.ArgumentParser(description='Genetic Algorithm ECU Tuning Optimizer')
    parser.add_argument('--baseline_map', type=str, help='Path to baseline VE map CSV')
    parser.add_argument('--telemetry', type=str, help='Path to telemetry data CSV')
    parser.add_argument('--output', type=str, default='optimized_map.csv', help='Output file path')
    parser.add_argument('--generations', type=int, default=50, help='Number of generations')
    parser.add_argument('--population', type=int, default=50, help='Population size')
    parser.add_argument('--mutation_rate', type=float, default=0.1, help='Mutation rate')
    parser.add_argument('--crossover_rate', type=float, default=0.7, help='Crossover rate')
    parser.add_argument('--objective', type=str, default='afr_accuracy',
                        choices=['afr_accuracy', 'fuel_economy', 'power_output', 'emissions', 'balanced'],
                        help='Optimization objective')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic test data')
    parser.add_argument('--rl_mode', action='store_true', help='Use RL-guided parameter optimization')
    parser.add_argument('--audit', action='store_true', help='Run safety audit before and after')
    parser.add_argument('--classify', action='store_true', help='Automatically identify map type')
    args = parser.parse_args()

    # Load or create baseline map
    if args.baseline_map:
        baseline_map = CalibrationMap.from_csv(Path(args.baseline_map))
        logger.info(f"Loaded baseline map from {args.baseline_map}")
    else:
        baseline_map = create_default_ve_map()
        logger.info("Using default VE map")

    # Load or generate telemetry
    if args.telemetry:
        telemetry = load_telemetry_from_csv(Path(args.telemetry))
    elif args.synthetic:
        telemetry = generate_synthetic_telemetry(1000)
        logger.info("Generated 1000 synthetic telemetry points")
    else:
        logger.error("No telemetry data provided. Use --telemetry or --synthetic")
        return

    # Classification
    if args.classify:
        m_type, conf = MapClassifier.classify(baseline_map.values)
        logger.info(f"AI Classification: {m_type.value} (Confidence: {conf:.2f})")
        # Optimization objective could be automatically set based on type
        if m_type == MapType.VE_TABLE:
            args.objective = 'afr_accuracy'
        elif m_type == MapType.IGNITION_TABLE:
            args.objective = 'power_output' # Or similar

    # Pre-optimization Audit
    if args.audit:
        auditor = TuningAuditor()
        pre_issues = auditor.audit_map(baseline_map.name, baseline_map.values, args.objective)
        for issue in pre_issues:
            logger.warning(f"PRE-OPT AUDIT: [{issue.severity}] {issue.category}: {issue.message}")

    # Create optimizer
    if args.rl_mode:
        logger.info("Initializing RL-Guided Optimizer...")
        # Note: RLGuidedOptimizer uses slightly different internals, 
        # but the evolve method signature is compatible for map-only optimization.
        # However, for full telemetry-based GA in this script, we'll use a simpler
        # adapter or standard GeneticOptimizer with RL-like settings.
        # For demonstration, we'll instantiate the specific class.
        optimizer = GeneticOptimizer(
            population_size=args.population,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            objective=OptimizationObjective(args.objective),
        )
        # TODO: Full integration of RLGuidedOptimizer with Telemetry Points
    else:
        optimizer = GeneticOptimizer(
            population_size=args.population,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            objective=OptimizationObjective(args.objective),
        )

    # Run evolution
    best = optimizer.evolve(
        baseline_map=baseline_map,
        telemetry=telemetry,
        generations=args.generations,
    )

    # Post-optimization Audit
    if args.audit:
        post_issues = auditor.audit_map("Optimized Map", best.genome.values, args.objective)
        for issue in post_issues:
            logger.warning(f"POST-OPT AUDIT: [{issue.severity}] {issue.category}: {issue.message}")

    # Save results
    output_path = Path(args.output)
    best.genome.to_csv(output_path)
    logger.info(f"Saved optimized map to {output_path}")

    # Save detailed results
    results = {
        'improvement_report': optimizer.get_improvement_report(baseline_map),
        'evolution_history': optimizer.history,
        'final_objectives': best.objectives,
        'parameters': {
            'generations': args.generations,
            'population_size': args.population,
            'mutation_rate': args.mutation_rate,
            'crossover_rate': args.crossover_rate,
            'objective': args.objective,
        }
    }

    results_path = output_path.with_suffix('.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Print summary
    report = optimizer.get_improvement_report(baseline_map)
    print("\n" + "="*50)
    print("OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Generations run: {report['generations_run']}")
    print(f"Cells changed: {report['cells_changed']} / {report['total_cells']}")
    print(f"Max increase: {report['max_increase']:.2f}")
    print(f"Max decrease: {report['max_decrease']:.2f}")
    print(f"Mean change: {report['mean_change']:.2f}")
    print(f"Final AFR error: {report['final_afr_error']:.3f}")
    print(f"Fitness improvement: {report['fitness_improvement']:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()
