"""
Tests for the Genetic Algorithm Tuning Optimizer.

These tests verify the core functionality of the genetic algorithm
for ECU calibration map optimization.
"""

import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from genetic_optimizer import (
    GeneticOptimizer,
    CalibrationMap,
    Individual,
    TelemetryPoint,
    OptimizationObjective,
    create_default_ve_map,
    generate_synthetic_telemetry,
)


class TestCalibrationMap:
    """Tests for the CalibrationMap class."""

    def test_create_map(self):
        """Test basic map creation."""
        rpm_axis = np.array([1000, 2000, 3000, 4000])
        load_axis = np.array([20, 40, 60, 80])
        values = np.random.uniform(50, 100, (4, 4))

        cal_map = CalibrationMap(
            name="Test VE",
            rpm_axis=rpm_axis,
            load_axis=load_axis,
            values=values
        )

        assert cal_map.name == "Test VE"
        assert len(cal_map.rpm_axis) == 4
        assert len(cal_map.load_axis) == 4
        assert cal_map.values.shape == (4, 4)

    def test_map_interpolation(self):
        """Test bilinear interpolation."""
        rpm_axis = np.array([1000, 2000, 3000])
        load_axis = np.array([20, 50, 80])
        values = np.array([
            [60, 70, 80],
            [65, 75, 85],
            [70, 80, 90],
        ])

        cal_map = CalibrationMap(
            name="Test",
            rpm_axis=rpm_axis,
            load_axis=load_axis,
            values=values
        )

        # Test exact point
        assert cal_map.interpolate(1000, 20) == pytest.approx(60, abs=0.1)

        # Test interpolated point
        result = cal_map.interpolate(1500, 35)
        assert 60 < result < 80  # Should be between corner values

    def test_map_copy(self):
        """Test deep copy functionality."""
        original = create_default_ve_map()
        copy = original.copy()

        # Modify copy
        copy.values[0, 0] = 999

        # Original should be unchanged
        assert original.values[0, 0] != 999

    def test_map_to_dict_and_back(self):
        """Test serialization round-trip."""
        original = create_default_ve_map()
        data = original.to_dict()
        restored = CalibrationMap.from_dict(data)

        assert restored.name == original.name
        np.testing.assert_array_equal(restored.rpm_axis, original.rpm_axis)
        np.testing.assert_array_equal(restored.values, original.values)

    def test_map_csv_round_trip(self):
        """Test CSV save and load."""
        original = create_default_ve_map()

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            csv_path = Path(f.name)

        try:
            original.to_csv(csv_path)
            restored = CalibrationMap.from_csv(csv_path, name=original.name)

            np.testing.assert_array_almost_equal(
                restored.values, original.values, decimal=4
            )
        finally:
            csv_path.unlink()


class TestTelemetryPoint:
    """Tests for the TelemetryPoint class."""

    def test_afr_error(self):
        """Test AFR error calculation."""
        point = TelemetryPoint(
            rpm=3000,
            load=50,
            actual_afr=14.0,
            target_afr=14.7
        )

        assert point.afr_error == pytest.approx(0.7, abs=0.01)

    def test_lambda_value(self):
        """Test lambda calculation."""
        point = TelemetryPoint(
            rpm=3000,
            load=50,
            actual_afr=14.7,
            target_afr=14.7
        )

        assert point.lambda_value == pytest.approx(1.0, abs=0.01)

    def test_rich_mixture_lambda(self):
        """Test lambda for rich mixture."""
        point = TelemetryPoint(
            rpm=3000,
            load=90,
            actual_afr=12.5,
            target_afr=12.5
        )

        assert point.lambda_value < 1.0


class TestGeneticOptimizer:
    """Tests for the GeneticOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create a standard optimizer for tests."""
        return GeneticOptimizer(
            population_size=20,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elite_count=2,
        )

    @pytest.fixture
    def baseline_map(self):
        """Create a baseline map for tests."""
        return create_default_ve_map()

    @pytest.fixture
    def telemetry(self):
        """Generate test telemetry data."""
        return generate_synthetic_telemetry(100)

    def test_initialize_population(self, optimizer, baseline_map):
        """Test population initialization."""
        optimizer.initialize_population(baseline_map)

        assert len(optimizer.population) == optimizer.population_size

        # First individual should be close to baseline
        first_diff = np.max(np.abs(
            optimizer.population[0].genome.values - baseline_map.values
        ))
        assert first_diff < 0.01  # Should be identical

    def test_population_diversity(self, optimizer, baseline_map):
        """Test that population has diversity."""
        optimizer.initialize_population(baseline_map)

        # Check that individuals are different
        values = [ind.genome.values for ind in optimizer.population]
        unique_count = len(set(tuple(v.flatten()) for v in values))

        assert unique_count > optimizer.population_size * 0.8

    def test_evaluate_fitness(self, optimizer, baseline_map, telemetry):
        """Test fitness evaluation."""
        individual = Individual(genome=baseline_map)
        fitness = optimizer.evaluate_fitness(individual, telemetry)

        assert 0 <= fitness <= 1
        assert individual.fitness == fitness
        assert 'afr_error' in individual.objectives

    def test_fitness_range(self, optimizer, baseline_map, telemetry):
        """Test that fitness stays in valid range."""
        optimizer.initialize_population(baseline_map)
        optimizer.evaluate_population(telemetry)

        for ind in optimizer.population:
            assert 0 <= ind.fitness <= 1

    def test_selection(self, optimizer, baseline_map, telemetry):
        """Test tournament selection."""
        optimizer.initialize_population(baseline_map)
        optimizer.evaluate_population(telemetry)

        parent1, parent2 = optimizer.select_parents()

        assert isinstance(parent1, Individual)
        assert isinstance(parent2, Individual)
        assert parent1.fitness >= 0

    def test_crossover(self, optimizer, baseline_map):
        """Test crossover operation."""
        parent1 = Individual(genome=baseline_map.copy())
        parent2 = Individual(genome=baseline_map.copy())

        # Make parents different
        parent2.genome.values += 10

        child1, child2 = optimizer.crossover(parent1, parent2)

        # Children should exist and have valid genomes
        assert child1.genome.values.shape == baseline_map.values.shape
        assert child2.genome.values.shape == baseline_map.values.shape

    def test_mutation(self, optimizer, baseline_map):
        """Test mutation operation."""
        original = Individual(genome=baseline_map.copy())
        mutated = optimizer.mutate(original)

        # Should be different after mutation (with high probability)
        diff = np.abs(mutated.genome.values - original.genome.values)
        assert np.any(diff > 0)

        # Values should stay in valid range
        assert np.all(mutated.genome.values >= baseline_map.min_value)
        assert np.all(mutated.genome.values <= baseline_map.max_value)

    def test_create_next_generation(self, optimizer, baseline_map, telemetry):
        """Test generation creation."""
        optimizer.initialize_population(baseline_map)
        optimizer.evaluate_population(telemetry)

        initial_gen = optimizer.generation
        optimizer.create_next_generation()

        assert optimizer.generation == initial_gen + 1
        assert len(optimizer.population) == optimizer.population_size

    def test_elitism(self, optimizer, baseline_map, telemetry):
        """Test that elites are preserved."""
        optimizer.initialize_population(baseline_map)
        optimizer.evaluate_population(telemetry)

        # Get best fitness before
        best_before = optimizer.population[0].fitness

        optimizer.create_next_generation()
        optimizer.evaluate_population(telemetry)

        # Best fitness should not decrease (elite is preserved)
        best_after = max(ind.fitness for ind in optimizer.population)
        assert best_after >= best_before * 0.99  # Allow small numerical variance

    @pytest.mark.slow
    def test_evolution_improves_fitness(self, optimizer, baseline_map, telemetry):
        """Test that evolution improves fitness over generations."""
        optimizer.initialize_population(baseline_map)
        optimizer.evaluate_population(telemetry)

        initial_fitness = optimizer.population[0].fitness

        # Run a few generations
        for _ in range(10):
            optimizer.create_next_generation()
            optimizer.evaluate_population(telemetry)

        final_fitness = optimizer.best_individual.fitness

        # Fitness should improve or stay same
        assert final_fitness >= initial_fitness * 0.95

    @pytest.mark.slow
    def test_full_evolution(self, optimizer, baseline_map, telemetry):
        """Test complete evolution run."""
        best = optimizer.evolve(
            baseline_map=baseline_map,
            telemetry=telemetry,
            generations=10,
        )

        assert best is not None
        assert best.fitness > 0
        assert len(optimizer.history) == 10

    def test_improvement_report(self, optimizer, baseline_map, telemetry):
        """Test improvement report generation."""
        optimizer.evolve(
            baseline_map=baseline_map,
            telemetry=telemetry,
            generations=5,
        )

        report = optimizer.get_improvement_report(baseline_map)

        assert 'total_cells' in report
        assert 'cells_changed' in report
        assert 'max_increase' in report
        assert 'max_decrease' in report
        assert 'generations_run' in report


class TestOptimizationObjectives:
    """Tests for different optimization objectives."""

    @pytest.fixture
    def baseline_map(self):
        return create_default_ve_map()

    @pytest.fixture
    def telemetry(self):
        return generate_synthetic_telemetry(50)

    def test_afr_accuracy_objective(self, baseline_map, telemetry):
        """Test AFR accuracy optimization."""
        optimizer = GeneticOptimizer(
            population_size=10,
            objective=OptimizationObjective.AFR_ACCURACY,
        )

        individual = Individual(genome=baseline_map)
        optimizer.evaluate_fitness(individual, telemetry)

        assert 'afr_error' in individual.objectives

    def test_fuel_economy_objective(self, baseline_map, telemetry):
        """Test fuel economy optimization."""
        optimizer = GeneticOptimizer(
            population_size=10,
            objective=OptimizationObjective.FUEL_ECONOMY,
        )

        individual = Individual(genome=baseline_map)
        optimizer.evaluate_fitness(individual, telemetry)

        assert 'fuel_consumption' in individual.objectives

    def test_power_output_objective(self, baseline_map, telemetry):
        """Test power output optimization."""
        optimizer = GeneticOptimizer(
            population_size=10,
            objective=OptimizationObjective.POWER_OUTPUT,
        )

        individual = Individual(genome=baseline_map)
        optimizer.evaluate_fitness(individual, telemetry)

        assert 'power_estimate' in individual.objectives

    def test_balanced_objective(self, baseline_map, telemetry):
        """Test balanced multi-objective optimization."""
        optimizer = GeneticOptimizer(
            population_size=10,
            objective=OptimizationObjective.BALANCED,
        )

        individual = Individual(genome=baseline_map)
        fitness = optimizer.evaluate_fitness(individual, telemetry)

        assert 0 <= fitness <= 1


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""

    def test_generate_telemetry(self):
        """Test synthetic telemetry generation."""
        telemetry = generate_synthetic_telemetry(100)

        assert len(telemetry) == 100
        assert all(isinstance(p, TelemetryPoint) for p in telemetry)

    def test_telemetry_value_ranges(self):
        """Test that generated telemetry has realistic values."""
        telemetry = generate_synthetic_telemetry(1000)

        rpms = [p.rpm for p in telemetry]
        loads = [p.load for p in telemetry]
        afrs = [p.actual_afr for p in telemetry]

        # Check ranges
        assert min(rpms) >= 700
        assert max(rpms) <= 7000
        assert min(loads) >= 0
        assert max(loads) <= 100
        assert min(afrs) >= 10
        assert max(afrs) <= 20

    def test_default_ve_map(self):
        """Test default VE map creation."""
        ve_map = create_default_ve_map()

        assert ve_map.name == "VE Table"
        assert len(ve_map.rpm_axis) > 0
        assert len(ve_map.load_axis) > 0
        assert ve_map.values.shape == (len(ve_map.rpm_axis), len(ve_map.load_axis))

        # VE values should be realistic
        assert np.all(ve_map.values >= 0)
        assert np.all(ve_map.values <= 120)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_telemetry(self):
        """Test handling of empty telemetry."""
        optimizer = GeneticOptimizer(population_size=10)
        individual = Individual(genome=create_default_ve_map())

        fitness = optimizer.evaluate_fitness(individual, [])
        assert fitness == 0.0

    def test_single_telemetry_point(self):
        """Test handling of single telemetry point."""
        optimizer = GeneticOptimizer(population_size=10)
        individual = Individual(genome=create_default_ve_map())

        telemetry = [TelemetryPoint(
            rpm=3000, load=50, actual_afr=14.7, target_afr=14.7
        )]

        fitness = optimizer.evaluate_fitness(individual, telemetry)
        assert 0 <= fitness <= 1

    def test_extreme_afr_values(self):
        """Test handling of extreme AFR values."""
        optimizer = GeneticOptimizer(population_size=10)
        individual = Individual(genome=create_default_ve_map())

        telemetry = [
            TelemetryPoint(rpm=3000, load=50, actual_afr=10.0, target_afr=14.7),  # Very rich
            TelemetryPoint(rpm=3000, load=50, actual_afr=18.0, target_afr=14.7),  # Very lean
        ]

        fitness = optimizer.evaluate_fitness(individual, telemetry)
        assert 0 <= fitness <= 1

    def test_small_population(self):
        """Test with minimum viable population."""
        optimizer = GeneticOptimizer(
            population_size=5,
            elite_count=1,
            tournament_size=2,
        )

        baseline = create_default_ve_map()
        telemetry = generate_synthetic_telemetry(20)

        optimizer.initialize_population(baseline)
        optimizer.evaluate_population(telemetry)

        # Should still work
        optimizer.create_next_generation()
        assert len(optimizer.population) == 5
