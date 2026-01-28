import sys
from pathlib import Path
import numpy as np
import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from genetic_optimizer import (
    GeneticOptimizer,
    create_default_ve_map,
    generate_synthetic_telemetry,
    OptimizationObjective
)

def test_optimizer_convergence_regression():
    """
    Regression test: Verify that the genetic optimizer consistently 
    reduces error over generations on a stable synthetic dataset.
    """
    # Fix random seed for regression stability
    np.random.seed(42)
    
    optimizer = GeneticOptimizer(
        population_size=40,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elite_count=3,
        objective=OptimizationObjective.AFR_ACCURACY
    )
    
    baseline_map = create_default_ve_map()
    telemetry = generate_synthetic_telemetry(200)
    
    # Run evolution
    generations = 40
    best = optimizer.evolve(
        baseline_map=baseline_map,
        telemetry=telemetry,
        generations=generations,
    )
    
    # Assertions for regression
    initial_fitness = optimizer.history[0]['best_fitness']
    final_fitness = optimizer.history[-1]['best_fitness']
    
    # 1. Fitness should significantly improve
    improvement = final_fitness - initial_fitness
    print(f"Improvement: {improvement}")
    assert improvement > 0.04, f"Expected significant fitness improvement, got {improvement}"
    
    # 2. AFR error should decrease
    initial_error = optimizer.history[0]['best_afr_error']
    final_error = optimizer.history[-1]['best_afr_error']
    print(f"Error: {initial_error} -> {final_error}")
    assert final_error < initial_error, f"Expected error to decrease, but it went from {initial_error} to {final_error}"
    
    # 3. Best individual should have a high score (convergence check)
    print(f"Final fitness: {best.fitness}")
    assert best.fitness > 0.4, f"Expected convergence to fitness > 0.4, got {best.fitness}"

def test_optimizer_multi_objective_regression():
    """
    Regression test: Verify that the balanced objective actually 
    considers multiple factors.
    """
    np.random.seed(42)
    
    optimizer_balanced = GeneticOptimizer(
        population_size=20,
        objective=OptimizationObjective.BALANCED
    )
    
    baseline_map = create_default_ve_map()
    telemetry = generate_synthetic_telemetry(50)
    
    best = optimizer_balanced.evolve(
        baseline_map=baseline_map,
        telemetry=telemetry,
        generations=5
    )
    
    # Check that objectives are tracked
    assert 'fuel_consumption' in best.objectives
    assert 'power_estimate' in best.objectives
    assert 'afr_error' in best.objectives
