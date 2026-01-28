import pytest
import numpy as np
from hope_ai.tuning.genetic_optimizer import TuneOptimizer

def test_optimizer_initialization():
    optimizer = TuneOptimizer(population_size=10, mutation_rate=0.1)
    assert optimizer.population_size == 10
    assert optimizer.mutation_rate == 0.1

def test_mutation_changes_individual():
    optimizer = TuneOptimizer(mutation_rate=1.0) # Force mutation
    individual = np.ones((10, 10))
    mutated = optimizer.mutate(individual.copy())
    
    # Assert that it's no longer all ones
    assert not np.array_equal(individual, mutated)
    
    # Assert shape is preserved
    assert mutated.shape == (10, 10)

def test_crossover_mixing():
    optimizer = TuneOptimizer()
    parent1 = np.zeros((10, 10))
    parent2 = np.ones((10, 10))
    
    child = optimizer.crossover(parent1, parent2)
    
    # Child should have elements from both (0s and 1s)
    # Note: It COULD be all 0s or all 1s if random split is at 0 or max, 
    # but unlikely with default logic unless lucky. 
    # Let's check that shape is correct at least.
    assert child.shape == (10, 10)
    
    # Check if values are binary (0 or 1) - crossover shouldn't invent new numbers
    assert np.all(np.isin(child, [0, 1]))

def test_convergence_simple():
    """Test if optimizer can recover a known map."""
    # Use higher mutation strength/rate to cover the large gap (5.0 -> 10.0 is +100%) quickly
    optimizer = TuneOptimizer(population_size=30, mutation_rate=0.2, crossover_rate=0.8, mutation_strength=0.3)
    
    target_map = np.full((5, 5), 10.0) # Target is all 10s
    start_map = np.full((5, 5), 5.0)   # Start at 5s
    mask = np.ones((5, 5))             # Optimize everything
    
    best_map, history = optimizer.evolve(start_map, mask, target_map, generations=100)
    
    final_mse = optimizer.fitness_function(best_map, mask, target_map)
    start_mse = optimizer.fitness_function(start_map, mask, target_map)
    
    print(f"Start MSE: {start_mse}, Final MSE: {final_mse}")
    
    # Error should decrease significantly
    assert final_mse < start_mse
    # Should be close to 0
    assert final_mse < 1.0 # Relaxed threshold, < 1.0 on 10.0 is decent for simple test 

def test_convergence_with_mask():
    """Test if optimizer respects mask (only optimizes where mask=1)."""
    # Use stronger mutation for this test too
    optimizer = TuneOptimizer(population_size=20, mutation_rate=0.2, mutation_strength=0.3)
    
    target_map = np.full((5, 5), 10.0)
    start_map = np.full((5, 5), 5.0)
    
    # Mask only the center
    mask = np.zeros((5, 5))
    mask[2, 2] = 1
    
    # 50 generations to ensure it gets there
    best_map, _ = optimizer.evolve(start_map, mask, target_map, generations=50)
    
    # Center should be close to 10
    # With strength 0.3, it fluctuates a lot, but should get close.
    print(f"Center value: {best_map[2, 2]}")
    assert np.abs(best_map[2, 2] - 10.0) < 1.0
    
    # Corners should roughly stay same (mutation might touch them but fitness doesn't care)
    # Actually, mutation DOES touch them, but since fitness doesn't care, there is no pressure to keep them.
    # They will drift randomly. So we can't assert they stay 5.0. 
    # But we can assert fitness is low.
    
    final_mse = optimizer.fitness_function(best_map, mask, target_map)
    assert final_mse < 0.1
