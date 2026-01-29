
import sys
import os
from pathlib import Path
import pytest
import numpy as np

# Add scripts directory to path
# Assuming this test file is in src/ai-training/tests/
# and scripts are in src/ai-training/scripts/ Or hope_ai is in src/ai-training/hope_ai/
# We need to make sure we can import 'hope_ai.tuning.rl_guided_optimizer' or related.

# The file viewed was c:\Users\Test\Documents\H.O.P.E\src\ai-training\hope_ai\tuning\rl_guided_optimizer.py
# So we need src/ai-training in sys.path to import hope_ai

sys.path.insert(0, str(Path(__file__).parent.parent))

from hope_ai.tuning.rl_guided_optimizer import RLGuidedOptimizer

class TestRLEmissionsGuardrail:
    
    def test_guardrail_clamping(self):
        """Test that the emissions guardrail clamps values to safe range."""
        optimizer = RLGuidedOptimizer()
        
        # Create invalid map with values outside [10, 150]
        invalid_map = np.array([
            [5.0, 160.0],
            [10.0, 150.0]
        ])
        
        corrected_map = optimizer._apply_emissions_guardrail(invalid_map)
        
        # Verify clamping
        assert np.all(corrected_map >= 10.0)
        assert np.all(corrected_map <= 150.0)
        
        assert corrected_map[0, 0] == 10.0
        assert corrected_map[0, 1] == 150.0
        assert corrected_map[1, 0] == 10.0
        assert corrected_map[1, 1] == 150.0

    def test_evolution_respects_guardrail(self):
        """Test that full evolution loop respects limits."""
        optimizer = RLGuidedOptimizer(population_size=10, mutation_strength=100.0) # High mutation to force out of bounds
        
        current_map = np.full((5, 5), 50.0)
        target_map = np.full((5, 5), 80.0)
        mask = np.ones((5, 5))
        
        best_map, _ = optimizer.evolve(current_map, mask, target_map, generations=2)
        
        assert np.all(best_map >= 10.0)
        assert np.all(best_map <= 150.0)

    def test_mab_initialization(self):
        """Test Multi-Armed Bandit state initialization."""
        optimizer = RLGuidedOptimizer()
        assert len(optimizer.arms) == 4
        assert np.all(optimizer.arm_counts == 0)
        assert optimizer.epsilon == 0.2
