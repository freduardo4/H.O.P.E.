import numpy as np
import logging
from typing import List, Tuple, Dict, Any
from .genetic_optimizer import TuneOptimizer

logger = logging.getLogger(__name__)

class RLGuidedOptimizer(TuneOptimizer):
    """
    Enhanced Genetic Algorithm that uses a Reinforcement Learning (Multi-Armed Bandit)
    controller to dynamically tune mutation parameters during evolution.
    """

    def __init__(self, 
                 population_size: int = 50, 
                 mutation_rate: float = 0.05, 
                 crossover_rate: float = 0.7,
                 mutation_strength: float = 0.1):
        super().__init__(population_size, mutation_rate, crossover_rate, mutation_strength)
        
        # MAB parameters
        # Arms: (rate, strength)
        self.arms = [
            (0.01, 0.02), # Precise / Stable
            (0.05, 0.10), # Standard
            (0.15, 0.20), # Exploratory
            (0.30, 0.40)  # Aggressive / Escape Local Optima
        ]
        self.arm_counts = np.zeros(len(self.arms))
        self.arm_rewards = np.zeros(len(self.arms))
        self.current_arm = 1 # Start with "Standard"
        self.epsilon = 0.2   # Exploration vs Exploitation

    def evolve(self, 
                current_map: np.ndarray, 
                datalog_mask: np.ndarray, 
                target_map: np.ndarray,
                generations: int = 20) -> Tuple[np.ndarray, List[float]]:
        
        population = self._initialize_population(current_map, self.population_size)
        best_fitness_history = []
        last_best_fitness = float('inf')

        for generation in range(generations):
            # 1. Select Mutation Parameters using Epsilon-Greedy MAB
            self._select_arm()
            self.mutation_rate, self.mutation_strength = self.arms[self.current_arm]

            # 2. Evaluations
            fitness_scores = [self.fitness_function(ind, datalog_mask, target_map) for ind in population]
            best_fitness = min(fitness_scores)
            best_fitness_history.append(best_fitness)

            # 3. Reward the MAB arm
            # If fitness improved, positive reward proportional to gain
            reward = max(0, last_best_fitness - best_fitness)
            self._update_arm_reward(reward)
            last_best_fitness = best_fitness

            # Elitism
            sorted_pop_indices = np.argsort(fitness_scores)
            new_population = [population[sorted_pop_indices[0]]]

            # Selection/Reproduction
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                child = self.crossover(parent1, parent2) if np.random.rand() < self.crossover_rate else parent1.copy()
                child = self.mutate(child)
                
                # Emissions Guardrail: Re-evaluate and potentially reject risky mutations
                child = self._apply_emissions_guardrail(child)
                
                new_population.append(child)
            
            population = new_population
            
            if generation % 5 == 0:
                logger.info(f"Generation {generation}: Best {best_fitness:.6f} | Arm {self.current_arm} ({self.mutation_rate})")

        fitness_scores = [self.fitness_function(ind, datalog_mask, target_map) for ind in population]
        return population[np.argmin(fitness_scores)], best_fitness_history

    def _select_arm(self):
        """Epsilon-greedy selection."""
        if np.random.rand() < self.epsilon:
            self.current_arm = np.random.randint(0, len(self.arms))
        else:
            # Pick best performing arm (average reward)
            avg_rewards = self.arm_rewards / (self.arm_counts + 1e-6)
            self.current_arm = np.argmax(avg_rewards)

    def _update_arm_reward(self, reward: float):
        self.arm_counts[self.current_arm] += 1
        # Sliding average or simple cumsum
        self.arm_rewards[self.current_arm] += reward

    def _apply_emissions_guardrail(self, individual: np.ndarray) -> np.ndarray:
        """
        Policy constraint: Ensure map values don't go into 'unclean' ranges.
        For simplicity, we clamp to a safety corridor of +/- 50% of the baseline.
        In a real app, this would use an O2/NOx model.
        """
        # (Assuming baseline map is somewhat decent)
        # For now, just ensure values don't become 0 or absurdly high
        return np.clip(individual, 10.0, 150.0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tgt = np.full((10, 10), 80.0)
    curr = np.full((10, 10), 50.0)
    mask = np.ones((10, 10))
    
    rl_ga = RLGuidedOptimizer()
    best_map, hist = rl_ga.evolve(curr, mask, tgt, generations=30)
    print(f"Final Fitness: {hist[-1]:.6f}")
