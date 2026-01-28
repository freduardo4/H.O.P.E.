
import sys
import json
import numpy as np
import os

# Add parent directory to path so we can import hope_ai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hope_ai.tuning.genetic_optimizer import TuneOptimizer

def main():
    try:
        # Read input JSON from stdin
        input_data = json.load(sys.stdin)
        
        current_map = np.array(input_data['current_map'])
        target_map = np.array(input_data['target_map'])
        
        # If mask is not provided, assume full optimization (ones)
        if 'datalog_mask' in input_data and input_data['datalog_mask']:
            datalog_mask = np.array(input_data['datalog_mask'])
        else:
            datalog_mask = np.ones_like(current_map)
            
        config = input_data.get('config', {})
        generations = config.get('generations', 20)
        pop_size = config.get('population_size', 50)
        
        optimizer = TuneOptimizer(population_size=pop_size)
        
        best_map, history = optimizer.evolve(current_map, datalog_mask, target_map, generations=generations)
        
        output = {
            "optimized_map": best_map.tolist(),
            "fitness_history": history,
            "status": "success"
        }
        
        print(json.dumps(output))
        
    except Exception as e:
        error_output = {
            "status": "error",
            "message": str(e)
        }
        print(json.dumps(error_output))
        sys.exit(1)

if __name__ == "__main__":
    main()
