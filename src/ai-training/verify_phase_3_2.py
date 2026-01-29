import subprocess
import json
import numpy as np
import os

def run_explain_cli(input_data):
    process = subprocess.Popen(
        ['python', 'scripts/explain_cli.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(input=json.dumps(input_data))
    
    if stderr:
        print("STDERR:", stderr)
    
    return json.loads(stdout)

def main():
    # Setup mock data for verification
    # Using a 10x4 sequence (10 timesteps, 4 features)
    anomaly_seq = np.random.randn(10, 4).tolist()
    # Add a huge spike in feature 0 to make it anomalous
    anomaly_seq[5][0] = 5.0 
    
    background_data = np.random.randn(20, 10, 4).tolist()
    
    # We need a dummy model path or we skip the actual detector load in Test 
    # For now, let's just test the CLI interface logic with a mock model path
    # If a real model is needed, we'd need to train one first.
    
    test_input = {
        "model_path": "../models/lstm_autoencoder_latest", # Placeholder
        "anomaly_sequence": anomaly_seq,
        "background_data": background_data,
        "feature_names": ["MAF", "RPM", "Load", "Temp"],
        "nsamples": 50
    }
    
    print(">>> Testing Explain CLI (expecting error if model doesn't exist, but checking structure)")
    # Since we don't have a real model in this specific test environment easily, 
    # we expect it might fail if it tries to load the model.
    # However, let's see if we can at least verify the script is locatable and runnable.
    
    try:
        result = run_explain_cli(test_input)
        print(f"Status: {result.get('status')}")
        if result.get('status') == 'success':
            print("Explanation generated successfully!")
            print(result['explanation']['summary'])
        else:
            print(f"Expected failure/info: {result.get('message')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
