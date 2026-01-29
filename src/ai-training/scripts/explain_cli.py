import sys
import json
import numpy as np
import os
from pathlib import Path

# Add parent directory to path so we can import hope_ai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hope_ai.xai_explainer import XAIExplainer
from scripts.inference import AnomalyDetector

def main():
    try:
        # Read input JSON from stdin
        # Expected format: 
        # {
        #   "model_path": "path/to/model",
        #   "anomaly_sequence": [...], 
        #   "background_data": [[...], [...]], 
        #   "feature_names": ["MAF", "RPM", ...]
        # }
        input_data = json.load(sys.stdin)
        
        model_path = input_data.get('model_path')
        if not model_path:
            raise ValueError("model_path is required")
            
        anomaly_seq = np.array(input_data['anomaly_sequence'], dtype=np.float32)
        background = np.array(input_data['background_data'], dtype=np.float32)
        feature_names = input_data.get('feature_names')
        
        # Initialize
        detector = AnomalyDetector(model_path)
        if feature_names:
            detector.config['feature_names'] = feature_names
            
        explainer = XAIExplainer(detector)
        
        # Explain
        nsamples = input_data.get('nsamples', 100)
        result = explainer.explain(anomaly_seq, background, nsamples=nsamples)
        
        output = {
            "status": "success",
            "explanation": result
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
