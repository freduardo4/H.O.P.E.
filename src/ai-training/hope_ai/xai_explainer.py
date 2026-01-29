import numpy as np
import pandas as pd
import torch
import shap
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys

# Optional imports for XAI
try:
    import shap
except ImportError:
    shap = None

try:
    import lime
    import lime.lime_tabular
except ImportError:
    lime = None
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .inference import AnomalyDetector

logger = logging.getLogger(__name__)

class XAIExplainer:
    """
    Provides explainability for anomaly detection using SHAP and LIME.
    Calculates feature contributions to identify why a specific data sequence was 
    flagged as anomalous.
    """

    def __init__(self, detector: AnomalyDetector):
        self.detector = detector
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Load feature names if available in model config
        if detector.config and 'feature_names' in detector.config:
            self.feature_names = detector.config['feature_names']
        else:
            from .config import OBD2_FEATURES
            self.feature_names = OBD2_FEATURES or []

    def _prepare_shap(self, background_data: np.ndarray):
        """
        Initializes the SHAP KernelExplainer with background 'normal' data.
        """
        if shap is None:
            logger.warning("SHAP is not installed. Explainer will not function.")
            return

        def predict_score(x):
            seq_len = self.detector.config.get('sequence_length', 60)
            n_features = self.detector.config.get('n_features', len(self.feature_names))
            
            x_reshaped = x.reshape(-1, seq_len, n_features)
            _, scores = self.detector.predict(x_reshaped)
            return scores

<<<<<<< HEAD
        background_flat = background_data.reshape(background_data.shape[0], -1)
        self.shap_explainer = shap.KernelExplainer(predict_score, background_flat)
        logger.info("SHAP explainer initialized with background data.")

    def _prepare_lime(self, background_data: np.ndarray):
        """
        Initializes the LIME TabularExplainer.
        """
        if lime is None:
            logger.warning("LIME is not installed. Explainer will not function.")
            return

        background_flat = background_data.reshape(background_data.shape[0], -1)
        
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            background_flat,
            feature_names=[f"{f}_t{i}" for i in range(background_data.shape[1]) for f in (self.feature_names or [])],
            class_names=['anomaly_score'],
            mode='regression'
        )
        logger.info("LIME explainer initialized.")

    def explain_shap(self, 
                    anomaly_sequence: np.ndarray, 
                    background_data: np.ndarray,
                    nsamples: int = 100) -> Dict[str, Any]:
        """
        Calculates SHAP values for an anomalous sequence.
        """
        if shap is None:
            return {"method": "SHAP", "error": "SHAP library not installed"}

        if self.shap_explainer is None:
            self._prepare_shap(background_data)
        
        if self.shap_explainer is None:
            raise RuntimeError("Failed to initialize SHAP explainer")

        sequence_flat = anomaly_sequence.reshape(1, -1)
        shap_values = self.shap_explainer.shap_values(sequence_flat, nsamples=nsamples)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        shap_reshaped = shap_values.reshape(anomaly_sequence.shape)
        feature_contributions = np.mean(np.abs(shap_reshaped), axis=0)
        
        total = np.sum(feature_contributions)
        contribution_pct = (feature_contributions / (total + 1e-9)) * 100
        
        ranking = []
        for i, name in enumerate(self.feature_names):
            ranking.append({
                "feature": name,
                "importance": float(feature_contributions[i]),
                "contribution_pct": float(contribution_pct[i])
            })
            
        ranking = sorted(ranking, key=lambda x: x["importance"], reverse=True)

        return {
            "method": "SHAP",
            "ranking": ranking,
            "summary": f"Top contributor (SHAP): {ranking[0]['feature']} ({ranking[0]['contribution_pct']:.1f}%)"
        }

    def explain_lime(self, 
                    anomaly_sequence: np.ndarray, 
                    background_data: np.ndarray) -> Dict[str, Any]:
        """
        Calculates LIME explanations for an anomalous sequence.
        """
        if lime is None:
            return {"method": "LIME", "error": "LIME library not installed"}

        if self.lime_explainer is None:
            self._prepare_lime(background_data)

        if self.lime_explainer is None:
            raise RuntimeError("Failed to initialize LIME explainer")

        def predict_score(x):
            seq_len = self.detector.config.get('sequence_length', 60)
            n_features = self.detector.config.get('n_features', len(self.feature_names))
            x_reshaped = x.reshape(-1, seq_len, n_features)
            _, scores = self.detector.predict(x_reshaped)
            return scores

        sequence_flat = anomaly_sequence.flatten()
        exp = self.lime_explainer.explain_instance(
            sequence_flat, 
            predict_score,
            num_features=10
        )

        # Process LIME explanation to aggregate by feature across time
        lime_map = exp.as_list()
        aggregated_importance = {name: 0.0 for name in self.feature_names}
        
        for feat_time, val in lime_map:
            for name in self.feature_names:
                if name in feat_time:
                    aggregated_importance[name] += abs(val)

        total = sum(aggregated_importance.values())
        ranking = []
        for name, imp in aggregated_importance.items():
            ranking.append({
                "feature": name,
                "importance": imp,
                "contribution_pct": (imp / (total + 1e-9)) * 100
            })

        ranking = sorted(ranking, key=lambda x: x["importance"], reverse=True)

        return {
            "method": "LIME",
            "ranking": ranking,
            "summary": f"Top contributor (LIME): {ranking[0]['feature']} ({ranking[0]['contribution_pct']:.1f}%)"
        }

    def generate_narrative(self, data: Dict[str, Any], method: str = "SHAP") -> str:
        """
        Generates a human-readable diagnostic narrative.
        """
        # Convert input data to numpy
        sequence = np.array(data['sequence'])
        
        # In a real scenario, we'd need background data. 
        # For this mock/demo, we'll generate some noise.
        background = np.random.normal(0, 1, (10, *sequence.shape))
        
        if method.upper() == "SHAP":
            explanation = self.explain_shap(sequence, background)
        else:
            explanation = self.explain_lime(sequence, background)
            
        if "error" in explanation:
             return f"Analysis Failed: {explanation['error']}"

        top_feat = explanation['ranking'][0]
        narrative = f"Diagnostic Analysis ({explanation['method']}):\n"
        narrative += f"Anomaly detected with high confidence.\n"
        narrative += f"Primary cause identified as '{top_feat['feature']}', "
        narrative += f"contributing {top_feat['contribution_pct']:.1f}% to the anomaly score.\n"
        
        if top_feat['contribution_pct'] > 40:
            narrative += "Action recommended: Inspect sensor wiring and verify calibration for this PID."
        else:
            narrative += "Action recommended: Monitor multiple related systems for intermittent faults."
            
        return narrative
if __name__ == "__main__":
    # Test stub (would require a loaded model)
    pass
