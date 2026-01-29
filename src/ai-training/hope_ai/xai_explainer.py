import numpy as np
import pandas as pd
import torch
import shap
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .inference import AnomalyDetector

logger = logging.getLogger(__name__)

class XAIExplainer:
    """
    Provides explainability for anomaly detection using SHAP (SHapley Additive exPlanations).
    Calculates feature contributions to identify why a specific data sequence was 
    flagged as anomalous.
    """

    def __init__(self, detector: AnomalyDetector):
        self.detector = detector
        self.explainer = None
        self.feature_names = None
        
        # Load feature names if available in model config
        if detector.config and 'feature_names' in detector.config:
            self.feature_names = detector.config['feature_names']

    def _prepare_explainer(self, background_data: np.ndarray):
        """
        Initializes the SHAP KernelExplainer with background 'normal' data.
        """
        # Define a wrapper function for the detector's score output
        def predict_score(x):
            # SHAP might pass 2D data (samples, features) for KernelExplainer
            # We need to reshape to (samples, seq_len, features) if using sequence model
            seq_len = self.detector.config.get('sequence_length', 10)
            n_features = self.detector.config.get('n_features', 1)
            
            x_reshaped = x.reshape(-1, seq_len, n_features)
            _, scores = self.detector.predict(x_reshaped)
            return scores

        # background_data should be flattened to (n_samples, seq_len * n_features)
        # for some SHAP explainers if they don't support multi-dim directly
        background_flat = background_data.reshape(background_data.shape[0], -1)
        self.explainer = shap.KernelExplainer(predict_score, background_flat)
        logger.info("SHAP explainer initialized with background data.")

    def explain(self, 
                anomaly_sequence: np.ndarray, 
                background_data: np.ndarray,
                nsamples: int = 100) -> Dict[str, Any]:
        """
        Calculates SHAP values for an anomalous sequence.
        
        Args:
            anomaly_sequence: Data sequence of shape (seq_len, n_features)
            background_data: Normal data for reference of shape (n_samples, seq_len, n_features)
            nsamples: Number of samples for KernelExplainer approximation
            
        Returns:
            Dictionary containing SHAP values and contribution ranking.
        """
        if self.explainer is None:
            self._prepare_explainer(background_data)

        # Flatten sequence for KernelExplainer
        sequence_flat = anomaly_sequence.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(sequence_flat, nsamples=nsamples)
        
        # Reshape SHAP values back to (seq_len, n_features)
        # Handle case where shap_values might be a list (for multi-output, though here we have 1 score)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        shap_reshaped = shap_values.reshape(anomaly_sequence.shape)
        
        # Aggregate contributions across the sequence (mean absolute SHAP)
        feature_contributions = np.mean(np.abs(shap_reshaped), axis=0)
        
        # Normalize contributions to percentages
        total = np.sum(feature_contributions)
        contribution_pct = (feature_contributions / (total + 1e-9)) * 100
        
        # Prepare ranking
        if self.feature_names is None:
            self.feature_names = [f"PID_{i:02X}" for i in range(anomaly_sequence.shape[-1])]
            
        ranking = []
        for i, name in enumerate(self.feature_names):
            ranking.append({
                "feature": name,
                "importance": float(feature_contributions[i]),
                "contribution_pct": float(contribution_pct[i])
            })
            
        ranking = sorted(ranking, key=lambda x: x["importance"], reverse=True)

        return {
            "shap_values": shap_reshaped.tolist(),
            "ranking": ranking,
            "summary": f"Top contributor: {ranking[0]['feature']} ({ranking[0]['contribution_pct']:.1f}%)"
        }

if __name__ == "__main__":
    # Test stub (would require a loaded model)
    pass
