import numpy as np
from enum import Enum
from typing import Dict, Tuple

class MapType(Enum):
    VE_TABLE = "VE Volumetric Efficiency"
    IGNITION_TABLE = "Ignition Timing"
    AFR_TARGET = "Target AFR"
    BOOST_TARGET = "Boost Target"
    UNKNOWN = "Unknown Map"

class MapClassifier:
    """
    Classifies ECU calibration maps based on statistical patterns and shape analysis.
    """

    @staticmethod
    def classify(values: np.ndarray) -> Tuple[MapType, float]:
        """
        Classifies the map and returns (MapType, Confidence).
        """
        if values.size == 0:
            return MapType.UNKNOWN, 0.0

        scores = {}
        
        # 1. Feature Extraction
        mean_val = np.mean(values)
        max_val = np.max(values)
        min_val = np.min(values)
        std_val = np.std(values)
        val_range = max_val - min_val
        
        # Calculate gradients (rough measure of "smoothness" and direction)
        grad_y, grad_x = np.gradient(values)
        mean_grad_x = np.mean(grad_x)
        mean_grad_y = np.mean(grad_y)

        # 2. VE Table Heuristics
        # Typical: Values 30-110, increases with RPM (X) and Load (Y), fairly smooth
        ve_score = 0
        if 40 <= mean_val <= 95: ve_score += 2
        if max_val > 100: ve_score += 1
        if mean_grad_x > 0: ve_score += 1 # Generally increases with RPM
        if mean_grad_y > 0: ve_score += 1 # Generally increases with Load
        scores[MapType.VE_TABLE] = ve_score

        # 3. Ignition Table Heuristics
        # Typical: Values 0-50, generally increases with RPM, decreases with Load
        ign_score = 0
        if 5 <= mean_val <= 30: ign_score += 2
        if max_val < 60: ign_score += 1
        if mean_grad_x > 0: ign_score += 1 # Increases with RPM
        if mean_grad_y < 0: ign_score += 2 # Decreases with Load (Crucial sign)
        scores[MapType.IGNITION_TABLE] = ign_score

        # 4. AFR Target Heuristics
        # Typical: Values 11-16, very narrow range, decreases with Load (gets richer)
        afr_score = 0
        if 12 <= mean_val <= 15: afr_score += 2
        if val_range < 6: afr_score += 2
        if 14.0 <= np.median(values) <= 15.0: afr_score += 1 # Stoich bias
        if mean_grad_y < 0: afr_score += 1 # Gets richer (lower number) as load increases
        scores[MapType.AFR_TARGET] = afr_score

        # 5. Boost Target Heuristics
        # Typical: Higher pressure (e.g. 100-250 kPa or 0-25 psi), increases with load/RPM
        boost_score = 0
        if mean_val > 100: boost_score += 1 # kPa
        if val_range > 50: boost_score += 1
        scores[MapType.BOOST_TARGET] = boost_score

        # Determine best match
        best_type = max(scores, key=scores.get)
        total_possible = 5 # max possible score for any heuristic
        confidence = min(1.0, scores[best_type] / total_possible)

        if scores[best_type] < 2:
            return MapType.UNKNOWN, 0.0

        return best_type, confidence

if __name__ == "__main__":
    # Test with mockup data
    ve = np.fromfunction(lambda r, c: 50 + (r*2) + (c*3), (16, 16))
    ign = np.fromfunction(lambda r, c: 30 - (r*1.5) + (c*1), (16, 16)) 
    afr = np.full((16,16), 14.7)
    afr[10:, :] = 12.5 # richer at high load
    
    for m, name in [(ve, "VE"), (ign, "Ignition"), (afr, "AFR")]:
        t, c = MapClassifier.classify(m)
        print(f"Map {name} classified as: {t.value} (Confidence: {c:.2f})")
