import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class AuditIssue:
    severity: str # "CRITICAL", "WARNING", "INFO"
    category: str
    message: str
    location: tuple = None # (row, col)

class TuningAuditor:
    """
    Analyzes calibration maps for logical conflicts, safety violations and 
    excessive roughness.
    """

    def audit_map(self, name: str, values: np.ndarray, map_type: str = "VE") -> List[AuditIssue]:
        issues = []
        
        # 1. Smoothness Check (Gradient spikes)
        self._check_smoothness(values, issues)
        
        # 2. Limit Checks (Global bounds)
        self._check_limits(values, map_type, issues)
        
        # 3. Monotonicity (Curve sanity)
        self._check_monotonicity(values, map_type, issues)

        return issues

    def _check_smoothness(self, values: np.ndarray, issues: List[AuditIssue]):
        """Detects sudden jumps between adjacent cells."""
        rows, cols = values.shape
        for r in range(rows):
            for c in range(cols):
                val = values[r, c]
                # Check neighbors
                for dr, dc in [(0, 1), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if nr < rows and nc < cols:
                        nval = values[nr, nc]
                        diff_pct = abs(val - nval) / (max(abs(val), 1.0))
                        
                        if diff_pct > 0.4: # 40% jump is very suspicious
                            issues.append(AuditIssue(
                                "CRITICAL", "Smoothness", 
                                f"Massive jump ({diff_pct:.1%}) between adjacent cells.", (r, c)
                            ))
                        elif diff_pct > 0.2:
                            issues.append(AuditIssue(
                                "WARNING", "Smoothness", 
                                f"Significant step ({diff_pct:.1%}) may cause unstable control.", (r, c)
                            ))

    def _check_limits(self, values: np.ndarray, map_type: str, issues: List[AuditIssue]):
        """Checks for dangerous absolute values."""
        if "AFR" in map_type.upper():
            if np.any(values > 17.0):
                issues.append(AuditIssue("CRITICAL", "Safety", "Lean AFR target (>17.0) detected. Risk of engine damage."))
            if np.any(values < 10.0):
                issues.append(AuditIssue("WARNING", "Safety", "Extremely rich AFR target (<10.0) detected. Risk of bore wash."))
        
        if "IGNITION" in map_type.upper():
            if np.any(values > 55.0):
                issues.append(AuditIssue("CRITICAL", "Safety", "Ignition advance > 55 deg is physically improbable for most engines."))
            if np.any(values < -10.0):
                issues.append(AuditIssue("WARNING", "Safety", "Excessive ignition retard (< -10 deg) may cause high EGTs."))

    def _check_monotonicity(self, values: np.ndarray, map_type: str, issues: List[AuditIssue]):
        """Checks if curves follow expected physical trends."""
        if "IGNITION" in map_type.upper():
            # Ignition should generally decrease with load (rows)
            # Find average column slope relative to rows
            grad_y = np.gradient(values, axis=0) # Change per row
            if np.mean(grad_y) > 2.0: # Positive gradient for ignition vs load is suspicious
                issues.append(AuditIssue(
                    "WARNING", "Logic", 
                    "Ignition timing increases with load. Check for inverted axis or logic error."
                ))

if __name__ == "__main__":
    # Test with bad data
    bad_map = np.zeros((8, 8))
    bad_map[4, 4] = 100 # Spike
    
    auditor = TuningAuditor()
    results = auditor.audit_map("Test Map", bad_map, "VE")
    
    for issue in results:
        print(f"[{issue.severity}] {issue.category}: {issue.message} at {issue.location}")
