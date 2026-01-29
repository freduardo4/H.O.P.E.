# Map Diff Tool

The Map Diff Tool is a critical component for visualizing and comparing ECU calibration changes.

## 1. Overview
ECU maps (Fuel, Ignition, Boost) are multi-dimensional arrays. Visually comparing raw hex values is error-prone. The Map Diff tool provides:
- **3D Surface Comparison**: Visualizes "spikes" or "dips" that could indicate tuning errors.
- **Heatmap Deltas**: Colors cells by percentage change (Green = Leaner/Safe, Red = Richer/Aggressive).

## 2. Usage
1.  Load **Source Calibration** (e.g., Stock).
2.  Load **Target Calibration** (e.g., Stage 1).
3.  Select the map (e.g., `VE_Main` or `Ignition_Base`).
4.  Toggle **Diff View** to see the exact offset between the two files.

## 3. Automated Safety
The tool automatically flags:
- Values exceeding sensor limits (e.g., MAP > 300kPa).
- Non-monotonic axis breakpoints.
- Large gradients (>15% change in adjacent cells) which can cause knock.
