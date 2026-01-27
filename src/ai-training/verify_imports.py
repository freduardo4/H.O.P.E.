import sys
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print("numpy imported")
except ImportError as e:
    print(f"numpy failed: {e}")

try:
    import torch
    print(f"torch imported: {torch.__version__}")
except ImportError as e:
    print(f"torch failed: {e}")

try:
    import pandas as pd
    print("pandas imported")
except ImportError as e:
    print(f"pandas failed: {e}")

print("Verification complete")
