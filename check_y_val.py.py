# check_y_val.py
import numpy as np
from pathlib import Path

y_val_path = Path("/app/models/y_val.npy")
if not y_val_path.exists():
    print("y_val.npy not found! Run a model first.")
else:
    y_val = np.load(y_val_path)
    unique, counts = np.unique(y_val, return_counts=True)
    print("y_val distribution:")
    for u, c in zip(unique, counts):
        print(f"  Class {int(u)}: {c} samples ({c/len(y_val):.1%})")
    
    if len(unique) == 1:
        print("WARNING: Only ONE class in y_val → AUC undefined!")
    else:
        print("Good: Both classes present.")