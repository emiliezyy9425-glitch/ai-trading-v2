#!/usr/bin/env python3
import numpy as np
from pathlib import Path

y_val_path = Path("/app/models/y_val.npy")

if not y_val_path.exists():
    print("y_val.npy not found! Run at least one model first.")
else:
    y_val = np.load(y_val_path)
    unique, counts = np.unique(y_val, return_counts=True)
    total = len(y_val)
    print("y_val distribution:")
    for u, c in zip(unique, counts):
        print(f"  Class {int(u)}: {c:,} samples ({c/total:.1%})")
    
    if len(unique) < 2:
        print("\nWARNING: Only ONE class in y_val → ROC-AUC is undefined!")
    else:
        print("\nGood: Both classes are present.")
