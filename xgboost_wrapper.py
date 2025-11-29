import numpy as np
import pandas as pd
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",  # Fixed typo
    handlers=[
        logging.FileHandler("logs/xgboost.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_fibonacci(fib_str, price):
    """Parse and normalize Fibonacci levels (format: '0.236:100.0,0.382:105.0,...')"""
    try:
        if pd.isna(fib_str) or not fib_str:
            return [0] * 5
        levels = [float(v.split(':')[1]) for v in fib_str.split(',')[:5]]
        normalized = [level / price if price != 0 else 0 for level in levels]
        return normalized + [0] * (5 - len(normalized))
    except Exception as e:
        logger.error(f"Error parsing Fibonacci: {e}")
        return [0] * 5

def predict(features):
    """Predict using the XGBoost model via ml_predictor."""
    from ml_predictor import predict_with_all_models  # Moved inside function
    predictions = predict_with_all_models(features)
    return predictions['XGBoost']