"""PyTorch runtime helpers."""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional
import torch

def _likely_has_gpu() -> bool:
    """Best-effort detection for NVIDIA GPU runtime."""
    if shutil.which("nvidia-smi") is None:
        stub = Path(__file__).resolve().parent / "scripts" / "nvidia-smi"
        if stub.exists():
            os.environ["PATH"] = f"{stub.parent}{os.pathsep}{os.environ.get('PATH', '')}"
        if shutil.which("nvidia-smi") is None:
            return False

    try:
        result = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True, timeout=1)
        return result.returncode == 0 and "GPU" in (result.stdout or "")
    except Exception:
        return False

def configure_pytorch(logger: Optional[logging.Logger] = None) -> None:
    """Ensure PyTorch uses GPU if available."""
    if not _likely_has_gpu():
        if logger:
            logger.info("No NVIDIA runtime detected. PyTorch will run on CPU.")
        return

    if torch.cuda.is_available():
        if logger:
            logger.info(f"PyTorch detected {torch.cuda.device_count()} GPU(s). Using device: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True  # Auto-optimize for hardware
        torch.backends.cuda.matmul.allow_tf32 = True  # Faster matmul
    else:
        if logger:
            logger.warning("GPU detected but not available in PyTorch. Check CUDA/cuDNN install.")

def has_gpu_runtime() -> bool:
    return _likely_has_gpu() and torch.cuda.is_available()