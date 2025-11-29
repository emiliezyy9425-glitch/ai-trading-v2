"""Lightweight stub implementations for model fallbacks.

These classes mimic the interfaces of the production models closely enough
for the inference pipeline and unit tests.  They intentionally produce
stable, deterministic predictions so that downstream behaviour is
repeatable when real artefacts are unavailable.
"""

from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace


def _uniform_probabilities(batch_size: int, *, buy: float = 0.4, sell: float = 0.3, hold: float = 0.3) -> np.ndarray:
    """Return a ``(batch_size, 3)`` array of class probabilities."""

    probs = np.array([buy, sell, hold], dtype=float)
    probs = probs / probs.sum()
    return np.tile(probs, (batch_size, 1))


class StubTreeModel:
    """Simple estimator with a ``predict_proba`` method."""

    def __init__(self, feature_names: list[str]):
        self.n_features_in_ = len(feature_names)
        self.feature_names_in_ = list(feature_names)
        self.feature_name_ = list(feature_names)
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, values):
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return _uniform_probabilities(array.shape[0])


class StubLSTMModel:
    """Keras-like stub with ``predict`` and ``compile`` hooks."""

    def __init__(self, feature_dim: int):
        self.input_shape = (None, 1, feature_dim)

    def compile(self, *args, **kwargs):  # pragma: no cover - intentionally empty
        return None

    def predict(self, values, verbose: int = 0):  # pragma: no cover - simple branch
        array = np.asarray(values, dtype=float)
        if array.ndim < 3:
            array = array.reshape(array.shape[0], 1, -1)
        batch = array.shape[0]
        return _uniform_probabilities(batch)


class _StubObservationSpace:
    def __init__(self, feature_dim: int):
        self.shape = (feature_dim,)


class StubPPOModel:
    """Stable-Baselines3 compatible stub."""

    def __init__(self, feature_dim: int):
        self.observation_space = _StubObservationSpace(feature_dim)

    def predict(self, observation, deterministic: bool = True):
        batch = np.asarray(observation).shape[0]
        # ``2`` corresponds to ``Hold`` in the downstream mapping.
        actions = np.full((batch,), 2, dtype=int)
        return actions, None


class StubTransformerModel:
    """Transformers-style module returning uniform logits."""

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim

    def eval(self):  # pragma: no cover - trivial method
        return self

    def __call__(self, input_tensor):
        batch = input_tensor.shape[0]
        logits = torch.zeros((batch, 3), dtype=torch.float32)
        return SimpleNamespace(logits=logits)


__all__ = [
    "StubTreeModel",
    "StubLSTMModel",
    "StubPPOModel",
    "StubTransformerModel",
]
