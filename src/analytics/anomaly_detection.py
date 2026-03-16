"""Anomaly detection algorithms and scoring utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gaussian_density(
    X: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
) -> np.ndarray:
    """Compute the joint Gaussian density for each sample in X."""

    probs = (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(
        -((X - mu) ** 2) / (2 * var)
    )
    return np.prod(probs, axis=1)


def gaussian_anomaly_score(
    df: pd.DataFrame,
    features: list[str],
    mu: np.ndarray,
    var: np.ndarray,
) -> np.ndarray:
    """Compute a Gaussian-based anomaly score (joint probability) for each row."""

    X = df[features].values
    return gaussian_density(X, mu, var)


def smooth_anomaly_flag(
    df: pd.DataFrame,
    anomaly_score_col: str = "anomaly_score",
    epsilon: float | None = None,
    window: int = 3,
) -> pd.Series:
    """Compute a smoothed anomaly flag series from raw anomaly scores."""
    if epsilon is None:
        raise ValueError("epsilon must be provided to compute anomaly flags")

    raw_flag = df[anomaly_score_col] < epsilon
    # simple smoothing to reduce isolated false alarms
    return (
        raw_flag
        .rolling(window, min_periods=1)
        .max()
        .astype(bool)
    )
