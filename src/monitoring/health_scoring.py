"""Health scoring utilities for equipment monitoring."""

from __future__ import annotations

import pandas as pd

from common.config import HEALTH_SCORE_NORMAL, HEALTH_SCORE_WARNING, HEALTH_SCORE_CRITICAL


def compute_health_score(
    df: pd.DataFrame,
    anomaly_flag_col: str = "anomaly_flag",
) -> float:
    """Compute a simple health score based on anomaly rate.

    Returns a score between 0.0 and 1.0 where higher is healthier.
    """
    if df.empty or anomaly_flag_col not in df.columns:
        return HEALTH_SCORE_NORMAL

    anomaly_rate = df[anomaly_flag_col].mean()
    score = max(0.0, 1.0 - anomaly_rate)
    return score


def categorize_health_state(
    score: float,
    warning_threshold: float = 0.75,
    critical_threshold: float = 0.40,
) -> str:
    """Categorize health state based on the computed health score."""
    if score >= warning_threshold:
        return "normal"
    if score >= critical_threshold:
        return "warning"
    return "critical"
