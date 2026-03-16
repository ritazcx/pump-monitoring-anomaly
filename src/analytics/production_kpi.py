"""Production KPI calculations for smart factory monitoring."""

from __future__ import annotations

import pandas as pd


def calculate_downtime(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    anomaly_flag_col: str = "anomaly_flag",
) -> float:
    """Estimate total downtime in minutes based on anomaly flags."""
    if anomaly_flag_col not in df.columns or df.empty:
        return 0.0

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)

    # Assume row spacing is roughly constant; use time delta between measurements.
    df["delta_min"] = df[timestamp_col].diff().dt.total_seconds().div(60).fillna(0)
    # For the first row, assume the same delta as next row if available.
    if len(df) > 1 and df.at[0, "delta_min"] == 0:
        df.at[0, "delta_min"] = df.at[1, "delta_min"]

    downtime = df.loc[df[anomaly_flag_col], "delta_min"].sum()
    return float(downtime)


def calculate_utilization(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    anomaly_flag_col: str = "anomaly_flag",
) -> float:
    """Calculate utilization as percentage of time without anomalies."""
    if df.empty:
        return 0.0

    total_minutes = (pd.to_datetime(df[timestamp_col]).max() - pd.to_datetime(df[timestamp_col]).min()).total_seconds() / 60
    if total_minutes <= 0:
        return 0.0

    downtime = calculate_downtime(df, timestamp_col, anomaly_flag_col)
    utilization = max(0.0, min(1.0, (total_minutes - downtime) / total_minutes))
    return utilization


def calculate_throughput(
    df: pd.DataFrame,
    flow_col: str = "flow_rate",
) -> float:
    """Calculate throughput as the mean flow rate."""
    if flow_col not in df.columns or df.empty:
        return 0.0

    return float(df[flow_col].mean())


def calculate_oee(
    df: pd.DataFrame,
    availability: float | None = None,
    performance: float | None = None,
    quality: float | None = None,
) -> float:
    """Compute a simplified OEE (Overall Equipment Effectiveness)."""
    if availability is None or performance is None or quality is None:
        return 0.0

    return float(max(0.0, min(1.0, availability * performance * quality)))
