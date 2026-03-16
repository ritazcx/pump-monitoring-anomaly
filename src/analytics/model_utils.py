import numpy as np
import pandas as pd
import streamlit as st

from common.paths import DATA_PROCESSED_DIR, DATA_RAW_DIR, MODELS_DIR, RAW_SENSOR_FILE
from pipeline.feature_engineering import add_engineered_features
from analytics.anomaly_detection import gaussian_anomaly_score, smooth_anomaly_flag


MODEL_FILE = MODELS_DIR / "gaussian_model_phase3_v3.npz"


def _default_data_path() -> str:
    """Choose processed data if available, otherwise fall back to raw."""
    processed_path = DATA_PROCESSED_DIR / RAW_SENSOR_FILE
    raw_path = DATA_RAW_DIR / RAW_SENSOR_FILE

    if processed_path.exists():
        return str(processed_path)
    return str(raw_path)


@st.cache_data
def load_data():
    data_path = _default_data_path()
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = add_engineered_features(df)
    return df


@st.cache_resource
def load_model():
    model = np.load(str(MODEL_FILE), allow_pickle=True)
    mu = model["mu"]
    var = model["var"]
    epsilon = model["epsilon"]
    features = model["feature_cols"].tolist()
    return mu, var, epsilon, features


@st.cache_data
def compute_scores(df, features, mu, var, epsilon):
    df = df.copy()
    df["anomaly_score"] = gaussian_anomaly_score(df, features, mu, var)
    df["anomaly_flag_raw"] = df["anomaly_score"] < epsilon
    df["anomaly_flag"] = smooth_anomaly_flag(df, epsilon=epsilon, window=3)
    return df


def infer_issue_vectorized(df):
    """Vectorized issue inference for better performance.

    Thresholds are aligned with the injected anomaly profiles.  Rules are
    applied sequentially and only overwrite rows that are still "unknown",
    preventing later rules from clobbering earlier matches.
    """
    issue = pd.Series("unknown", index=df.index)

    # 1. bearing_wear: strong vibration rise + temperature increase
    #    injection peaks: vibration ≈3.35, temperature ≈68.7
    mask = (df["vibration"] > 3.0) & (df["temperature"] > 67.5)
    issue.loc[mask] = "bearing_wear"

    # 2. cavitation: moderate vibration spike and pressure oscillation ±0.1
    mask = (
        (issue == "unknown")
        & (df["vibration"] > 2.6)
        & (df["pressure"].sub(5.0).abs() > 0.10)
    )
    issue.loc[mask] = "cavitation"

    # 3. blockage: flow drops below normal min and power edges upward
    mask = (
        (issue == "unknown")
        & (df["flow_rate"] < 90)
        & (df["power"] > 12.4)
    )
    issue.loc[mask] = "blockage"

    # 4. overheating: temperature climbs above 70 (only if still unknown)
    mask = (issue == "unknown") & (df["temperature"] > 70)
    issue.loc[mask] = "overheating"

    return issue


@st.cache_data
def add_issue_patterns(df):
    """Cache the issue pattern computation to avoid recomputation on slider changes."""
    df = df.copy()
    df["issue_pattern"] = infer_issue_vectorized(df)
    return df