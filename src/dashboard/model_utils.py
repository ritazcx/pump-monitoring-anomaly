import numpy as np
import pandas as pd
import streamlit as st
from feature_engineering import add_engineered_features

DATA_PATH = "data/pump_sensor_data.csv"
MODEL_PATH = "model/gaussian_model_phase3_v3.npz"


def gaussian_anomaly_score(df, features, mu, var):
    # Vectorized computation instead of iterrows()
    X = df[features].values  # Shape: (n_samples, n_features)
    
    # Compute probability for each feature (vectorized)
    # Shape: (n_samples, n_features)
    probs = (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(
        -((X - mu) ** 2) / (2 * var)
    )
    
    # Product across features for each sample
    scores = np.prod(probs, axis=1)
    
    return scores


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = add_engineered_features(df)
    return df


@st.cache_resource
def load_model():
    model = np.load(MODEL_PATH, allow_pickle=True)
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

    # simple smoothing to reduce isolated false alarms
    df["anomaly_flag"] = (
        df["anomaly_flag_raw"]
        .rolling(3, min_periods=1)
        .max()
        .astype(bool)
    )

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