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
    """Vectorized issue inference for better performance."""
    issue = pd.Series("unknown", index=df.index)
    
    # bearing_wear: high vibration + high temperature
    issue[(df["vibration"] > 2.8) & (df["temperature"] > 67)] = "bearing_wear"
    
    # cavitation: high vibration + pressure instability
    issue[(df["vibration"] > 2.6) & (abs(df["pressure"] - 5) > 0.3)] = "cavitation"
    
    # blockage: low flow + high power
    issue[(df["flow_rate"] < 92) & (df["power"] > 12.5)] = "blockage"
    
    # overheating: high temperature
    issue[(df["temperature"] > 70) & (issue == "unknown")] = "overheating"
    
    return issue


@st.cache_data
def add_issue_patterns(df):
    """Cache the issue pattern computation to avoid recomputation on slider changes."""
    df = df.copy()
    df["issue_pattern"] = infer_issue_vectorized(df)
    return df