import numpy as np
import pandas as pd
import streamlit as st
from feature_engineering import add_engineered_features

DATA_PATH = "data/pump_sensor_data.csv"
MODEL_PATH = "model/gaussian_model_phase3_v3.npz"


def gaussian_anomaly_score(df, features, mu, var):
    scores = []

    for _, row in df.iterrows():
        p = 1.0

        for i, f in enumerate(features):
            x = row[f]
            sigma2 = var[i]
            mu_f = mu[i]

            prob = (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(
                -((x - mu_f) ** 2) / (2 * sigma2)
            )
            p *= prob

        scores.append(p)

    return np.array(scores)


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