from pathlib import Path
import numpy as np
import pandas as pd

def add_engineered_features(df):
    df = df.copy()

    # Avoid division by zero
    eps = 1e-6

    # # 1. Power-to-flow ratio
    # df["power_flow_ratio"] = df["power"] / (df["flow_rate"] + eps)

    # # 2. Flow-to-pressure ratio
    # df["flow_pressure_ratio"] = df["flow_rate"] / (df["pressure"] + eps)

    # Difference from previous timestamp
    df["flow_rate_diff"] = df["flow_rate"].diff().fillna(0.0)


    return df