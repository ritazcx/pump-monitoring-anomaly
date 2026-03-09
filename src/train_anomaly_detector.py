from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# 1. Paths
# -----------------------------
project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "pump_sensor_data.csv"
model_file = project_root / "data" / "gaussian_model.npz"

# -----------------------------
# 2. Configuration
# -----------------------------
feature_cols = ["temperature", "vibration", "pressure", "flow_rate", "power"]

# Chronological split
train_start, train_end = 0, 2999   # inclusive

# -----------------------------
# 3. Load data
# -----------------------------
df = pd.read_csv(input_file)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# -----------------------------
# 4. Training split
# -----------------------------
train_df = df.iloc[train_start:train_end + 1].copy()

# Train only on normal rows
train_normal_df = train_df[train_df["label"] == 0].copy()

X_train = train_normal_df[feature_cols].values

# -----------------------------
# 5. Estimate Gaussian parameters
# -----------------------------
mu = np.mean(X_train, axis=0)
var = np.var(X_train, axis=0)

# Avoid divide-by-zero issues
var = np.where(var < 1e-6, 1e-6, var)

# -----------------------------
# 6. Save model
# -----------------------------
np.savez(
    model_file,
    mu=mu,
    var=var,
    feature_cols=np.array(feature_cols)
)

print(f"Model saved to: {model_file}")
print("\nFeatures:")
print(feature_cols)

print("\nMean (mu):")
for col, value in zip(feature_cols, mu):
    print(f"{col}: {value:.6f}")

print("\nVariance (var):")
for col, value in zip(feature_cols, var):
    print(f"{col}: {value:.6f}")