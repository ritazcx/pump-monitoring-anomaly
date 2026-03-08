from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# 1. Paths
# -----------------------------
project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "pump_sensor_data_normal.csv"
output_file = project_root / "data" / "pump_sensor_data.csv"

# -----------------------------
# 2. Read normal baseline data
# -----------------------------
df = pd.read_csv(input_file)
df["timestamp"] = pd.to_datetime(df["timestamp"])

np.random.seed(42)

# -----------------------------
# 3. Helper functions
# -----------------------------
def mark_anomaly(df, start_idx, end_idx, anomaly_type):
    df.loc[start_idx:end_idx, "label"] = 1
    df.loc[start_idx:end_idx, "anomaly_type"] = anomaly_type


def smooth_profile(n, peak=1.0):
    """
    Creates a smooth anomaly shape:
    ramp up -> hold -> ramp down
    """
    x = np.linspace(0, 1, n)
    profile = np.sin(np.pi * x)  # 0 -> 1 -> 0
    return peak * profile


# -----------------------------
# 4. Inject anomaly windows
# -----------------------------

# =========================================
# A. Bearing wear: 3150–3270
# Main effects:
# - vibration gradually increases
# - temperature rises slightly
# - power rises slightly
# =========================================
start, end = 3150, 3270
n = end - start + 1

wear_profile = smooth_profile(n, peak=1.0)

df.loc[start:end, "vibration"] += 0.75 * wear_profile + np.random.normal(0, 0.03, n)
df.loc[start:end, "temperature"] += 0.70 * wear_profile + np.random.normal(0, 0.05, n)
df.loc[start:end, "power"] += 0.35 * wear_profile + np.random.normal(0, 0.03, n)

mark_anomaly(df, start, end, "bearing_wear")


# =========================================
# B. Cavitation: 3500–3600
# Main effects:
# - moderate vibration spikes
# - pressure oscillates
# - flow becomes unstable
# =========================================
start, end = 3500, 3600
n = end - start + 1

oscillation = np.sin(np.linspace(0, 6 * np.pi, n))
cav_profile = smooth_profile(n, peak=1.0)

df.loc[start:end, "vibration"] += (
    0.20 * cav_profile
    + 0.20 * np.abs(oscillation)
    + np.random.normal(0, 0.04, n)
)

df.loc[start:end, "pressure"] += (
    0.10 * oscillation
    + np.random.normal(0, 0.03, n)
)

df.loc[start:end, "flow_rate"] += (
    1.80 * oscillation
    + np.random.normal(0, 0.50, n)
)

mark_anomaly(df, start, end, "cavitation")


# =========================================
# C. Blockage: 3870–3980
# Main effects:
# - flow rate drops
# - power rises slightly
# - temperature rises slightly
# =========================================
start, end = 3870, 3980
n = end - start + 1

block_profile = smooth_profile(n, peak=1.0)

df.loc[start:end, "flow_rate"] -= 7.0 * block_profile + np.random.normal(0, 0.40, n)
df.loc[start:end, "power"] += 0.70 * block_profile + np.random.normal(0, 0.04, n)
df.loc[start:end, "temperature"] += 0.50 * block_profile + np.random.normal(0, 0.04, n)
df.loc[start:end, "pressure"] += 0.08 * block_profile + np.random.normal(0, 0.02, n)

mark_anomaly(df, start, end, "blockage")


# =========================================
# D. Overheating: 4250–4360
# Main effects:
# - temperature rises clearly
# - power rises slightly
# - little vibration change
# =========================================
start, end = 4250, 4360
n = end - start + 1

heat_profile = smooth_profile(n, peak=1.0)

df.loc[start:end, "temperature"] += 3.0 * heat_profile + np.random.normal(0, 0.06, n)
df.loc[start:end, "power"] += 0.45 * heat_profile + np.random.normal(0, 0.03, n)
df.loc[start:end, "vibration"] += 0.05 * heat_profile + np.random.normal(0, 0.02, n)

mark_anomaly(df, start, end, "overheating")


# -----------------------------
# 5. Clip values to keep them plausible
# -----------------------------
df["flow_rate"] = df["flow_rate"].clip(lower=88, upper=110)
df["power"] = df["power"].clip(lower=11.0, upper=13.8)
df["temperature"] = df["temperature"].clip(lower=62, upper=71)
df["pressure"] = df["pressure"].clip(lower=4.6, upper=5.4)
df["vibration"] = df["vibration"].clip(lower=1.8, upper=3.4)

# -----------------------------
# 6. Save final dataset
# -----------------------------
df.to_csv(output_file, index=False)

print(f"Saved file: {output_file}")
print("\nLabel counts:")
print(df["label"].value_counts())

print("\nAnomaly type counts:")
print(df["anomaly_type"].value_counts())