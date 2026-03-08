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

# Reproducibility
np.random.seed(42)

# -----------------------------
# 3. Helper function
# -----------------------------
def mark_anomaly(df, start_idx, end_idx, anomaly_type):
    df.loc[start_idx:end_idx, "label"] = 1
    df.loc[start_idx:end_idx, "anomaly_type"] = anomaly_type

# -----------------------------
# 4. Inject anomaly windows
# -----------------------------
# Final layout:
# 2300–2399 -> bearing_wear
# 2450–2529 -> cavitation
# 2600–2679 -> blockage
# 2800–2879 -> overheating

# =============================
# A. Bearing wear
# Pattern:
# - vibration gradually rises
# - temperature rises slightly
# - power rises slightly
# =============================
start, end = 2300, 2399
n = end - start + 1

vibration_ramp = np.linspace(0.3, 1.6, n)
temperature_ramp = np.linspace(0.3, 2.0, n)
power_ramp = np.linspace(0.1, 0.8, n)

df.loc[start:end, "vibration"] += vibration_ramp + np.random.normal(0, 0.05, n)
df.loc[start:end, "temperature"] += temperature_ramp + np.random.normal(0, 0.08, n)
df.loc[start:end, "power"] += power_ramp + np.random.normal(0, 0.05, n)

mark_anomaly(df, start, end, "bearing_wear")

# =============================
# B. Cavitation
# Pattern:
# - vibration spikes
# - pressure fluctuates
# - flow becomes unstable
# =============================
start, end = 2450, 2529
n = end - start + 1

# Vibration spikes
vibration_spikes = np.abs(np.random.normal(0.6, 0.35, n))

# Oscillation for flow and pressure
oscillation = np.sin(np.linspace(0, 8 * np.pi, n))

df.loc[start:end, "vibration"] += vibration_spikes
df.loc[start:end, "pressure"] += 0.18 * oscillation + np.random.normal(0, 0.05, n)
df.loc[start:end, "flow_rate"] += 2.5 * oscillation + np.random.normal(0, 0.8, n)

mark_anomaly(df, start, end, "cavitation")

# =============================
# C. Blockage
# Pattern:
# - flow rate drops
# - power increases
# - temperature rises slightly
# - pressure shifts modestly
# =============================
start, end = 2600, 2679
n = end - start + 1

flow_drop = np.linspace(8, 16, n)
power_increase = np.linspace(0.5, 1.5, n)
temp_increase = np.linspace(0.2, 1.2, n)
pressure_shift = np.linspace(0.05, 0.20, n)

df.loc[start:end, "flow_rate"] -= flow_drop + np.random.normal(0, 0.8, n)
df.loc[start:end, "power"] += power_increase + np.random.normal(0, 0.08, n)
df.loc[start:end, "temperature"] += temp_increase + np.random.normal(0, 0.08, n)
df.loc[start:end, "pressure"] += pressure_shift + np.random.normal(0, 0.04, n)

mark_anomaly(df, start, end, "blockage")

# =============================
# D. Overheating
# Pattern:
# - temperature rises clearly
# - power remains somewhat elevated
# - vibration changes little
# =============================
start, end = 2800, 2879
n = end - start + 1

temp_rise = np.linspace(2.0, 6.5, n)
power_rise = np.linspace(0.3, 1.0, n)

df.loc[start:end, "temperature"] += temp_rise + np.random.normal(0, 0.12, n)
df.loc[start:end, "power"] += power_rise + np.random.normal(0, 0.06, n)
df.loc[start:end, "vibration"] += np.random.normal(0.05, 0.03, n)

mark_anomaly(df, start, end, "overheating")

# -----------------------------
# 5. Clip values to keep them plausible
# -----------------------------
df["flow_rate"] = df["flow_rate"].clip(lower=70, upper=112)
df["power"] = df["power"].clip(lower=10.5, upper=15.5)
df["temperature"] = df["temperature"].clip(lower=61, upper=75)
df["pressure"] = df["pressure"].clip(lower=4.3, upper=5.6)
df["vibration"] = df["vibration"].clip(lower=1.7, upper=4.8)

# -----------------------------
# 6. Save final dataset
# -----------------------------
df.to_csv(output_file, index=False)

print(f"Saved file: {output_file}")
print("\nLabel counts:")
print(df["label"].value_counts())

print("\nAnomaly type counts:")
print(df["anomaly_type"].value_counts())
