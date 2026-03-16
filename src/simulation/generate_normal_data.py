import numpy as np
import pandas as pd

from common.paths import DATA_RAW_DIR, NORMAL_SENSOR_FILE

# -----------------------------
# 1. Reproducibility
# -----------------------------
np.random.seed(42)

# -----------------------------
# 2. Basic configuration
# -----------------------------
n_rows = 5000
pump_id = "P-101"
start_time = "2026-01-01 00:00:00"

output_file = DATA_RAW_DIR / NORMAL_SENSOR_FILE

# Create timestamp column (1-minute interval)
timestamps = pd.date_range(start=start_time, periods=n_rows, freq="min")

# -----------------------------
# 3. Generate normal flow first
# -----------------------------
# Flow is treated as one core operational driver.
# Add slow sinusoidal variation + random noise to make it look realistic.
time_index = np.arange(n_rows)

flow_rate = (
    100
    + 3.0 * np.sin(2 * np.pi * time_index / 800)   # slow operating cycle
    + np.random.normal(0, 1.8, n_rows)             # random noise
)

# -----------------------------
# 4. Generate power from flow
# -----------------------------
# Higher flow -> slightly higher power
power = (
    12.0
    + 0.03 * (flow_rate - 100)
    + np.random.normal(0, 0.20, n_rows)
)

# -----------------------------
# 5. Generate temperature from power
# -----------------------------
# Higher power -> slightly higher temperature
temperature = (
    65.0
    + 0.35 * (power - 12.0)
    + np.random.normal(0, 0.65, n_rows)
)

# -----------------------------
# 6. Generate pressure from flow
# -----------------------------
# Mild relationship with flow
pressure = (
    5.0
    + 0.01 * (flow_rate - 100)
    + np.random.normal(0, 0.07, n_rows)
)

# -----------------------------
# 7. Generate vibration
# -----------------------------
# Mostly stable, lightly noisy, weakly related to power
vibration = (
    2.2
    + 0.04 * (power - 12.0)
    + np.random.normal(0, 0.10, n_rows)
)

# -----------------------------
# 8. Optional clipping
# -----------------------------
# Prevent physically silly negative or extreme values caused by randomness.
flow_rate = np.clip(flow_rate, 90, 110)
power = np.clip(power, 11.0, 13.2)
temperature = np.clip(temperature, 62, 68)
pressure = np.clip(pressure, 4.7, 5.3)
vibration = np.clip(vibration, 1.8, 2.6)

# -----------------------------
# 9. Build DataFrame
# -----------------------------
df = pd.DataFrame({
    "timestamp": timestamps,
    "pump_id": pump_id,
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "flow_rate": flow_rate,
    "power": power,
    "label": 0,
    "anomaly_type": "normal"
})

# -----------------------------
# 10. Preview
# -----------------------------
print(df.head())
print("\nShape:", df.shape)
print("\nSummary statistics:")
print(df[["temperature", "vibration", "pressure", "flow_rate", "power"]].describe())

# -----------------------------
# 11. Save CSV
# -----------------------------
df.to_csv(output_file, index=False)
print(f"\nSaved file: {output_file}")
