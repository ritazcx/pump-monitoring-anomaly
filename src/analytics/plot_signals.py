import matplotlib.pyplot as plt
import pandas as pd

from common.paths import DATA_PROCESSED_DIR, DATA_RAW_DIR, RAW_SENSOR_FILE

# -----------------------------
# 1. Choose input file here
# -----------------------------
processed_path = DATA_PROCESSED_DIR / RAW_SENSOR_FILE
raw_path = DATA_RAW_DIR / RAW_SENSOR_FILE
file_path = processed_path if processed_path.exists() else raw_path

# -----------------------------
# 2. Read data
# -----------------------------
df = pd.read_csv(file_path)

# Convert timestamp column to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Check whether anomaly labels exist
has_label = "label" in df.columns
has_anomalies = has_label and (df["label"] == 1).any()

# -----------------------------
# 3. Plot setup
# -----------------------------
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

signals = [
    ("temperature", "Temperature", "°C"),
    ("vibration", "Vibration", "mm/s"),
    ("pressure", "Pressure", "bar"),
    ("flow_rate", "Flow Rate", "L/min"),
    ("power", "Power", "kW"),
]

for ax, (col, title, ylabel) in zip(axes, signals):
    ax.plot(df["timestamp"], df[col], label=title, linewidth=1)

    # If anomaly labels exist, highlight anomaly points
    if has_anomalies:
        anomaly_points = df[df["label"] == 1]
        ax.scatter(
            anomaly_points["timestamp"],
            anomaly_points[col],
            s=10,
            label="Anomaly"
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Timestamp")
plt.tight_layout()
plt.show()
