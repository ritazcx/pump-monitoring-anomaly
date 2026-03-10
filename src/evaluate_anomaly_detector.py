from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from feature_engineering import add_engineered_features

# -----------------------------
# 1. Paths
# -----------------------------
project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "pump_sensor_data.csv"
model_file = project_root / "data" / "gaussian_model_phase3_v3.npz"

# -----------------------------
# 2. Gaussian probability function
# -----------------------------
def gaussian_density(X, mu, var):
    coeff = 1 / np.sqrt(2 * np.pi * var)
    exp_term = np.exp(-((X - mu) ** 2) / (2 * var))
    p_per_feature = coeff * exp_term
    p_total = np.prod(p_per_feature, axis=1)
    return p_total

# -----------------------------
# 3. Select epsilon on CV
# -----------------------------
def select_epsilon(y_true, p_val):
    best_epsilon = None
    best_f1 = -1

    epsilons = np.linspace(np.min(p_val), np.max(p_val), 1000)

    for epsilon in epsilons:
        y_pred = (p_val < epsilon).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1

# -----------------------------
# 4. Reporting helper
# -----------------------------
def report_performance_by_anomaly_type(df_slice, title):
    print(f"\n=== {title} ===")

    normal_subset = df_slice[df_slice["anomaly_type"] == "normal"].copy()
    normal_total = len(normal_subset)
    normal_false_alarms = (normal_subset["predicted_label"] == 1).sum()
    normal_correct = (normal_subset["predicted_label"] == 0).sum()
    normal_false_alarm_rate = normal_false_alarms / normal_total if normal_total > 0 else 0.0

    print("\nNormal rows:")
    print(f"Total normal points:      {normal_total}")
    print(f"Correctly normal:         {normal_correct}")
    print(f"False alarms (FP):        {normal_false_alarms}")
    print(f"False alarm rate:         {normal_false_alarm_rate:.4f}")

    anomaly_types = [atype for atype in df_slice["anomaly_type"].unique() if atype != "normal"]

    for atype in anomaly_types:
        subset = df_slice[df_slice["anomaly_type"] == atype].copy()

        total_points = len(subset)
        detected_points = (subset["predicted_label"] == 1).sum()
        missed_points = (subset["predicted_label"] == 0).sum()
        detection_rate = detected_points / total_points if total_points > 0 else 0.0

        avg_score = subset["anomaly_score"].mean()
        min_score = subset["anomaly_score"].min()
        max_score = subset["anomaly_score"].max()

        print(f"\nAnomaly Type: {atype}")
        print(f"Total points:         {total_points}")
        print(f"Detected points:      {detected_points}")
        print(f"Missed points:        {missed_points}")
        print(f"Detection rate:       {detection_rate:.4f}")
        print(f"Avg anomaly score:    {avg_score:.6e}")
        print(f"Min anomaly score:    {min_score:.6e}")
        print(f"Max anomaly score:    {max_score:.6e}")

# -----------------------------
# 5. Configuration
# -----------------------------
cv_start, cv_end = 3000, 3999
test_start, test_end = 4000, 4999

# -----------------------------
# 6. Load data
# -----------------------------
df = pd.read_csv(input_file)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Apply feature engineering
df = add_engineered_features(df)

# -----------------------------
# 7. Load model
# -----------------------------
model = np.load(model_file, allow_pickle=True)
mu = model["mu"]
var = model["var"]
feature_cols = list(model["feature_cols"])

print("Loaded model from:", model_file)
print("Features:", feature_cols)

# -----------------------------
# 8. Prepare CV and test sets
# -----------------------------
cv_df = df.iloc[cv_start:cv_end + 1].copy()
test_df = df.iloc[test_start:test_end + 1].copy()

X_cv = cv_df[feature_cols].values
y_cv = cv_df["label"].values

X_test = test_df[feature_cols].values
y_test = test_df["label"].values

# -----------------------------
# 9. Compute probabilities
# -----------------------------
p_cv = gaussian_density(X_cv, mu, var)
p_test = gaussian_density(X_test, mu, var)

# -----------------------------
# 10. Select epsilon on CV
# -----------------------------
epsilon, best_cv_f1 = select_epsilon(y_cv, p_cv)

print(f"\nSelected epsilon from CV: {epsilon:.12e}")
print(f"Best CV F1: {best_cv_f1:.4f}")

# -----------------------------
# 11. CV metrics
# -----------------------------
y_cv_pred = (p_cv < epsilon).astype(int)

print("\n=== CV Metrics ===")
print("Precision:", f"{precision_score(y_cv, y_cv_pred, zero_division=0):.4f}")
print("Recall:   ", f"{recall_score(y_cv, y_cv_pred, zero_division=0):.4f}")
print("F1:       ", f"{f1_score(y_cv, y_cv_pred, zero_division=0):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_cv, y_cv_pred))

# -----------------------------
# 12. Test metrics
# -----------------------------
y_test_pred = (p_test < epsilon).astype(int)

print("\n=== Test Metrics ===")
print("Precision:", f"{precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print("Recall:   ", f"{recall_score(y_test, y_test_pred, zero_division=0):.4f}")
print("F1:       ", f"{f1_score(y_test, y_test_pred, zero_division=0):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# -----------------------------
# 13. Full monitoring region analysis
# -----------------------------
monitor_start, monitor_end = 3000, 4999
monitor_df = df.iloc[monitor_start:monitor_end + 1].copy()

X_monitor = monitor_df[feature_cols].values
p_monitor = gaussian_density(X_monitor, mu, var)
y_monitor_pred = (p_monitor < epsilon).astype(int)

monitor_df["anomaly_score"] = p_monitor
monitor_df["predicted_label"] = y_monitor_pred

report_performance_by_anomaly_type(
    monitor_df,
    "Performance by Anomaly Type (Full Monitoring Region: rows 3000–4999)"
)