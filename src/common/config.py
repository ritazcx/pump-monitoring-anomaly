"""Global configuration constants for the Smart Factory Monitoring Prototype."""

# Anomaly detection thresholds and smoothing
ANOMALY_SMOOTHING_WINDOW = 3  # points

# Episode building
EPISODE_GAP_MINUTES = 5

# Health scoring
HEALTH_SCORE_NORMAL = 1.0
HEALTH_SCORE_WARNING = 0.5
HEALTH_SCORE_CRITICAL = 0.0

# KPI defaults
OEE_TARGET_UTILIZATION = 0.8
OEE_TARGET_THROUGHPUT = 1.0
