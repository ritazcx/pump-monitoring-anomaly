def infer_issue(row):
    if row["vibration"] > 2.8 and row["temperature"] > 67:
        return "bearing_wear"

    if row["vibration"] > 2.6 and abs(row["pressure"] - 5) > 0.3:
        return "cavitation"

    if row["flow_rate"] < 92 and row["power"] > 12.5:
        return "blockage"

    if row["temperature"] > 70:
        return "overheating"

    return "unknown"


def get_issue_details(issue_pattern):
    issue_map = {
        "bearing_wear": {
            "description": "High vibration with rising temperature suggests possible bearing wear.",
            "action": "Inspect pump bearings and lubrication condition.",
            "urgency": "Schedule maintenance review soon."
        },
        "cavitation": {
            "description": "Unstable vibration and pressure fluctuations suggest cavitation.",
            "action": "Check inlet pressure and suction conditions.",
            "urgency": "Inspect promptly if the pattern continues."
        },
        "blockage": {
            "description": "Low flow combined with higher power indicates possible blockage.",
            "action": "Inspect pipeline or impeller for blockage.",
            "urgency": "Investigate before performance degrades further."
        },
        "overheating": {
            "description": "Temperature exceeds normal operating range.",
            "action": "Check cooling system and pump load.",
            "urgency": "Escalate if temperature keeps rising."
        },
        "unknown": {
            "description": "An anomaly was detected but does not match a known pattern.",
            "action": "Review sensor trends and inspect pump condition.",
            "urgency": "Monitor closely."
        }
    }
    return issue_map.get(issue_pattern, issue_map["unknown"])


def detect_signal_changes(df_window):
    if len(df_window) < 5:
        return []

    latest = df_window.iloc[-1]
    prev = df_window.iloc[-5]

    changes = []

    if latest["temperature"] > prev["temperature"] + 1.0:
        changes.append("Temperature rising")

    if latest["vibration"] > prev["vibration"] + 0.2:
        changes.append("Vibration increasing")

    if latest["flow_rate"] < prev["flow_rate"] - 3.0:
        changes.append("Flow decreasing")

    if latest["pressure"] < prev["pressure"] - 0.2:
        changes.append("Pressure dropping")

    if latest["power"] > prev["power"] + 0.4:
        changes.append("Power increasing")

    return changes