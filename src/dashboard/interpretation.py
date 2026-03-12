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


def detect_signal_changes(df_window, pattern=None):
    """
    Detect signal progression with timestamps.
    """
    if df_window.empty or len(df_window) < 3:
        return []

    changes = []

    df = df_window.sort_values("timestamp").reset_index(drop=True)

    base = df.iloc[0]

    for i in range(1, len(df)):
        row = df.iloc[i]
        ts = row["timestamp"].strftime("%Y-%m-%d %H:%M")

        vib_delta = row["vibration"] - base["vibration"]
        pressure_delta = row["pressure"] - base["pressure"]
        flow_delta = row["flow_rate"] - base["flow_rate"]
        power_delta = row["power"] - base["power"]

        if pattern == "cavitation":
            if vib_delta > 0.2:
                changes.append(
                    f"{ts} — Vibration increased, consistent with cavitation onset."
                )
                break

            if pressure_delta < -0.05:
                changes.append(
                    f"{ts} — Pressure dropped during the cavitation pattern window."
                )
                break

        elif pattern == "blockage":
            if flow_delta < -3:
                changes.append(
                    f"{ts} — Flow rate began decreasing."
                )

            if power_delta > 0.2:
                changes.append(
                    f"{ts} — Power consumption increased while flow decreased."
                )
                break

            if pressure_delta > 0.03:
                changes.append(
                    f"{ts} — Pressure started rising slightly."
                )
                break

        elif pattern == "bearing_wear":
            if vib_delta > 0.2:
                changes.append(
                    f"{ts} — Vibration increased progressively."
                )
                break

    return changes