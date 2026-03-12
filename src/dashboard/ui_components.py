import plotly.express as px
import streamlit as st
import pandas as pd

from interpretation import get_issue_details, detect_signal_changes


@st.cache_data
def plot_sensor_chart(df, column):
    """Cache chart generation based on window data and column."""
    fig = px.line(df, x="timestamp", y=column, render_mode="webgl")

    anomalies = df[df["anomaly_flag"]]

    fig.add_scatter(
        x=anomalies["timestamp"],
        y=anomalies[column],
        mode="markers",
        marker=dict(color="red", size=6),
        name="Anomaly"
    )

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white"
    )
    fig.update_traces(line=dict(width=2))
    return fig


def render_issue_panel(df_window, episodes_df, displayed_episode):
    st.subheader("Issue Interpretation")

    if displayed_episode is None:
        st.info("No known issue pattern detected in selected window.")
        return

    detected_issue = displayed_episode["dominant_pattern"]
    info = get_issue_details(detected_issue)

    status = str(displayed_episode["status"]).title()
    start_time = pd.to_datetime(displayed_episode["start_time"])
    end_time = pd.to_datetime(displayed_episode["end_time"])
    duration = int(displayed_episode["duration_min"])

    st.error(f"Detected Pattern: {detected_issue.replace('_', ' ').title()}")
    st.markdown(f"**Episode status:** {status}")
    st.markdown(f"**Start time:** {start_time.strftime('%Y-%m-%d %H:%M')}")
    st.markdown(f"**End time:** {end_time.strftime('%Y-%m-%d %H:%M')}")
    st.markdown(f"**Duration:** {duration} min")

    st.markdown("**Explanation**")
    st.write(info["description"])

    # Signal progression window:
    # from earliest start to latest end across all episodes
    # with the same known dominant pattern
    same_pattern_episodes = episodes_df[
        episodes_df["dominant_pattern"] == detected_issue
    ]

    progression_start = pd.to_datetime(same_pattern_episodes["start_time"].min())
    progression_end = pd.to_datetime(same_pattern_episodes["end_time"].max())

    progression_df = df_window[
        (df_window["timestamp"] >= progression_start) &
        (df_window["timestamp"] <= progression_end)
    ].copy()

    changes = detect_signal_changes(progression_df, pattern=detected_issue)

    st.markdown("**Signal Progression**")

    if changes:
        for change in changes:
            if "—" in change:
                ts, message = change.split("—", 1)
                ts = ts.strip()
                message = message.strip()
            else:
                ts = ""
                message = change

            st.markdown(
                f"""
                <div style="
                    border-left: 3px solid #d1d5db;
                    padding: 0.45rem 0.75rem;
                    margin-bottom: 0.5rem;
                    background-color: #f8f9fb;
                    border-radius: 0.4rem;
                ">
                    <div style="
                        font-size: 0.78rem;
                        color: #6b7280;
                        margin-bottom: 0.15rem;
                    ">{ts}</div>
                    <div style="
                        font-size: 0.95rem;
                        color: #111827;
                        line-height: 1.4;
                    ">{message}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("No significant signal progression detected for this pattern window.")

    st.markdown("**Recommended action**")
    st.write(info["action"])

    st.markdown("**Urgency**")
    st.write(info["urgency"])


def render_alert_table(df_window):
    st.subheader("Recent Alerts")

    alerts = df_window[df_window["anomaly_flag"]].copy()

    if alerts.empty:
        st.info("No alerts in selected window.")
        return

    alerts = alerts[["timestamp", "anomaly_score", "issue_pattern"]].tail(10)
    alerts["anomaly_score"] = alerts["anomaly_score"].map(lambda x: f"{x:.6f}")

    st.dataframe(alerts, use_container_width=True, hide_index=True)