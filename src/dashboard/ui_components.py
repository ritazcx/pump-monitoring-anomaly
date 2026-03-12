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


def render_issue_panel(df_window, current_episode):
    st.subheader("Issue Interpretation")

    anomaly_rows = df_window.loc[df_window["anomaly_flag"]]

    if not anomaly_rows.empty:
        pattern_df = (
            anomaly_rows["issue_pattern"]
            .value_counts()
            .rename_axis("issue_pattern")
            .reset_index(name="count")
        )
        st.dataframe(pattern_df, use_container_width=True, hide_index=True)

    if current_episode is None:
        info = get_issue_details("normal")
        st.success("No persistent abnormal condition detected in selected window.")
    else:
        detected_issue = current_episode["dominant_pattern"]
        info = get_issue_details(detected_issue)

        status = str(current_episode["status"]).title()
        start_time = pd.to_datetime(current_episode["start_time"]).strftime("%Y-%m-%d %H:%M")
        end_time = pd.to_datetime(current_episode["end_time"]).strftime("%Y-%m-%d %H:%M")
        duration = int(current_episode["duration_min"])

        st.error(f"Detected Pattern: {detected_issue.replace('_', ' ').title()}")
        st.markdown(f"**Episode status:** {status}")
        st.markdown(f"**Start time:** {start_time}")
        st.markdown(f"**End time:** {end_time}")
        st.markdown(f"**Duration:** {duration} min")

        st.markdown("**Explanation**")
        st.write(info["description"])

    changes = detect_signal_changes(df_window)
    if changes:
        st.markdown("**Signal progression**")
        for change in changes:
            st.write(f"- {change}")

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