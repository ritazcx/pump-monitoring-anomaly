import plotly.express as px
import streamlit as st

from interpretation import get_issue_details, detect_signal_changes


def plot_sensor_chart(df, column):
    fig = px.line(df, x="timestamp", y=column)

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


def render_issue_panel(latest, df_window):
    st.subheader("Issue Interpretation")

    latest_issue = latest["issue_pattern"]
    info = get_issue_details(latest_issue)

    if latest["anomaly_flag"]:
        st.error(f"Detected Pattern: {latest_issue.replace('_', ' ').title()}")
        st.write("Explanation:")
        st.write(info["description"])
    else:
        st.success("No abnormal pattern detected in selected window.")

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