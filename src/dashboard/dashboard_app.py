import streamlit as st

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_utils import load_data, load_model, compute_scores, add_issue_patterns
from interpretation import get_issue_details, detect_signal_changes
from ui_components import plot_sensor_chart, render_issue_panel, render_alert_table

# ---------------------------------------------------
# Page configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Pump Health Monitoring Dashboard",
    layout="wide"
)

# ---------------------------------------------------
# Optional styling (simple dashboard spacing)
# ---------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("Pump Health Monitoring Dashboard")

st.markdown("AI-enabled monitoring for early detection of abnormal pump behavior")

st.divider()

# ---------------------------------------------------
# Load Data + Score
# ---------------------------------------------------
if "df_processed" not in st.session_state:

    df = load_data()
    mu, var, epsilon, features = load_model()

    df = compute_scores(df, features, mu, var, epsilon)
    df = add_issue_patterns(df)

    st.session_state.df_processed = df
    st.session_state.model = (mu, var, epsilon, features)

df = st.session_state.df_processed
mu, var, epsilon, features = st.session_state.model

# ---------------------------------------------------
# Slider
# ---------------------------------------------------
min_time = df["timestamp"].min()
max_time = df["timestamp"].max()

start_time, end_time = st.slider(
    "Time window",
    min_value=min_time.to_pydatetime(),
    max_value=max_time.to_pydatetime(),
    value=(min_time.to_pydatetime(), max_time.to_pydatetime()),
)

@st.cache_data
def filter_window(df, start, end):
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    
df_window = filter_window(df, start_time, end_time)

if df_window.empty:
    st.warning("No data in selected window.")
    st.stop()

latest = df_window.iloc[-1]

# ---------------------------------------------------
# Top KPI Section
# ---------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Pump ID",
        value="P-2401"
    )

with col2:
    if latest["anomaly_score"] < epsilon * 0.5:
        health_status = "CRITICAL"
    elif latest["anomaly_flag"]:
        health_status = "WARNING"
    else:
        health_status = "NORMAL"

    st.metric(
        label="Health Status",
        value=health_status
    )

    snapshot_text = (
        f"T:{latest['temperature']:.1f}°C  |  "
        f"V:{latest['vibration']:.2f}  |  "
        f"P:{latest['pressure']:.2f}  |  "
        f"F:{latest['flow_rate']:.1f}"
    )

    st.caption(f"Latest sensors → {snapshot_text}")

with col3:
    st.metric(
        label="Latest Anomaly Score",
        value=f"{latest['anomaly_score']:.4f}"
    )

with col4:
    # st.metric(
    #     label="Last Update",
    #     value=latest["timestamp"].strftime("%H:%M")
    # )
    num_anomalies = int(df_window["anomaly_flag"].sum())
    st.metric(
        "Anomalies in Window", 
        num_anomalies)

st.divider()

# ---------------------------------------------------
# Main Dashboard Layout
# ---------------------------------------------------
left_panel, right_panel = st.columns([3, 1])

# ---------------------------------------------------
# Sensor Trend Charts
# ---------------------------------------------------
with left_panel:

    st.subheader("Sensor Trends")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Temperature")
        st.plotly_chart(plot_sensor_chart(df_window, "temperature"), use_container_width=True)

    with chart_col2:
        st.subheader("Vibration")
        st.plotly_chart(plot_sensor_chart(df_window, "vibration"), use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.subheader("Pressure")
        st.plotly_chart(plot_sensor_chart(df_window, "pressure"), use_container_width=True)

    with chart_col4:
        st.subheader("Flow Rate")
        st.plotly_chart(plot_sensor_chart(df_window, "flow_rate"), use_container_width=True)

    st.subheader("Power Consumption")
    st.plotly_chart(plot_sensor_chart(df_window, "power"), use_container_width=True)

# ---------------------------------------------------
# Issue Interpretation Panel
# ---------------------------------------------------
with right_panel:
    render_issue_panel(latest, df_window)

# ---------------------------------------------------
# Recent Alerts Table
# ---------------------------------------------------
st.divider()
render_alert_table(df_window)
