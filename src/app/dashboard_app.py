import streamlit as st

import sys
import os

# Ensure the repository root (src/) is on the import path so package-style imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analytics.model_utils import load_data, load_model, compute_scores, add_issue_patterns
from monitoring.health_scoring import compute_health_score, categorize_health_state
from monitoring.interpretation import get_issue_details, detect_signal_changes
from app.ui_components import plot_sensor_chart, render_issue_panel, render_alert_table
from monitoring.episode_builder import build_condition_episodes

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

episodes_df = build_condition_episodes(df_window, gap_minutes=5)

known_episodes = episodes_df[episodes_df["dominant_pattern"] != "unknown"]

if not known_episodes.empty:
    displayed_episode = known_episodes.iloc[0]   # most recent known episode
else:
    displayed_episode = None

st.subheader("Condition Episodes")
st.dataframe(episodes_df, use_container_width=True)

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
    health_score = compute_health_score(df_window)
    health_state = categorize_health_state(health_score).upper()

    st.metric(
        label="Health Status",
        value=health_state,
        delta=f"{health_score:.2f}"
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
    # num_anomalies = int(df_window["anomaly_flag"].sum())
    # st.metric(
    #     "Anomalies in Window", 
    #     num_anomalies)

    active_episode_count = int((episodes_df["status"] == "active").sum())
    num_anomalies = int(df_window["anomaly_flag"].sum())

    st.metric(
        label="Active Episodes",
        value=active_episode_count
    )

    st.caption(f"Transient deviations / anomaly points: {num_anomalies}")

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
    render_issue_panel(df_window, episodes_df, displayed_episode)

# ---------------------------------------------------
# Recent Alerts Table
# ---------------------------------------------------
st.divider()
render_alert_table(df_window)
