import pandas as pd
import plotly.express as px
import streamlit as st

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
# Load data
# ---------------------------------------------------
DATA_PATH = "data/pump_sensor_data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()

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
    st.metric(
        label="Health Status",
        value="WARNING"
    )

with col3:
    st.metric(
        label="Latest Anomaly Score",
        value="0.94"
    )

with col4:
    st.metric(
        label="Last Update",
        value="17:00"
    )

st.divider()

# ---------------------------------------------------
# Main Dashboard Layout
# ---------------------------------------------------
left_panel, right_panel = st.columns([3, 1])

# ---------------------------------------------------
# Sensor Trend Charts
# ---------------------------------------------------
def plot_sensor_chart(df, column, title):

    fig = px.line(
        df,
        x="timestamp",
        y=column,
        title=title
    )

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig
    
with left_panel:

    st.subheader("Sensor Trends")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Temperature")
        fig = plot_sensor_chart(df, "temperature", "Temperature")
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.subheader("Vibration")
        fig = plot_sensor_chart(df, "vibration", "Vibration")
        st.plotly_chart(fig, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.subheader("Pressure")
        fig = plot_sensor_chart(df, "pressure", "Pressure")
        st.plotly_chart(fig, use_container_width=True)

    with chart_col4:
        st.subheader("Flow Rate")
        fig = plot_sensor_chart(df, "flow_rate", "Flow Rate")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Power Consumption")
    fig = plot_sensor_chart(df, "power", "Power Consumption")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# Issue Interpretation Panel
# ---------------------------------------------------
with right_panel:

    st.subheader("Issue Interpretation")

    st.warning("Detected Pattern: Possible Cavitation")

    st.markdown("""
**Signal progression**

- Pressure drop detected  
- Flow rate declined  
- Vibration increased  
- Temperature rising
""")

    st.markdown("""
**Recommended actions**

1. Reduce pump speed  
2. Inspect inlet valve  
3. Verify suction conditions
""")

    st.error("Urgency: Escalate to maintenance within 2 hours")

st.divider()

# ---------------------------------------------------
# Recent Alerts Table
# ---------------------------------------------------
st.subheader("Recent Alerts")

st.write("Alerts table placeholder")