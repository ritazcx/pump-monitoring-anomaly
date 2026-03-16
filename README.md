# Smart Factory Monitoring Prototype

A modular prototype for equipment monitoring, manufacturing analytics, anomaly detection, and dashboard visualization.

## Repository Structure

- data/ (raw/ processed/ reference/)
- docs/
- figures/
- models/
- notebooks/
- requirements.txt
- src/ (app/ analytics/ common/ monitoring/ pipeline/ simulation/ training/)

## Quick Start

1. pip install -r requirements.txt
2. python src/simulation/generate_normal_data.py
3. python src/simulation/inject_anomalies.py
4. streamlit run src/app/dashboard_app.py

