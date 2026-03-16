"""Centralized filesystem path definitions for the Smart Factory Monitoring Prototype."""

from pathlib import Path

# Root of the repository (two levels above this file: src/common/paths.py)
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_REFERENCE_DIR = DATA_DIR / "reference"

MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "figures"
DOCS_DIR = ROOT_DIR / "docs"

# Common filenames
RAW_SENSOR_FILE = "pump_sensor_data.csv"
NORMAL_SENSOR_FILE = "pump_sensor_data_normal.csv"
