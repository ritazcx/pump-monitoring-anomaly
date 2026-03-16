"""Data pipeline utilities for loading, transforming, and saving sensor datasets."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from common.paths import DATA_RAW_DIR, DATA_PROCESSED_DIR, RAW_SENSOR_FILE
from pipeline.feature_engineering import add_engineered_features


def load_raw_sensor_data(file_name: str = RAW_SENSOR_FILE) -> pd.DataFrame:
    """Load raw sensor data from the data/raw directory."""
    file_path = DATA_RAW_DIR / file_name
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def process_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to raw sensor data."""
    df_processed = add_engineered_features(df)
    return df_processed


def save_processed_sensor_data(df: pd.DataFrame, file_name: str = RAW_SENSOR_FILE) -> Path:
    """Save processed sensor data to the data/processed directory."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED_DIR / file_name
    df.to_csv(output_path, index=False)
    return output_path


def run_pipeline(
    raw_file: str = RAW_SENSOR_FILE,
    processed_file: str = RAW_SENSOR_FILE,
) -> pd.DataFrame:
    """Run the end-to-end pipeline: load raw data, process, and save."""
    df_raw = load_raw_sensor_data(raw_file)
    df_processed = process_sensor_data(df_raw)
    save_processed_sensor_data(df_processed, processed_file)
    return df_processed
