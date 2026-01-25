"""
HOPE Data Preprocessing Script

This script processes raw OBD2 data files and prepares them for training
the anomaly detection model.

Features:
- Reads various data formats (CSV, JSON, SQLite)
- Handles missing values and outliers
- Normalizes timestamps and aligns data
- Creates training sequences

Usage:
    python preprocess_data.py --input_dir ../data/raw --output_dir ../data/processed
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Feature columns expected in the data
REQUIRED_FEATURES = [
    'engine_rpm',
    'vehicle_speed',
    'engine_load',
    'coolant_temp',
    'intake_air_temp',
    'maf_flow',
    'throttle_position',
    'fuel_pressure',
    'short_term_fuel_trim',
    'long_term_fuel_trim',
]

# PID to feature name mapping (for raw OBD2 data)
PID_MAPPING = {
    '010C': 'engine_rpm',
    '010D': 'vehicle_speed',
    '0104': 'engine_load',
    '0105': 'coolant_temp',
    '010F': 'intake_air_temp',
    '0110': 'maf_flow',
    '0111': 'throttle_position',
    '010A': 'fuel_pressure',
    '0106': 'short_term_fuel_trim',
    '0107': 'long_term_fuel_trim',
}


def load_csv_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def load_json_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Load data from a JSON file."""
    try:
        df = pd.read_json(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def load_sqlite_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Load data from a SQLite database."""
    try:
        import sqlite3
        conn = sqlite3.connect(file_path)
        df = pd.read_sql_query("SELECT * FROM Readings", conn)
        conn.close()
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to standard format."""
    # Common column name variations
    column_mapping = {
        # Timestamps
        'timestamp': 'timestamp',
        'time': 'timestamp',
        'Timestamp': 'timestamp',
        'datetime': 'timestamp',

        # Session ID
        'session_id': 'session_id',
        'SessionId': 'session_id',
        'sessionId': 'session_id',

        # PID
        'pid': 'pid',
        'PID': 'pid',

        # Value
        'value': 'value',
        'Value': 'value',

        # RPM
        'rpm': 'engine_rpm',
        'RPM': 'engine_rpm',
        'engine_rpm': 'engine_rpm',
        'EngineRPM': 'engine_rpm',

        # Speed
        'speed': 'vehicle_speed',
        'Speed': 'vehicle_speed',
        'vehicle_speed': 'vehicle_speed',
        'VehicleSpeed': 'vehicle_speed',

        # Load
        'load': 'engine_load',
        'Load': 'engine_load',
        'engine_load': 'engine_load',
        'EngineLoad': 'engine_load',

        # Coolant
        'coolant': 'coolant_temp',
        'coolant_temp': 'coolant_temp',
        'CoolantTemp': 'coolant_temp',
        'coolant_temperature': 'coolant_temp',

        # Intake air temp
        'intake_temp': 'intake_air_temp',
        'IntakeAirTemp': 'intake_air_temp',
        'intake_air_temp': 'intake_air_temp',

        # MAF
        'maf': 'maf_flow',
        'MAF': 'maf_flow',
        'maf_flow': 'maf_flow',
        'MassAirFlow': 'maf_flow',

        # Throttle
        'throttle': 'throttle_position',
        'ThrottlePosition': 'throttle_position',
        'throttle_position': 'throttle_position',

        # Fuel pressure
        'fuel_pressure': 'fuel_pressure',
        'FuelPressure': 'fuel_pressure',

        # Fuel trims
        'stft': 'short_term_fuel_trim',
        'STFT': 'short_term_fuel_trim',
        'short_term_fuel_trim': 'short_term_fuel_trim',
        'ShortTermFuelTrim': 'short_term_fuel_trim',
        'ltft': 'long_term_fuel_trim',
        'LTFT': 'long_term_fuel_trim',
        'long_term_fuel_trim': 'long_term_fuel_trim',
        'LongTermFuelTrim': 'long_term_fuel_trim',
    }

    df = df.rename(columns=column_mapping)
    return df


def pivot_obd2_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format OBD2 readings to wide format."""
    if 'pid' not in df.columns:
        return df

    # Map PIDs to feature names
    df['feature'] = df['pid'].map(PID_MAPPING)
    df = df.dropna(subset=['feature'])

    # Pivot to wide format
    if 'session_id' in df.columns:
        pivot_df = df.pivot_table(
            index=['session_id', 'timestamp'],
            columns='feature',
            values='value',
            aggfunc='mean'
        ).reset_index()
    else:
        pivot_df = df.pivot_table(
            index='timestamp',
            columns='feature',
            values='value',
            aggfunc='mean'
        ).reset_index()

    return pivot_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the data."""
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp')

    # Apply realistic value ranges
    value_ranges = {
        'engine_rpm': (0, 10000),
        'vehicle_speed': (0, 350),
        'engine_load': (0, 100),
        'coolant_temp': (-40, 215),
        'intake_air_temp': (-40, 100),
        'maf_flow': (0, 500),
        'throttle_position': (0, 100),
        'fuel_pressure': (0, 1000),
        'short_term_fuel_trim': (-50, 50),
        'long_term_fuel_trim': (-50, 50),
    }

    for col, (min_val, max_val) in value_ranges.items():
        if col in df.columns:
            # Clip values to valid range
            df[col] = df[col].clip(min_val, max_val)

            # Replace outliers with NaN for interpolation
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df.loc[(df[col] < q1) | (df[col] > q99), col] = np.nan

    # Interpolate missing values
    for col in REQUIRED_FEATURES:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=5)
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

    return df


def resample_to_1hz(df: pd.DataFrame) -> pd.DataFrame:
    """Resample data to 1 Hz frequency."""
    if 'timestamp' not in df.columns:
        return df

    df = df.set_index('timestamp')

    # Resample to 1 second intervals
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_resampled = df[numeric_cols].resample('1s').mean()

    # Interpolate missing values after resampling
    df_resampled = df_resampled.interpolate(method='linear', limit=10)

    return df_resampled.reset_index()


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Extract feature matrix from DataFrame."""
    available_features = [f for f in REQUIRED_FEATURES if f in df.columns]

    if len(available_features) < len(REQUIRED_FEATURES):
        missing = set(REQUIRED_FEATURES) - set(available_features)
        logger.warning(f"Missing features: {missing}")

        # Add missing features with default values
        for feature in missing:
            df[feature] = 0.0

    return df[REQUIRED_FEATURES].values


def process_file(file_path: Path) -> Optional[np.ndarray]:
    """Process a single data file."""
    # Load based on file extension
    ext = file_path.suffix.lower()

    if ext == '.csv':
        df = load_csv_data(file_path)
    elif ext == '.json':
        df = load_json_data(file_path)
    elif ext in ['.db', '.sqlite', '.sqlite3']:
        df = load_sqlite_data(file_path)
    else:
        logger.warning(f"Unsupported file format: {ext}")
        return None

    if df is None or len(df) == 0:
        return None

    # Process pipeline
    df = normalize_columns(df)
    df = pivot_obd2_readings(df)
    df = clean_data(df)
    df = resample_to_1hz(df)

    # Extract features
    features = extract_features(df)

    if len(features) < 60:  # Minimum sequence length
        logger.warning(f"Not enough data in {file_path} ({len(features)} samples)")
        return None

    return features


def main():
    parser = argparse.ArgumentParser(description='Preprocess OBD2 data for training')
    parser.add_argument('--input_dir', type=str, default='../data/raw',
                        help='Directory containing raw data files')
    parser.add_argument('--output_dir', type=str, default='../data/processed',
                        help='Directory to save processed data')
    parser.add_argument('--min_samples', type=int, default=60,
                        help='Minimum samples per file')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all data files
    data_files = list(input_dir.glob('**/*.csv')) + \
                 list(input_dir.glob('**/*.json')) + \
                 list(input_dir.glob('**/*.db'))

    if not data_files:
        logger.warning(f"No data files found in {input_dir}")
        logger.info("Creating sample data structure...")

        # Create sample directory structure
        (input_dir / 'vehicle_001').mkdir(parents=True, exist_ok=True)
        (input_dir / 'vehicle_002').mkdir(parents=True, exist_ok=True)

        logger.info("Place your data files in the raw data directory and re-run this script")
        return

    logger.info(f"Found {len(data_files)} data files")

    # Process all files
    all_data = []

    for file_path in data_files:
        logger.info(f"Processing {file_path}...")
        features = process_file(file_path)

        if features is not None and len(features) >= args.min_samples:
            all_data.append(features)
            logger.info(f"  -> {len(features)} samples extracted")

    if not all_data:
        logger.error("No valid data processed")
        return

    # Concatenate all data
    combined_data = np.concatenate(all_data, axis=0)
    logger.info(f"Total processed samples: {len(combined_data)}")

    # Save processed data
    output_path = output_dir / 'training_data.npy'
    np.save(output_path, combined_data)
    logger.info(f"Saved processed data to {output_path}")

    # Save metadata
    metadata = {
        'n_samples': len(combined_data),
        'n_features': len(REQUIRED_FEATURES),
        'features': REQUIRED_FEATURES,
        'files_processed': len(all_data),
        'processing_date': datetime.now().isoformat(),
    }

    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("Preprocessing complete!")


if __name__ == '__main__':
    main()
