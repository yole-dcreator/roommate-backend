"""
preprocessor.py
Handles all data cleaning, encoding, and normalization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json, os

ORDINAL_MAPS = {
    'Gender': {'Female': 0, 'Male': 1},
    'Preferred living environment': {
        'Comfortable with a bit of clutter': 0,
        'Flexible/ not particular': 1,
        'Generally tidy': 2,
        'Very organized': 3
    },
    'Organize personal space frequency': {
        'Never': 0, 'Occasionally': 1,
        'Weekly': 2, 'A few times a week': 3, 'Daily': 4
    },
    'Productive environment': {
        'Quiet with occasional activity': 0,
        'Balanced': 1,
        'Lively with some quiet moments': 2
    },
    'Atmosphere created': {'Very calm': 0, 'Calm': 1, 'Energetic': 2},
    'Invite friends frequency': {
        'Never': 0, 'Rarely': 1, 'Occasionally': 2, 'Often': 3, 'Very often': 4
    },
    'Comfortable with roommate bringing guests': {'No': 0, 'Sometimes': 1, 'Yes': 2},
    'Sleep lights preference': {
        'Must be OFF': 0, 'Prefer OFF': 1, 'Sometimes ON': 2, 'Always ON': 3
    },
    'Fan speed preference': {'Low': 0, 'Medium': 1, 'High': 2, 'Very high': 3},
    'Preferred study time': {
        'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Late night': 3
    },
    'Okay with roommate studying late night': {'No': 0, 'It depends': 1, 'Yes': 2},
}

FEATURE_COLS = [
    'Gender', 'Current academic level',
    'Preferred living environment', 'Organize personal space frequency',
    'Productive environment', 'Atmosphere created',
    'Invite friends frequency', 'Comfortable with roommate bringing guests',
    'Sleep lights preference', 'Fan speed preference',
    'Preferred study time', 'Okay with roommate studying late night'
]


def load_and_clean(filepath):
    """Load CSV, check for missing values and duplicates."""
    df = pd.read_csv(filepath)
    report = {
        'total_rows': len(df),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'duplicate_matric': int(df['Matric Number'].duplicated().sum()),
    }
    # Fill missing with mode
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    # Drop full duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, report


def encode(df):
    """Apply ordinal encoding to all categorical columns."""
    df_enc = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        df_enc[col] = df_enc[col].map(mapping)
    df_enc['Current academic level'] = df_enc['Current academic level'].astype(int)
    return df_enc


def scale(df_enc):
    """Z-score normalise the feature matrix."""
    X = df_enc[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def run_pipeline(filepath):
    """Full preprocessing pipeline — returns df_raw, df_enc, X_scaled, scaler, report."""
    df_raw, report = load_and_clean(filepath)
    df_enc = encode(df_raw)
    X_scaled, scaler = scale(df_enc)
    return df_raw, df_enc, X_scaled, scaler, report
