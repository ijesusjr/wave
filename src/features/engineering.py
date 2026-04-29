"""
Feature engineering utilities for SWaT anomaly detection.
"""
import pandas as pd
import numpy as np


def add_rolling_features(df: pd.DataFrame, cols: list, windows: list = [10, 60]) -> pd.DataFrame:
    """
    Add rolling mean, std, and rate-of-change for given sensor columns.
    
    Parameters
    ----------
    df      : DataFrame with datetime index, 1-second sampling
    cols    : list of continuous sensor column names
    windows : list of window sizes in seconds
    
    Returns
    -------
    DataFrame with original + rolling features appended
    """
    out = df.copy()
    for col in cols:
        for w in windows:
            out[f'{col}_rmean_{w}s'] = df[col].rolling(w).mean()
            out[f'{col}_rstd_{w}s']  = df[col].rolling(w).std()
        out[f'{col}_roc'] = df[col].diff()  # rate of change (1-step)
    return out


def identify_sensor_types(df: pd.DataFrame) -> tuple:
    """
    Split columns into continuous sensors vs binary actuators.
    
    Returns
    -------
    (continuous_cols, binary_cols)
    """
    binary_cols     = [c for c in df.columns if df[c].nunique() <= 2]
    continuous_cols = [c for c in df.columns if c not in binary_cols]
    return continuous_cols, binary_cols
