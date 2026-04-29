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
    Split columns into continuous sensors, ordinal valves, and binary actuators.

    SWaT actuator taxonomy:
      - MV (motorized valves): ordinal, 3 states (0=closed, 1=transitioning, 2=open)
      - Binary actuators (pumps, UV): 2 states (0/1)
      - Continuous sensors: FIT, LIT, AIT, DPIT, PIT, etc.

    Returns
    -------
    (continuous_cols, mv_cols, binary_cols)
    """
    mv_cols         = [c for c in df.columns if c.strip().upper().startswith('MV')]
    binary_cols     = [c for c in df.columns if c not in mv_cols and df[c].nunique() <= 2]
    continuous_cols = [c for c in df.columns if c not in mv_cols and c not in binary_cols]
    return continuous_cols, mv_cols, binary_cols
