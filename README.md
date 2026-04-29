# WAVE — Water-process Anomaly and Variance Explorer

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-green)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview
Unsupervised anomaly detection pipeline applied to the **SWaT (Secure Water Treatment)** dataset — a real-world industrial control system dataset from iTrust, Singapore University of Technology and Design (SUTD).

The project simulates a realistic industrial scenario: **no fault labels are used during training**, reflecting the actual constraint faced by process engineers and reliability teams. Labels are used only for post-hoc evaluation.

## Problem Statement
Industrial plants generate continuous streams of sensor data. Anomalies — whether equipment faults, cyberattacks, or process deviations — are rare, often unlabeled, and potentially catastrophic. This project builds and compares three unsupervised detection approaches:

- **Isolation Forest** — tree-based density estimation
- **PCA-based (Hotelling T² + SPE)** — multivariate statistical process control, standard in chemical engineering
- **Autoencoder** — deep learning reconstruction error

## Dataset
SWaT dataset from iTrust, SUTD (access by request):
- 7 days normal operation (~500k rows)
- 4 days with 41 labeled attacks (~450k rows)
- 51 sensors/actuators across 6 sub-processes (P1–P6)

## Project Structure
```
wave/
├── data/
│   ├── raw/               # Original xlsx files (not versioned)
│   └── processed/         # Cleaned, feature-engineered parquet files
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_isolation_forest.ipynb
│   ├── 04_pca_hotelling.ipynb
│   ├── 05_autoencoder.ipynb
│   └── 06_comparison.ipynb
├── src/
│   ├── features/          # Feature engineering functions
│   ├── models/            # Model classes and wrappers
│   └── visualization/     # Reusable plot functions
├── app/                   # Streamlit application
├── reports/
│   └── figures/           # Saved plots
├── tests/
├── requirements.txt
└── README.md
```

## Quickstart
```bash
git clone https://github.com/ijesusjr/wave
cd wave
pip install -r requirements.txt
# Place SWaT xlsx files in data/raw/
jupyter notebook notebooks/01_EDA.ipynb
```

## Results
*(To be updated after modeling)*

| Model | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Isolation Forest | — | — | — |
| PCA (T² + SPE) | — | — | — |
| Autoencoder | — | — | — |

## Author
**Ildebrando** — Chemical Engineer + Data Scientist  
[GitHub](https://github.com/ijesusjr) · [LinkedIn](#)
