"""
Reusable plot functions for anomaly detection results.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_anomaly_scores(index, scores, threshold, y_true=None, title='Anomaly Scores', figsize=(16, 4)):
    """
    Plot anomaly scores over time with threshold line.
    Optionally overlay ground-truth attack windows.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(index, scores, lw=0.5, color='steelblue', label='Anomaly score')
    ax.axhline(threshold, color='red', lw=1.2, linestyle='--', label=f'Threshold = {threshold:.3f}')

    if y_true is not None:
        for i, (idx, val) in enumerate(zip(index, y_true)):
            if val == 1:
                ax.axvspan(idx, idx, alpha=0.1, color='orange')

    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Score')
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_contribution(feature_names, contributions, top_n=15, title='Top Contributing Features'):
    """
    Horizontal bar chart of top-N feature contributions to anomaly score.
    Useful for explainability / operator guidance.
    """
    idx = np.argsort(contributions)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(np.array(feature_names)[idx], np.array(contributions)[idx], color='coral')
    ax.set_title(title)
    ax.set_xlabel('Contribution')
    plt.tight_layout()
    return fig
