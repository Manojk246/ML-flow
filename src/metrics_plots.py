from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "r2": r2}


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Predicted vs Actual") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, s=20, alpha=0.7, ax=ax)
    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importance(importances: np.ndarray, feature_names: List[str], title: str = "Feature Importance") -> plt.Figure:
    order = np.argsort(importances)[::-1]
    imp_sorted = np.array(importances)[order]
    names_sorted = np.array(feature_names)[order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.3)))
    sns.barplot(x=imp_sorted, y=names_sorted, orient='h', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    fig.tight_layout()
    return fig


def plot_model_comparison(results: pd.DataFrame, metric: str = 'r2') -> plt.Figure:
    results_sorted = results.sort_values(by=metric, ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=results_sorted, x='model', y=metric, ax=ax)
    ax.set_title(f'Model comparison by {metric.upper()}')
    ax.set_xlabel('Model')
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0, 1)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, rotation=0, xytext=(0, 3), textcoords='offset points')
    fig.tight_layout()
    return fig
