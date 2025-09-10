# src/credit_risk/utils/optuna_common.py
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import mlflow
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour,
)

def _as_figure(obj) -> Figure:
    if isinstance(obj, Figure):
        return obj
    # single Axes
    if hasattr(obj, "figure"):
        return obj.figure
    # arrays of Axes / Figures
    try:
        import numpy as np  # local import to avoid hard dep here
        if isinstance(obj, np.ndarray) and obj.size > 0:
            first = obj.flat[0]
            if hasattr(first, "figure"):
                return first.figure
            if isinstance(first, Figure):
                return first
    except Exception:
        pass
    return plt.gcf()

def log_study_figures(study: optuna.Study, prefix: str = "figures/optuna") -> None:
    os.makedirs(prefix, exist_ok=True)
    items = [
        ("opt_history.png",       plot_optimization_history),
        ("param_importances.png", plot_param_importances),
        ("parallel_coord.png",    plot_parallel_coordinate),
        ("slice.png",             plot_slice),
        ("contour.png",           plot_contour),
    ]
    for name, fn in items:
        try:
            fig = _as_figure(fn(study))
            mlflow.log_figure(fig, f"{prefix}/{name}")
            plt.close(fig)
        except Exception as e:
            mlflow.log_text(str(e), f"{prefix}/{name.replace('.png','')}_error.txt")

def export_trials_csv(study: optuna.Study, path: str = "optuna/trials.csv") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = study.trials_dataframe(attrs=(
        "number","value","state","params","datetime_start","datetime_complete","duration"
    ))
    df.to_csv(path, index=False)
    mlflow.log_artifact(path)
