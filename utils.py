"""Fonctions réutilisables pour l'atelier de détection d'anomalies."""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def load_mouse_data(
    path: str | Path, n_reference_outliers: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Mouse doit avoir deux colonnes, forme obtenue : {data.shape}")
    if not 0 < n_reference_outliers < len(data):
        raise ValueError("Le nombre d'anomalies de référence doit être strictement compris entre 0 et n.")

    y_reference = np.zeros(len(data), dtype=int)
    y_reference[-n_reference_outliers:] = 1
    return data, y_reference


def fit_isolation_forest(
    X: np.ndarray, contamination: float, random_state: int
) -> Tuple[IsolationForest, np.ndarray, np.ndarray]:
    """Ajuste Isolation Forest et retourne labels (-1/1) et scores élevés pour les anomalies."""
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)
    labels = model.predict(X)
    scores = -model.decision_function(X)
    return model, labels, scores


def fit_lof(
    X: np.ndarray, n_neighbors: int, contamination: float
) -> Tuple[
    LocalOutlierFactor,
    np.ndarray,
    np.ndarray,
    LocalOutlierFactor,
    np.ndarray,
]:
    """Compare le score LOF transductif au score LOF prévu pour la nouveauté."""
    train_model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
    )
    train_labels = train_model.fit_predict(X)
    train_scores = -train_model.negative_outlier_factor_

    novelty_model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
    )
    novelty_model.fit(X)
    novelty_scores = -novelty_model.score_samples(X)
    return train_model, train_labels, train_scores, novelty_model, novelty_scores


def plot_scores(
    X: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    title: str,
    score_label: str,
    reference_labels: np.ndarray | None = None,
) -> None:
    """Affiche les points colorés par score et l'histogramme des scores."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    scatter = axes[0].scatter(X[:, 0], X[:, 1], c=scores, cmap="viridis", s=28)
    axes[0].scatter(
        X[labels == -1, 0], X[labels == -1, 1],
        facecolors="none", edgecolors="red", s=85, linewidths=1.3,
        label="Détectés",
    )
    if reference_labels is not None:
        axes[0].scatter(
            X[reference_labels == 1, 0], X[reference_labels == 1, 1],
            marker="x", c="black", s=48, label="Référence",
        )
    axes[0].set(title=title, xlabel="Variable 1", ylabel="Variable 2")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[0], label=score_label)

    axes[1].hist(scores, bins=25, color="steelblue", edgecolor="white")
    axes[1].set(title="Distribution des scores", xlabel=score_label, ylabel="Effectif")
    axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    plt.show()
