"""Evaluation metrics and plotting helpers.

Provides:

* supervised metrics (accuracy, per-class report, confusion matrix),
* unsupervised metrics that compare *discovered clusters* against the true
  classes -- Adjusted Rand Index, Normalised Mutual Information, and a
  "clustering accuracy" obtained by optimally matching cluster ids to class
  ids with the Hungarian algorithm,
* matplotlib helpers used by the notebook and the figure scripts.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


# --------------------------------------------------------------------------- #
# Unsupervised: match clusters to ground-truth classes                        #
# --------------------------------------------------------------------------- #
def match_clusters_to_labels(cluster_labels, true_labels):
    """Optimally map cluster ids to class ids (Hungarian assignment).

    Returns ``(mapping, mapped_labels)`` where ``mapping[cluster] = class`` and
    ``mapped_labels`` are the cluster labels translated into class ids so they
    can be compared directly with ``true_labels``.
    """
    cluster_labels = np.asarray(cluster_labels)
    true_labels = np.asarray(true_labels)
    n = max(cluster_labels.max(), true_labels.max()) + 1
    cost = np.zeros((n, n), dtype=np.int64)
    for c, t in zip(cluster_labels, true_labels):
        cost[c, t] += 1
    row, col = linear_sum_assignment(-cost)          # maximise overlap
    mapping = {int(r): int(c) for r, c in zip(row, col)}
    mapped = np.array([mapping[c] for c in cluster_labels])
    return mapping, mapped


def clustering_scores(cluster_labels, true_labels):
    """Return ARI, NMI and matched clustering accuracy as a dict."""
    _, mapped = match_clusters_to_labels(cluster_labels, true_labels)
    return {
        "ARI": float(adjusted_rand_score(true_labels, cluster_labels)),
        "NMI": float(normalized_mutual_info_score(true_labels, cluster_labels)),
        "matched_accuracy": float(accuracy_score(true_labels, mapped)),
    }


# --------------------------------------------------------------------------- #
# Supervised metrics                                                          #
# --------------------------------------------------------------------------- #
def supervised_scores(y_true, y_pred, class_names):
    """Return accuracy and a per-class precision/recall/F1 report dict."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": classification_report(
            y_true, y_pred, target_names=class_names,
            output_dict=True, zero_division=0,
        ),
    }


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #
def plot_sample_grid(X, y, class_names, n_per_class=3, savepath=None):
    """Grid of example patches, one row per class."""
    n_cls = len(class_names)
    fig, axes = plt.subplots(n_cls, n_per_class, figsize=(n_per_class * 1.5, n_cls * 1.5))
    for ci in range(n_cls):
        idx = np.where(y == ci)[0][:n_per_class]
        for j in range(n_per_class):
            ax = axes[ci, j]
            ax.axis("off")
            if j < len(idx):
                ax.imshow(X[idx[j]])
            if j == 0:
                ax.set_title(class_names[ci], fontsize=8, loc="left")
    fig.suptitle("EuroSAT example patches (Sentinel-2, 64x64 px)", y=1.001)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def plot_clusters_2d(coords, labels, title, class_names=None, savepath=None):
    """Scatter of 2-D PCA coordinates coloured by (cluster or class) label."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=6, alpha=0.6)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    if class_names is not None:
        cbar = fig.colorbar(sc, ax=ax, ticks=range(len(class_names)))
        cbar.ax.set_yticklabels(class_names, fontsize=7)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion(y_true, y_pred, class_names, title="Confusion matrix",
                   normalize=True, savepath=None):
    """Confusion-matrix heatmap (row-normalised by default)."""
    cm = confusion_matrix(y_true, y_pred).astype(float)
    if normalize:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thr = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}" if normalize else int(v),
                        ha="center", va="center", fontsize=6,
                        color="white" if v > thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def plot_curve(x, y, xlabel, ylabel, title, marker="o", savepath=None):
    """Generic line plot (used for the elbow / BIC / training curves)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker=marker)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig
