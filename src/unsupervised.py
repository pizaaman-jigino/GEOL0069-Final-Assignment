"""Unsupervised clustering: K-means and Gaussian Mixture Models.

These are the two unsupervised algorithms emphasised in the AI4EO class.
Both take the standardised per-image feature matrix and group the patches
*without ever seeing the labels*; we only use the labels afterwards, in
:mod:`src.evaluate`, to measure how well the discovered clusters line up with
the true land-cover classes.

* **K-means** partitions samples into ``k`` clusters by minimising the
  within-cluster sum of squared Euclidean distances. It assumes roughly
  spherical, equally sized clusters.
* **Gaussian Mixture Model (GMM)** is a soft, probabilistic generalisation:
  each cluster is a Gaussian with its own mean and (full) covariance, fitted
  by Expectation-Maximisation. It can capture elongated / correlated clusters
  and yields per-sample membership probabilities. Its Bayesian Information
  Criterion (BIC) gives a principled way to choose the number of components.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def run_kmeans(X: np.ndarray, n_clusters: int = 10, seed: int = 42):
    """Fit K-means and return ``(labels, model)``."""
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = model.fit_predict(X)
    return labels, model


def run_gmm(X: np.ndarray, n_components: int = 10, seed: int = 42,
            covariance_type: str = "full"):
    """Fit a Gaussian Mixture Model and return ``(labels, model)``."""
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=seed,
        max_iter=200,
        n_init=3,
    )
    labels = model.fit_predict(X)
    return labels, model


def kmeans_elbow(X: np.ndarray, k_range=range(2, 16), seed: int = 42):
    """Return inertia for each ``k`` (the classic elbow curve)."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        inertias.append(km.inertia_)
    return list(k_range), inertias


def gmm_bic(X: np.ndarray, k_range=range(2, 16), seed: int = 42,
            covariance_type: str = "full"):
    """Return the BIC for each candidate number of GMM components."""
    bics = []
    for k in k_range:
        gm = GaussianMixture(
            n_components=k, covariance_type=covariance_type,
            random_state=seed, max_iter=200, n_init=1,
        ).fit(X)
        bics.append(gm.bic(X))
    return list(k_range), bics
