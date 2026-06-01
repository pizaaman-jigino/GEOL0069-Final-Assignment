"""Feature engineering for classical (non-deep) machine learning.

Deep networks learn features directly from pixels, but the unsupervised
methods (K-means, GMM) and the Random Forest in this project operate on a
compact, interpretable feature vector summarising each image patch:

* per-channel mean and standard deviation (brightness + texture),
* the VARI vegetation index, computed from visible bands only
  (Gitelson et al., 2002) -- a proxy for the NDVI we would use if the
  near-infrared band were available,
* a simple greenness ratio and overall brightness.

These features are then standardised and, optionally, projected with PCA for
2-D visualisation of the clusters.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# EuroSAT RGB channel order is R, G, B (from Sentinel-2 bands B04, B03, B02).
R, G, B = 0, 1, 2


def image_features(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Compute a per-image feature matrix from image patches.

    Parameters
    ----------
    X : array, shape (N, H, W, C)
        Image patches, uint8 or float.

    Returns
    -------
    feats : float array, shape (N, F)
    names : list[str]  -- human-readable feature names
    """
    Xf = X.astype(np.float32) / 255.0
    means = Xf.mean(axis=(1, 2))          # (N, C)
    stds = Xf.std(axis=(1, 2))            # (N, C)

    r, g, b = means[:, R], means[:, G], means[:, B]
    eps = 1e-6
    # Visible Atmospherically Resistant Index (NIR-free vegetation proxy).
    vari = (g - r) / (g + r - b + eps)
    greenness = g / (r + g + b + eps)
    brightness = means.mean(axis=1)

    feats = np.column_stack([means, stds, vari, greenness, brightness])
    names = [
        "mean_R", "mean_G", "mean_B",
        "std_R", "std_G", "std_B",
        "VARI", "greenness", "brightness",
    ]
    return feats.astype(np.float32), names


def standardize(train: np.ndarray, *others: np.ndarray):
    """Z-score features, fitting the scaler on ``train`` only.

    Returns ``(scaler, train_scaled, *others_scaled)``.
    """
    scaler = StandardScaler().fit(train)
    out = [scaler.transform(train)] + [scaler.transform(o) for o in others]
    return (scaler, *out)


def pca_2d(feats_scaled: np.ndarray, seed: int = 42):
    """Project standardised features to 2-D with PCA (for plotting)."""
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(feats_scaled)
    return coords, pca
