"""Generate the two explanatory schematics required by the brief.

* :func:`make_technique_figure` -- how Sentinel-2 acquires the multispectral
  imagery that EuroSAT is built from (the *remote-sensing technique*).
* :func:`make_algorithm_figure` -- the end-to-end machine-learning pipeline
  used in this project (the *AI algorithm and its implementation*).

Both are drawn purely with matplotlib so they regenerate deterministically and
need no external assets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _box(ax, x, y, w, h, text, fc, ec="#333333", fs=9, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.3, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def _arrow(ax, x0, y0, x1, y1, color="#444444"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.3, color=color))


def make_technique_figure(savepath="figures/01_sentinel2_technique.png"):
    """Schematic of Sentinel-2 multispectral acquisition -> EuroSAT patch."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- Left: acquisition geometry ----------------------------------------
    axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis("off")
    axL.set_title("(a) Sentinel-2 multispectral acquisition", fontsize=10)
    # satellite
    _box(axL, 6.6, 8.3, 2.6, 1.1, "Sentinel-2\n(MSI, 786 km orbit)", "#cfe3ff", fs=8)
    # sun
    axL.scatter([1.2], [9.1], s=900, color="#ffd34d", zorder=2)
    axL.text(1.2, 9.1, "Sun", ha="center", va="center", fontsize=8)
    # incoming / reflected rays
    _arrow(axL, 1.6, 8.7, 4.6, 2.3, "#f0a500")          # solar irradiance down
    _arrow(axL, 5.0, 2.3, 7.4, 8.2, "#3a7d3a")          # reflectance up to sensor
    axL.text(2.7, 5.8, "solar\nirradiance", fontsize=7, color="#b06f00", rotation=-58)
    axL.text(6.7, 5.6, "surface\nreflectance", fontsize=7, color="#2e642e", rotation=66)
    # ground
    axL.add_patch(plt.Rectangle((0.4, 0.6), 9.2, 1.7, facecolor="#d7c29a", edgecolor="#8a7448"))
    axL.text(5.0, 1.45, "land surface (vegetation / water / built-up)", ha="center", fontsize=7.5)
    axL.text(5.0, 0.15, "13 spectral bands, 10-60 m resolution, 5-day revisit",
             ha="center", fontsize=7.5, style="italic")

    # ---- Right: a spectral signature + the resulting patch -----------------
    axR.set_title("(b) Per-pixel spectral signature", fontsize=10)
    wl = np.array([443, 490, 560, 665, 705, 740, 783, 842, 945, 1375, 1610, 2190])
    veg = np.array([3, 4, 9, 5, 18, 33, 38, 42, 35, 4, 22, 12], dtype=float)
    water = np.array([6, 5, 4, 3, 2, 1.5, 1, 1, 0.8, 0.4, 0.6, 0.4], dtype=float)
    soil = np.array([8, 11, 16, 22, 26, 30, 32, 33, 34, 6, 38, 30], dtype=float)
    axR.plot(wl, veg, "-o", ms=3, color="#2e8b2e", label="vegetation")
    axR.plot(wl, soil, "-o", ms=3, color="#b5651d", label="bare soil")
    axR.plot(wl, water, "-o", ms=3, color="#1f6fb2", label="water")
    axR.axvspan(490, 665, color="#dddddd", alpha=0.5)
    axR.text(577, 44, "visible\n(RGB used here)", fontsize=6.5, ha="center")
    axR.set_xlabel("wavelength (nm)", fontsize=8)
    axR.set_ylabel("reflectance (%)", fontsize=8)
    axR.legend(fontsize=7, loc="upper right")
    axR.tick_params(labelsize=7)
    axR.grid(alpha=0.3)

    fig.suptitle("Remote-sensing technique: optical multispectral imaging with Sentinel-2",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def make_algorithm_figure(savepath="figures/02_pipeline_algorithm.png"):
    """Schematic of the unsupervised + supervised classification pipeline.

    Faithful to the code: K-means, GMM **and** the Random Forest all consume
    the *engineered features*; only the CNN consumes *raw pixels*. Box colour
    encodes supervision (blue = unsupervised, green = supervised).
    """
    blue, green = "#cfe3ff", "#d6f5d6"
    fig, ax = plt.subplots(figsize=(11.5, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")

    _box(ax, 0.2, 3.4, 2.0, 1.4, "EuroSAT\nSentinel-2 patches\n(64x64x3)", "#e8e8e8", fs=8)

    # Two input representations.
    _box(ax, 2.9, 4.7, 2.3, 1.1, "Engineered features\n(band stats, VARI)", "#fde9c9", fs=8)
    _box(ax, 2.9, 1.5, 2.3, 1.1, "Raw pixel tensor\n(normalised)", "#fde9c9", fs=8)

    # Models (colour = supervision).
    _box(ax, 6.0, 6.3, 2.4, 0.9, "K-means", blue, fs=9)
    _box(ax, 6.0, 5.0, 2.4, 0.9, "Gaussian Mixture", blue, fs=9)
    _box(ax, 6.0, 2.9, 2.4, 0.9, "Random Forest", green, fs=9)
    _box(ax, 6.0, 1.4, 2.4, 0.9, "Compact CNN (PyTorch)", green, fs=9)

    _box(ax, 9.4, 4.9, 2.3, 1.3, "Unsupervised eval\nARI / NMI / matched acc", blue, fs=7.5)
    _box(ax, 9.4, 1.5, 2.3, 1.3, "Supervised eval\naccuracy / confusion", green, fs=7.5)

    # EuroSAT -> representations
    _arrow(ax, 2.2, 4.4, 2.9, 5.2)
    _arrow(ax, 2.2, 3.8, 2.9, 2.1)
    # Features -> K-means, GMM, Random Forest
    _arrow(ax, 5.2, 5.5, 6.0, 6.7)
    _arrow(ax, 5.2, 5.3, 6.0, 5.4)
    _arrow(ax, 5.2, 4.9, 6.0, 3.4)
    # Raw pixels -> CNN
    _arrow(ax, 5.2, 2.0, 6.0, 1.85)
    # Models -> eval
    _arrow(ax, 8.4, 6.7, 9.4, 5.9)
    _arrow(ax, 8.4, 5.4, 9.4, 5.4)
    _arrow(ax, 8.4, 3.3, 9.4, 2.4)
    _arrow(ax, 8.4, 1.85, 9.4, 2.0)

    ax.text(7.2, 7.6, "UNSUPERVISED  (no labels used)", fontsize=9, color="#1f4e79",
            ha="center", weight="bold")
    ax.text(7.2, 0.55, "SUPERVISED  (trained on labels)", fontsize=9, color="#1e6b1e",
            ha="center", weight="bold")
    ax.text(1.2, 7.2, "engineered features feed all classical models;\nthe CNN learns directly from pixels",
            fontsize=7, style="italic", color="#555555", ha="left")
    ax.set_title("AI algorithm & implementation: dual unsupervised / supervised pipeline",
                 fontsize=11)
    fig.tight_layout()
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    make_technique_figure()
    make_algorithm_figure()
    print("Saved technique + algorithm figures to figures/")
