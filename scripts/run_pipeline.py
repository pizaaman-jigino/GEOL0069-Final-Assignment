"""Run the full AI4EO pipeline from the command line and save artefacts.

This mirrors what the notebook does, but headless: it loads the data, runs the
unsupervised and supervised models, writes metrics to ``results/metrics.json``,
saves every figure to ``figures/``, and prints an environmental-cost report.

Usage
-----
    python scripts/run_pipeline.py --n-per-class 400 --epochs 12
    python scripts/run_pipeline.py --synthetic        # offline, no download
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for scripts
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import EUROSAT_CLASSES                                    # noqa: E402
from src import data, features, unsupervised, supervised, evaluate, carbon, figures  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Run the AI4EO EuroSAT pipeline.")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--n-per-class", type=int, default=400,
                    help="images sampled per class (None-like 0 = all)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--synthetic", action="store_true", help="force synthetic data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fig_dir = ROOT / "figures"; fig_dir.mkdir(exist_ok=True)
    res_dir = ROOT / "results"; res_dir.mkdir(exist_ok=True)
    npc = None if args.n_per_class in (0, -1) else args.n_per_class

    t_start = time.time()

    # ---- 1. schematics ------------------------------------------------------
    figures.make_technique_figure(str(fig_dir / "01_sentinel2_technique.png"))
    figures.make_algorithm_figure(str(fig_dir / "02_pipeline_algorithm.png"))

    # ---- 2. data ------------------------------------------------------------
    if args.synthetic:
        X, y, names = data.make_synthetic(n_per_class=npc or 300, seed=args.seed)
        source = "synthetic"
    else:
        X, y, names, source = data.get_dataset(
            args.data_dir, n_per_class=npc, seed=args.seed)
    print(f"\nData source: {source}   X={X.shape}  classes={len(names)}")
    evaluate.plot_sample_grid(X, y, names, savepath=str(fig_dir / "03_sample_patches.png"))

    # ---- 3. features + split ------------------------------------------------
    # Split into train / validation / test (60 / 15 / 25). The validation set
    # is used only for CNN model selection; the test set is held out for the
    # final, unbiased evaluation of every supervised model.
    feats, feat_names = features.image_features(X)
    Xtr_img, Xte_img, ytr, yte, Ftr, Fte = train_test_split(
        X, y, feats, test_size=0.25, stratify=y, random_state=args.seed)
    Xtr_img, Xva_img, ytr, yva, Ftr, _Fva = train_test_split(
        Xtr_img, ytr, Ftr, test_size=0.20, stratify=ytr, random_state=args.seed)
    scaler, Ftr_s, Fte_s = features.standardize(Ftr, Fte)
    coords_all, _ = features.pca_2d(scaler.transform(feats))

    results = {"data_source": source, "n_samples": int(len(y)),
               "n_per_class": npc, "classes": names}

    # ---- 4. unsupervised ----------------------------------------------------
    feats_scaled = scaler.transform(feats)
    # elbow (K-means) + BIC (GMM) for choosing the number of clusters
    ks, inertias = unsupervised.kmeans_elbow(feats_scaled, k_range=range(2, 16), seed=args.seed)
    kb, bics = unsupervised.gmm_bic(feats_scaled, k_range=range(2, 16), seed=args.seed)
    import matplotlib.pyplot as plt
    figeb, axeb = plt.subplots(1, 2, figsize=(11, 4))
    axeb[0].plot(ks, inertias, "o-"); axeb[0].axvline(10, ls="--", c="r", alpha=.6)
    axeb[0].set(title="K-means elbow", xlabel="k", ylabel="inertia"); axeb[0].grid(alpha=.3)
    axeb[1].plot(kb, bics, "o-"); axeb[1].axvline(10, ls="--", c="r", alpha=.6)
    axeb[1].set(title="GMM BIC (lower=better)", xlabel="components", ylabel="BIC"); axeb[1].grid(alpha=.3)
    figeb.tight_layout(); figeb.savefig(str(fig_dir / "04c_elbow_bic.png"), dpi=150, bbox_inches="tight")

    km_labels, _ = unsupervised.run_kmeans(feats_scaled, n_clusters=len(names), seed=args.seed)
    gmm_labels, _ = unsupervised.run_gmm(feats_scaled, n_components=len(names), seed=args.seed)
    results["kmeans"] = evaluate.clustering_scores(km_labels, y)
    results["gmm"] = evaluate.clustering_scores(gmm_labels, y)
    evaluate.plot_clusters_2d(coords_all, y, "PCA coloured by TRUE class", names,
                              savepath=str(fig_dir / "04a_pca_true.png"))
    evaluate.plot_clusters_2d(coords_all, km_labels, "PCA coloured by K-means cluster",
                              savepath=str(fig_dir / "04b_pca_kmeans.png"))

    # ---- 5. supervised: Random Forest --------------------------------------
    rf = supervised.train_random_forest(Ftr_s, ytr, seed=args.seed)
    rf_pred = rf.predict(Fte_s)
    results["random_forest"] = evaluate.supervised_scores(yte, rf_pred, names)
    evaluate.plot_confusion(yte, rf_pred, names, "Random Forest confusion matrix",
                            savepath=str(fig_dir / "05_rf_confusion.png"))
    # feature importances
    import pandas as pd
    imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values()
    figfi, axfi = plt.subplots(figsize=(6, 4))
    imp.plot.barh(ax=axfi); axfi.set_title("Random Forest feature importance")
    figfi.tight_layout(); figfi.savefig(str(fig_dir / "05b_rf_importance.png"), dpi=150, bbox_inches="tight")

    # ---- 6. supervised: CNN -------------------------------------------------
    with carbon.track_energy("cnn_train") as t_cnn:
        model, hist = supervised.train_cnn(
            Xtr_img, ytr, Xva_img, yva, n_classes=len(names),
            epochs=args.epochs, seed=args.seed)
    cnn_eval = supervised.evaluate_cnn(model, Xte_img, yte)
    results["cnn"] = {
        "accuracy": cnn_eval["accuracy"],
        "n_parameters": supervised.count_parameters(model),
        "history": hist,
        "train_seconds": t_cnn.report.seconds,
    }
    evaluate.plot_confusion(cnn_eval["y_true"], cnn_eval["y_pred"], names,
                            "CNN confusion matrix",
                            savepath=str(fig_dir / "06_cnn_confusion.png"))
    evaluate.plot_curve(range(1, len(hist["val_acc"]) + 1), hist["val_acc"],
                        "epoch", "validation accuracy", "CNN training curve",
                        savepath=str(fig_dir / "07_cnn_curve.png"))

    # ---- 6b. head-to-head comparison bar chart -----------------------------
    comp = [
        ("K-means", "unsup", results["kmeans"]["matched_accuracy"]),
        ("Gaussian Mixture", "unsup", results["gmm"]["matched_accuracy"]),
        ("Random Forest", "sup", results["random_forest"]["accuracy"]),
        ("Compact CNN", "sup", cnn_eval["accuracy"]),
    ]
    figc, axc = plt.subplots(figsize=(7, 4))
    axc.bar([c[0] for c in comp], [c[2] for c in comp],
            color=["#5b9bd5" if c[1] == "unsup" else "#70ad47" for c in comp])
    axc.axhline(0.1, ls="--", c="grey", label="random guess (1/10)")
    axc.set(ylabel="accuracy", title="Unsupervised vs supervised", ylim=(0, 1))
    axc.legend(); plt.setp(axc.get_xticklabels(), rotation=15)
    figc.tight_layout(); figc.savefig(str(fig_dir / "08_comparison.png"), dpi=150, bbox_inches="tight")

    # ---- 7. environmental cost ---------------------------------------------
    total_report = carbon.estimate_energy(time.time() - t_start)
    results["environmental_cost"] = total_report.to_dict()
    print("\n=== Environmental cost (whole pipeline) ===")
    print(total_report.summary())

    with open(res_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics written to {res_dir / 'metrics.json'}")
    print("Figures written to", fig_dir)


if __name__ == "__main__":
    main()
