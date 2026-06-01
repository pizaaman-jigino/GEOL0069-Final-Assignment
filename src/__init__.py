"""AI4EO EuroSAT land-cover classification package.

A compact, well-documented toolkit for an *AI for Earth Observation* (AI4EO)
project: classifying Sentinel-2 imagery into land-cover classes with both
**unsupervised** (K-means, Gaussian Mixture Models) and **supervised**
(Random Forest, a small CNN) machine-learning methods.

Modules
-------
data         : download / load the EuroSAT dataset (with a synthetic fallback)
features     : engineer per-image spectral features + PCA
unsupervised : K-means and Gaussian Mixture Model clustering
supervised   : Random Forest and a compact PyTorch CNN
evaluate     : metrics, cluster-to-label matching and plotting helpers
carbon       : estimate the energy use and CO2 footprint of a run
figures      : generate the explanatory technique / algorithm schematics
"""

__version__ = "1.0.0"

EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
