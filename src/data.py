"""Data access for the EuroSAT Sentinel-2 land-cover dataset.

EuroSAT (Helber et al., 2019) is a benchmark of 27,000 georeferenced
64 x 64 pixel image patches extracted from Sentinel-2 multispectral scenes,
labelled with 10 land-use / land-cover classes. This module downloads the
dataset from a reliable mirror, loads it into NumPy arrays, and provides a
fully synthetic fallback so that the pipeline always runs even with no
internet access.

References
----------
Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019).
    EuroSAT: A novel dataset and deep learning benchmark for land use and
    land cover classification. IEEE JSTARS, 12(7), 2217-2226.
"""
from __future__ import annotations

import os
import ssl
import time
import zipfile
import urllib.request
from pathlib import Path

import numpy as np

from . import EUROSAT_CLASSES

# Mirrors tried in order. The torchgeo Hugging Face mirror is the most
# reliable in practice; the DFKI host is the original but needs relaxed SSL.
_MIRRORS = [
    ("https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip", True),
    ("https://madm.dfki.de/files/sentinel/EuroSAT.zip", False),
]


def _ssl_context(verify: bool) -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if not verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def download_eurosat(data_dir: str | os.PathLike = "data", verbose: bool = True) -> Path:
    """Download and extract the EuroSAT RGB dataset.

    Parameters
    ----------
    data_dir : path
        Directory in which to store ``EuroSAT_RGB.zip`` and the extracted
        ``2750/<class>/*.jpg`` tree.
    verbose : bool
        Print download progress.

    Returns
    -------
    Path
        The directory that directly contains the per-class sub-folders.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    root = _find_image_root(data_dir)
    if root is not None:
        if verbose:
            print(f"EuroSAT already present at: {root}")
        return root

    zip_path = data_dir / "EuroSAT_RGB.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        _download_zip(zip_path, verbose)

    if verbose:
        print("Extracting archive ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)

    root = _find_image_root(data_dir)
    if root is None:
        raise RuntimeError("Extraction finished but no class folders were found.")
    if verbose:
        print(f"EuroSAT ready at: {root}")
    return root


def _download_zip(zip_path: Path, verbose: bool) -> None:
    last_err = None
    for url, verify in _MIRRORS:
        try:
            if verbose:
                print(f"Downloading EuroSAT from {url}")
            ctx = _ssl_context(verify)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            t0 = time.time()
            with urllib.request.urlopen(req, context=ctx, timeout=60) as r, open(zip_path, "wb") as f:
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if verbose and total:
                        print(f"\r  {done/1e6:6.1f}/{total/1e6:6.1f} MB "
                              f"({time.time()-t0:4.0f}s)", end="", flush=True)
            if verbose:
                print()
            return
        except Exception as e:  # noqa: BLE001 - try the next mirror
            last_err = e
            if verbose:
                print(f"\n  mirror failed: {type(e).__name__}: {e}")
            if zip_path.exists():
                zip_path.unlink()
    raise RuntimeError(f"All EuroSAT mirrors failed. Last error: {last_err}")


def _find_image_root(data_dir: Path) -> Path | None:
    """Locate the folder that directly contains the 10 class sub-folders."""
    if not data_dir.exists():
        return None
    candidates = [data_dir, data_dir / "2750", data_dir / "EuroSAT"]
    candidates += [p for p in data_dir.iterdir() if p.is_dir()]
    for cand in candidates:
        if cand.is_dir() and all((cand / c).is_dir() for c in EUROSAT_CLASSES):
            return cand
    return None


def load_eurosat(
    data_dir: str | os.PathLike = "data",
    n_per_class: int | None = None,
    img_size: int = 64,
    seed: int = 42,
    download: bool = True,
    verbose: bool = True,
):
    """Load EuroSAT into memory as NumPy arrays.

    Parameters
    ----------
    n_per_class : int or None
        If given, randomly sample this many images per class (useful for fast
        CPU experiments). ``None`` loads the full dataset.
    img_size : int
        Side length the patches are resized to (EuroSAT is natively 64 px).
    seed : int
        Random seed controlling the per-class subsampling.
    download : bool
        Download the dataset if it is not present.

    Returns
    -------
    X : uint8 array, shape (N, H, W, 3)
    y : int array, shape (N,)  -- class indices into ``EUROSAT_CLASSES``
    class_names : list[str]
    """
    from PIL import Image

    data_dir = Path(data_dir)
    root = _find_image_root(data_dir)
    if root is None:
        if not download:
            raise FileNotFoundError("EuroSAT not found and download=False.")
        root = download_eurosat(data_dir, verbose=verbose)

    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, cls in enumerate(EUROSAT_CLASSES):
        files = sorted((root / cls).glob("*.jpg")) + sorted((root / cls).glob("*.png")) \
            + sorted((root / cls).glob("*.tif"))
        if n_per_class is not None and len(files) > n_per_class:
            idx = rng.choice(len(files), size=n_per_class, replace=False)
            files = [files[i] for i in sorted(idx)]
        for fp in files:
            img = Image.open(fp).convert("RGB")
            if img.size != (img_size, img_size):
                img = img.resize((img_size, img_size), Image.BILINEAR)
            X.append(np.asarray(img, dtype=np.uint8))
        if verbose:
            print(f"  loaded {len(files):5d} x {cls}")
        y.extend([ci] * len(files))

    X = np.stack(X).astype(np.uint8)
    y = np.asarray(y, dtype=np.int64)
    # Shuffle so that classes are interleaved.
    perm = rng.permutation(len(y))
    return X[perm], y[perm], list(EUROSAT_CLASSES)


def make_synthetic(
    n_per_class: int = 300,
    img_size: int = 64,
    seed: int = 42,
    verbose: bool = True,
):
    """Generate a synthetic EuroSAT-like dataset (offline fallback / CI).

    Each class is given a distinct mean RGB colour plus class-specific spatial
    texture (smooth fields, oriented stripes for roads/rivers, high-frequency
    speckle for built-up areas). This is *not* real data -- it only exists so
    the full pipeline can be exercised end-to-end without a download.

    Returns the same ``(X, y, class_names)`` signature as :func:`load_eurosat`.
    """
    rng = np.random.default_rng(seed)
    # Plausible mean colours (R, G, B) per class, 0-255.
    palette = {
        "AnnualCrop":           (150, 160,  70),
        "Forest":               ( 40,  90,  45),
        "HerbaceousVegetation": (110, 150,  80),
        "Highway":              (120, 120, 120),
        "Industrial":           (160, 150, 150),
        "Pasture":              (120, 165,  90),
        "PermanentCrop":        (130, 145,  75),
        "Residential":          (170, 140, 130),
        "River":                ( 60, 110, 150),
        "SeaLake":              ( 30,  70, 140),
    }
    texture = {
        "Highway": "stripe", "River": "stripe",
        "Industrial": "speckle", "Residential": "speckle",
    }
    X, y = [], []
    H = W = img_size
    yy, xx = np.mgrid[0:H, 0:W]
    for ci, cls in enumerate(EUROSAT_CLASSES):
        base = np.array(palette[cls], dtype=np.float32)
        kind = texture.get(cls, "smooth")
        for _ in range(n_per_class):
            img = np.ones((H, W, 3), dtype=np.float32) * base
            if kind == "smooth":
                lowf = rng.normal(0, 18, (8, 8, 3)).astype(np.float32)
                lowf = np.kron(lowf, np.ones((H // 8, W // 8, 1)))
                img += lowf[:H, :W]
            elif kind == "stripe":
                ang = rng.uniform(0, np.pi)
                band = np.sin((xx * np.cos(ang) + yy * np.sin(ang)) * 0.6 + rng.uniform(0, 6))
                img += (band[..., None] * np.array([35, 35, 35]))
            elif kind == "speckle":
                img += rng.normal(0, 45, (H, W, 3)).astype(np.float32)
            img += rng.normal(0, 6, (H, W, 3)).astype(np.float32)
            X.append(np.clip(img, 0, 255).astype(np.uint8))
        if verbose:
            print(f"  synth {n_per_class:5d} x {cls}")
        y.extend([ci] * n_per_class)
    X = np.stack(X)
    y = np.asarray(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], list(EUROSAT_CLASSES)


def get_dataset(
    data_dir: str | os.PathLike = "data",
    n_per_class: int | None = 300,
    img_size: int = 64,
    seed: int = 42,
    allow_synthetic: bool = True,
    verbose: bool = True,
):
    """Convenience loader: try real EuroSAT, fall back to synthetic.

    Returns ``(X, y, class_names, source)`` where ``source`` is ``"eurosat"``
    or ``"synthetic"`` so callers (and the environmental-cost log) know which
    data path was taken.
    """
    try:
        X, y, names = load_eurosat(
            data_dir, n_per_class=n_per_class, img_size=img_size,
            seed=seed, download=True, verbose=verbose,
        )
        return X, y, names, "eurosat"
    except Exception as e:  # noqa: BLE001
        if not allow_synthetic:
            raise
        if verbose:
            print(f"[warn] Falling back to synthetic data ({type(e).__name__}: {e})")
        n = n_per_class or 300
        X, y, names = make_synthetic(n_per_class=n, img_size=img_size, seed=seed, verbose=verbose)
        return X, y, names, "synthetic"
