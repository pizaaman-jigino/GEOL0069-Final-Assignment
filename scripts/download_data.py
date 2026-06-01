"""Download (and extract) the EuroSAT RGB dataset.

Usage
-----
    python scripts/download_data.py [--data-dir data]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import download_eurosat  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Download the EuroSAT RGB dataset.")
    ap.add_argument("--data-dir", default="data", help="where to store the data")
    args = ap.parse_args()
    root = download_eurosat(args.data_dir, verbose=True)
    print(f"\nDone. Class folders are under: {root}")


if __name__ == "__main__":
    main()
