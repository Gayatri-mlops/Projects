"""Download the exact two public Kaggle datasets named in the project report.

Usage:
    python scripts/download_data.py

The script uses kagglehub. Public datasets normally download without a Kaggle
API token. If Kaggle asks you to authenticate, follow kagglehub's prompt or set
up your Kaggle credentials, then rerun the script.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = json.loads((ROOT / "data_manifest.json").read_text())
RAW.mkdir(parents=True, exist_ok=True)


def copy_expected_file(download_dir: Path, expected_name: str, destination: Path) -> None:
    matches = list(download_dir.rglob(expected_name))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {expected_name!r} under Kaggle download directory {download_dir}"
        )
    shutil.copy2(matches[0], destination)
    print(f"Saved {destination.relative_to(ROOT)}")


def main() -> None:
    for key, spec in MANIFEST.items():
        print(f"Downloading {key}: {spec['kaggle_handle']}")
        path = Path(kagglehub.dataset_download(spec["kaggle_handle"]))
        copy_expected_file(path, spec["expected_filename"], RAW / spec["expected_filename"])

    print("\nDone. Run: python run_analysis.py")


if __name__ == "__main__":
    main()
