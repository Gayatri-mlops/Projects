from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_processing import clean_earthquakes, clean_plates, load_raw_data, validate_raw_shapes
from src.features import add_distance_to_boundary, model_features
from src.modeling import fit_all_models
from src.visualization import (
    save_depth_map,
    save_eda_plots,
    save_magnitude_map,
    save_regression_gains_lift,
    save_residual_plot,
)

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Earthquake and tectonic-plate analysis reconstructed from the Group 10 final report")
    parser.add_argument(
        "--skip-svr",
        action="store_true",
        help="Skip SVR to make a quick test run faster. Do not use for the full report reproduction.",
    )
    parser.add_argument(
        "--skip-maps",
        action="store_true",
        help="Skip Folium HTML map generation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Train/validation and stochastic-model seed. The final report does not print a seed; 42 is retained from earlier submitted model code.",
    )
    args = parser.parse_args()

    eq_path = RAW / "database.csv"
    plate_path = RAW / "all.csv"
    if not eq_path.exists() or not plate_path.exists():
        raise SystemExit(
            "Missing data/raw/database.csv or data/raw/all.csv. "
            "Run `python scripts/download_data.py` first."
        )

    OUT.mkdir(exist_ok=True)
    eq_raw, plate_raw = load_raw_data(eq_path, plate_path)
    validate_raw_shapes(eq_raw, plate_raw, ROOT / "data_manifest.json")

    eq, nulls = clean_earthquakes(eq_raw)
    plates = clean_plates(plate_raw)
    nulls.to_csv(OUT / "null_summary.csv")
    eq.head(100).to_csv(OUT / "cleaned_data_preview.csv", index=False)

    print("Cleaned earthquake shape:", eq.shape)
    print("Plate shape:", plates.shape)
    print("Event types:", eq["Type"].value_counts().to_dict())

    # Distance feature follows the submitted KDTree implementation.
    eq = add_distance_to_boundary(eq, plates)
    eq[["Latitude", "Longitude", "Distance To Boundary", "Nearest Plate"]].head(100).to_csv(
        OUT / "boundary_distance_preview.csv", index=False
    )

    save_eda_plots(eq, OUT)
    if not args.skip_maps:
        save_magnitude_map(eq, plates, OUT / "earthquake_magnitude_map.html")
        save_depth_map(eq, plates, OUT / "earthquake_depth_map.html")

    X, y_mag, y_depth = model_features(eq)
    metrics, fitted = fit_all_models(
        X,
        y_mag,
        y_depth,
        random_state=args.random_state,
        test_size=0.40,
        skip_svr=args.skip_svr,
    )
    metrics.to_csv(OUT / "reconstructed_metrics.csv", index=False)
    print("\nValidation metrics:\n")
    print(metrics[metrics["split"] == "validation"].to_string(index=False))

    for target, items in fitted.items():
        for key, payload in items.items():
            if not key.endswith("__validation"):
                continue
            model = key.removesuffix("__validation")
            slug = model.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
            target_slug = target.lower()
            y_true = payload["y_true"]
            y_pred = payload["y_pred"]
            save_residual_plot(
                y_true,
                y_pred,
                f"{target} Residuals",
                OUT / f"{slug}_{target_slug}_residuals.png",
            )
            save_regression_gains_lift(
                y_true,
                y_pred,
                f"{model} - {target}",
                OUT / f"{slug}_{target_slug}",
            )

    ref = pd.read_csv(ROOT / "docs" / "report_reference_metrics.csv")
    recon = metrics[metrics["split"] == "validation"].copy()
    comparison = recon.merge(
        ref[ref["split"] == "validation"],
        on=["model", "target", "split"],
        how="left",
        suffixes=("_reconstructed", "_report"),
    )
    comparison.to_csv(OUT / "metrics_vs_report.csv", index=False)
    print("\nSaved outputs to", OUT)


if __name__ == "__main__":
    main()
