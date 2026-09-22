from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_raw_data(earthquake_csv: str | Path, plates_csv: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(earthquake_csv), pd.read_csv(plates_csv)


def validate_raw_shapes(
    earthquakes: pd.DataFrame,
    plates: pd.DataFrame,
    manifest_path: str | Path,
) -> None:
    manifest = json.loads(Path(manifest_path).read_text())
    eq_spec = manifest["earthquakes"]
    plate_spec = manifest["tectonic_plates"]

    if earthquakes.shape != (eq_spec["expected_rows"], eq_spec["expected_columns"]):
        raise ValueError(
            f"Unexpected earthquake dataset shape {earthquakes.shape}; "
            f"expected {(eq_spec['expected_rows'], eq_spec['expected_columns'])}."
        )
    if plates.shape != (plate_spec["expected_rows"], plate_spec["expected_columns"]):
        raise ValueError(
            f"Unexpected plate dataset shape {plates.shape}; "
            f"expected {(plate_spec['expected_rows'], plate_spec['expected_columns'])}."
        )


def null_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isna().sum()
    pct = 100 * counts / len(df)
    return pd.DataFrame({"null_count": counts, "null_percent": pct})


def clean_earthquakes(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the data-cleaning steps shown in the reports.

    1. Drop every column containing any null value, leaving 12 complete fields.
    2. Correct the three malformed date/time records documented in the project.
    3. Convert Date to datetime and Time to timedelta.
    4. Create Date_Time, Unix Timestamp, and Year features.
    """
    summary = null_summary(raw)
    clean = raw.drop(columns=summary.index[summary["null_count"] > 0]).copy()

    # Exact corrections shown in the project workflow / source dataset tutorial.
    date_fixes = {
        3378: "02/23/1975",
        7512: "04/28/1985",
        20650: "03/13/2011",
    }
    time_fixes = {
        3378: "02:58:41",
        7512: "02:53:41",
        20650: "02:23:34",
    }
    for idx, value in date_fixes.items():
        if idx in clean.index:
            clean.loc[idx, "Date"] = value
    for idx, value in time_fixes.items():
        if idx in clean.index:
            clean.loc[idx, "Time"] = value

    clean["Date"] = pd.to_datetime(clean["Date"], format="%m/%d/%Y")
    clean["Time"] = pd.to_timedelta(clean["Time"])
    clean["Date_Time"] = clean["Date"] + clean["Time"]
    clean["Timestamp"] = clean["Date_Time"].astype("int64") // 10**9
    clean["Year"] = clean["Date"].dt.year
    return clean, summary


def clean_plates(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"plate", "lat", "lon"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Plate dataset missing columns: {sorted(missing)}")
    plates = raw[["plate", "lat", "lon"]].dropna().copy()
    plates["lat"] = pd.to_numeric(plates["lat"], errors="raise")
    plates["lon"] = pd.to_numeric(plates["lon"], errors="raise")
    return plates
