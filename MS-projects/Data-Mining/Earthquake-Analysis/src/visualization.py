from __future__ import annotations

from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_eda_plots(df: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))
    sns.lineplot(x=df["Date"].dt.year, y=df["Magnitude"])
    plt.title("Earthquake Magnitudes Over the Years")
    plt.xlabel("Year")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig(out / "magnitude_over_years.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.lineplot(x=df["Date"].dt.year, y=df["Magnitude"], hue=df["Type"])
    plt.title("Earthquake Magnitude Over Years by Type")
    plt.xlabel("Year")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig(out / "magnitude_over_years_by_type.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.histplot(df["Depth"], kde=True)
    plt.title("Distribution of the Depth of Earthquake")
    plt.xlabel("Depth")
    plt.ylabel("Number of Earthquakes")
    plt.tight_layout()
    plt.savefig(out / "depth_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.histplot(df["Magnitude"], kde=True)
    plt.title("Distribution of the Magnitude of Earthquake")
    plt.xlabel("Magnitude")
    plt.ylabel("Number of Earthquakes")
    plt.tight_layout()
    plt.savefig(out / "magnitude_distribution.png", dpi=180)
    plt.close()

    # Additional EDA plots shown in the model-exploration milestone.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Magnitude", y="Depth", hue="Magnitude", palette="rocket_r", legend="brief")
    plt.title("Magnitude vs Depth of Earthquakes")
    plt.tight_layout()
    plt.savefig(out / "magnitude_vs_depth.png", dpi=180)
    plt.close()

    if "Distance To Boundary" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="Distance To Boundary", y="Magnitude", hue="Magnitude", palette="rocket_r", legend="brief")
        plt.title("Distance to Tectonic Plate Boundary vs Earthquake Magnitude")
        plt.tight_layout()
        plt.savefig(out / "magnitude_vs_distance_to_boundary.png", dpi=180)
        plt.close()


def _draw_plate_lines(m: folium.Map, plates: pd.DataFrame) -> None:
    """Draw plate polylines while splitting dateline jumps, as in the project workflow."""
    for plate, group in plates.groupby("plate", sort=False):
        points = list(zip(group["lat"].to_numpy(), group["lon"].to_numpy()))
        if len(points) < 2:
            continue
        split_points = [0]
        for i in range(len(points) - 1):
            if abs(points[i][1] - points[i + 1][1]) > 300:
                split_points.append(i + 1)
        split_points.append(len(points))
        for start, end in zip(split_points[:-1], split_points[1:]):
            segment = points[start:end]
            if len(segment) >= 2:
                folium.PolyLine(segment, popup=str(plate), color="#58508d", fill=False).add_to(m)


def save_magnitude_map(df: pd.DataFrame, plates: pd.DataFrame, output: str | Path) -> None:
    m = folium.Map(tiles="cartodbpositron", zoom_start=2)
    _draw_plate_lines(m, plates)

    def color_mag(value: float) -> str:
        if value < 6.0:
            return "#ffcf6a"
        if value < 6.5:
            return "#fb8270"
        return "#bc5090"

    for row in df.itertuples(index=False):
        folium.Circle(
            location=[row.Latitude, row.Longitude],
            radius=2000,
            color=color_mag(row.Magnitude),
            fill=False,
        ).add_to(m)
    m.save(output)


def save_depth_map(df: pd.DataFrame, plates: pd.DataFrame, output: str | Path) -> None:
    m = folium.Map(tiles="cartodbpositron", zoom_start=2)
    _draw_plate_lines(m, plates)

    def color_depth(value: float) -> str:
        if value < 50:
            return "#ffcf6a"
        if value < 100:
            return "#fb8270"
        return "#bc5090"

    for row in df.itertuples(index=False):
        folium.Circle(
            location=[row.Latitude, row.Longitude],
            radius=2000,
            color=color_depth(row.Depth),
            fill=False,
        ).add_to(m)
    m.save(output)


def save_residual_plot(y_true, y_pred, title: str, output: str | Path) -> None:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    plt.figure(figsize=(8, 4.5))
    plt.hist(residuals, bins=50)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def save_regression_gains_lift(y_true, y_pred, title: str, output_prefix: str | Path) -> None:
    """Create regression cumulative-gains and decile-lift charts.

    Validation observations are sorted by predicted target value from highest to
    lowest. Cumulative actual target values form the gains curve, while each
    decile's actual mean divided by the overall actual mean forms lift.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    order = np.argsort(-y_pred)
    actual = y_true[order]
    cumulative = np.cumsum(actual)
    x = np.arange(1, len(actual) + 1)
    baseline = np.linspace(0, cumulative[-1], len(actual))

    prefix = Path(output_prefix)
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, cumulative)
    plt.plot(x, baseline, linestyle="--")
    plt.title("Cumulative Gains Chart")
    plt.xlabel("# of records")
    plt.ylabel(f"Cumulative {title.split(' - ')[-1]}")
    plt.tight_layout()
    plt.savefig(prefix.with_name(prefix.name + "_gains.png"), dpi=180)
    plt.close()

    bins = np.array_split(actual, 10)
    global_mean = np.mean(y_true)
    lift = [np.mean(b) / global_mean if len(b) and global_mean else np.nan for b in bins]
    plt.figure(figsize=(7, 4.5))
    x_pct = np.arange(10, 101, 10)
    bars = plt.bar(x_pct, lift, width=7)
    for bar, value in zip(bars, lift):
        if np.isfinite(value):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    plt.title("Decile Lift Chart")
    plt.xlabel("Percentile")
    plt.ylabel(f"Cumulative {title.split(' - ')[-1]}")
    plt.tight_layout()
    plt.savefig(prefix.with_name(prefix.name + "_lift.png"), dpi=180)
    plt.close()
