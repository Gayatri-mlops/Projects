from __future__ import annotations

import pandas as pd
from sklearn.neighbors import KDTree


def add_distance_to_boundary(
    earthquakes: pd.DataFrame,
    plates: pd.DataFrame,
) -> pd.DataFrame:
    """Add the project's nearest tectonic-boundary distance feature.

    The Group 10 model-exploration milestone shows the implementation used
    ``sklearn.neighbors.KDTree`` directly on ``[lon, lat]`` coordinates and
    queried the nearest boundary point for each earthquake.  This function
    reproduces that implementation so the GitHub code follows the submitted
    project workflow as closely as possible.

    Note: because the KDTree operates directly on longitude/latitude values,
    the numeric distance is Euclidean distance in coordinate-degree space.
    One milestone figure labels the axis in km, but the displayed code does
    not convert degrees to kilometers; this reconstruction follows the code.
    """
    plate_coords = plates[["lon", "lat"]].to_numpy(float)
    quake_coords = earthquakes[["Longitude", "Latitude"]].to_numpy(float)

    tree = KDTree(plate_coords)
    distances, indices = tree.query(quake_coords, k=1)

    result = earthquakes.copy()
    result["Distance To Boundary"] = distances.ravel()
    nearest = plates.iloc[indices.ravel()].reset_index(drop=True)
    result["Nearest Plate"] = nearest["plate"].to_numpy()
    return result


def model_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return the input and output features listed in the final report."""
    cols = ["Latitude", "Longitude", "Timestamp", "Distance To Boundary", "Type"]
    X = df[cols].copy()
    y_mag = df["Magnitude"].copy()
    y_depth = df["Depth"].copy()
    return X, y_mag, y_depth
