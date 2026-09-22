from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing import clean_earthquakes
from src.features import add_distance_to_boundary, model_features
from src.modeling import fit_all_models

rng = np.random.default_rng(42)
n = 180
raw = pd.DataFrame({
    "Date": ["01/02/1965"] * n,
    "Time": ["13:44:18"] * n,
    "Latitude": rng.uniform(-60, 60, n),
    "Longitude": rng.uniform(-170, 170, n),
    "Type": rng.choice(["Earthquake", "Nuclear Explosion", "Explosion", "Rock Burst"], n),
    "Depth": rng.gamma(2.0, 30.0, n),
    "Depth Error": [np.nan] * n,
    "Magnitude": rng.normal(5.9, 0.35, n),
    "ID": [f"X{i}" for i in range(n)],
    "Source": ["US"] * n,
    "Location Source": ["US"] * n,
    "Magnitude Source": ["US"] * n,
    "Status": ["Reviewed"] * n,
})
plates = pd.DataFrame({
    "plate": ["demo"] * 50,
    "lat": np.linspace(-60, 60, 50),
    "lon": np.linspace(-160, 160, 50),
})
clean, _ = clean_earthquakes(raw)
clean = add_distance_to_boundary(clean, plates)
X, ym, yd = model_features(clean)
metrics, _ = fit_all_models(X, ym, yd, random_state=42, skip_svr=True)
assert not metrics.empty
print(metrics.head().to_string(index=False))
print("\nSmoke test passed.")
