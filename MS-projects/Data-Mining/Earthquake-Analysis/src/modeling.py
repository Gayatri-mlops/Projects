from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

NUMERIC = ["Latitude", "Longitude", "Timestamp", "Distance To Boundary"]
CATEGORICAL = ["Type"]


@dataclass
class ModelResult:
    model: str
    target: str
    split: str
    ME: float
    RMSE: float
    MAE: float
    MPE: float | None
    MAPE: float | None


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_builders(random_state: int = 42) -> dict[str, Callable[[], Pipeline]]:
    """Build the five regression models used in the submitted project.

    Final-report settings take precedence.  Where the final report is silent,
    non-conflicting implementation details are taken from the submitted
    milestones:
      * KNN: n_neighbors=6 and StandardScaler
      * Random Forest: n_estimators=100, random_state=42
      * SVR: RBF kernel, C=100, epsilon=0.1, scaled inputs
      * Neural network: one hidden layer with 10 neurons, ReLU, Adam,
        max_iter=1000 (final report); scaled numeric inputs for stability
    """
    return {
        "Linear Regression": lambda: Pipeline(
            [("prep", _preprocessor(False)), ("model", LinearRegression())]
        ),
        "KNN (k=6)": lambda: Pipeline(
            [("prep", _preprocessor(True)), ("model", KNeighborsRegressor(n_neighbors=6))]
        ),
        "Random Forest Regression": lambda: Pipeline(
            [
                ("prep", _preprocessor(False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Support Vector Regression": lambda: Pipeline(
            [
                ("prep", _preprocessor(True)),
                ("model", SVR(kernel="rbf", C=100, epsilon=0.1, cache_size=1000)),
            ]
        ),
        "Neural Network": lambda: Pipeline(
            [
                ("prep", _preprocessor(True)),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(10,),
                        activation="relu",
                        solver="adam",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    include_pct: bool,
) -> dict:
    """Compute the same error statistics shown in the final report.

    Residual/error sign is defined as actual - predicted, matching the report's
    residual plots and mean-error convention.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    out = {
        "ME": float(np.mean(errors)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MPE": None,
        "MAPE": None,
    }
    if include_pct:
        mask = y_true != 0
        pct = 100.0 * errors[mask] / y_true[mask]
        out["MPE"] = float(np.mean(pct))
        out["MAPE"] = float(np.mean(np.abs(pct)))
    return out


def fit_all_models(
    X: pd.DataFrame,
    y_mag: pd.Series,
    y_depth: pd.Series,
    random_state: int = 42,
    test_size: float = 0.40,
    skip_svr: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Fit all report models with the final report's 60/40 split."""
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    results: list[ModelResult] = []
    fitted: dict = {}

    for target_name, y in [("Magnitude", y_mag), ("Depth", y_depth)]:
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        fitted[target_name] = {}

        for model_name, builder in model_builders(random_state).items():
            if skip_svr and model_name == "Support Vector Regression":
                continue
            print(f"Training {model_name} for {target_name}...")
            pipe = builder()
            pipe.fit(X_train, y_train)
            fitted[target_name][model_name] = pipe

            pred_train = pipe.predict(X_train)
            pred_val = pipe.predict(X_val)

            include_pct = target_name == "Magnitude"
            for split_name, truth, pred in [
                ("train", y_train, pred_train),
                ("validation", y_val, pred_val),
            ]:
                m = regression_metrics(truth, pred, include_pct=include_pct)
                results.append(ModelResult(model_name, target_name, split_name, **m))

            fitted[target_name][model_name + "__validation"] = {
                "y_true": y_val.to_numpy(),
                "y_pred": pred_val,
            }

    return pd.DataFrame([r.__dict__ for r in results]), fitted
