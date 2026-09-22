import numpy as np
import pandas as pd

from src.data_processing import clean_earthquakes
from src.features import add_distance_to_boundary
from src.modeling import model_builders, regression_metrics


def _raw_eq():
    return pd.DataFrame(
        {
            "Date": ["01/02/1965", "01/04/1965"],
            "Time": ["13:44:18", "11:29:49"],
            "Latitude": [19.246, 1.863],
            "Longitude": [145.616, 127.352],
            "Type": ["Earthquake", "Earthquake"],
            "Depth": [131.6, 80.0],
            "Depth Error": [np.nan, np.nan],
            "Magnitude": [6.0, 5.8],
            "ID": ["A", "B"],
            "Source": ["ISCGEM", "ISCGEM"],
            "Location Source": ["ISCGEM", "ISCGEM"],
            "Magnitude Source": ["ISCGEM", "ISCGEM"],
            "Status": ["Automatic", "Automatic"],
        }
    )


def test_cleaning_drops_null_columns_and_creates_timestamp():
    clean, _ = clean_earthquakes(_raw_eq())
    assert "Depth Error" not in clean.columns
    assert str(clean["Date"].dtype).startswith("datetime64")
    assert str(clean["Time"].dtype).startswith("timedelta64")
    assert "Timestamp" in clean.columns


def test_boundary_distance_zero_for_exact_point():
    eq, _ = clean_earthquakes(_raw_eq())
    plates = pd.DataFrame({"plate": ["x"], "lat": [19.246], "lon": [145.616]})
    out = add_distance_to_boundary(eq.iloc[[0]], plates)
    assert out.iloc[0]["Distance To Boundary"] < 1e-12
    assert out.iloc[0]["Nearest Plate"] == "x"


def test_metrics_use_actual_minus_predicted_error_sign():
    m = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 3.0]), include_pct=True)
    assert abs(m["ME"] + 0.5) < 1e-9
    assert abs(m["MAE"] - 0.5) < 1e-9


def test_report_hyperparameters():
    builders = model_builders(random_state=42)
    assert builders["KNN (k=6)"]().named_steps["model"].n_neighbors == 6
    rf = builders["Random Forest Regression"]().named_steps["model"]
    assert rf.n_estimators == 100
    svr = builders["Support Vector Regression"]().named_steps["model"]
    assert svr.kernel == "rbf" and svr.C == 100 and svr.epsilon == 0.1
    mlp = builders["Neural Network"]().named_steps["model"]
    assert mlp.hidden_layer_sizes == (10,)
    assert mlp.activation == "relu"
    assert mlp.solver == "adam"
    assert mlp.max_iter == 1000
