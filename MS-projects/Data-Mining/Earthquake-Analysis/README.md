# Earthquakes and Tectonic Plates Analysis

Python reconstruction of the Northeastern University Group 10 project **Earthquakes and Tectonic Plates Analysis**.

The **Final Project Report (submitted April 17, 2025)** is treated as the source of truth for the repository. Earlier submitted milestones are used only to recover implementation details that the final report does not state and that do not conflict with it.

## What this repo reproduces

- earthquake records from 1965–2016
- tectonic-plate boundary data
- missing-value analysis and report-style data cleaning
- correction of the three malformed Date/Time rows
- exploratory magnitude/depth visualizations
- tectonic-plate maps for earthquake magnitude and depth
- nearest tectonic-boundary distance feature
- final-report model inputs: Latitude, Longitude, Timestamp, Distance To Boundary, Earthquake Type
- targets: Magnitude and Depth
- final-report 60% training / 40% validation split
- Linear Regression
- K-Nearest Neighbors (`k=6`)
- Random Forest Regression
- Support Vector Regression
- Neural Network (`MLPRegressor`, one hidden layer with 10 neurons, ReLU, Adam, `max_iter=1000`)
- residual histograms
- cumulative-gains and decile-lift charts
- comparison against the metrics printed in the final report

## Datasets

The project used two Kaggle datasets.

### 1. Significant Earthquakes, 1965–2016

- Kaggle: `usgs/earthquake-database`
- expected file: `database.csv`
- 23,412 rows × 21 raw columns
- https://www.kaggle.com/datasets/usgs/earthquake-database

### 2. Tectonic Plate Boundaries

- Kaggle: `cwthompson/tectonic-plate-boundaries`
- expected file: `all.csv`
- 12,321 rows × 3 columns: `plate`, `lat`, `lon`
- 56 tectonic plates described by the dataset/project
- https://www.kaggle.com/datasets/cwthompson/tectonic-plate-boundaries

The raw datasets are not committed to the repository. Download them with:

```bash
python scripts/download_data.py
```

## Quick start

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python scripts/download_data.py
python run_analysis.py
```

For a faster test run:

```bash
python run_analysis.py --skip-svr --skip-maps
```

Generated files are written to `outputs/`.

## Preprocessing reproduced from the project

The raw earthquake data contains missing values in several measurement/error columns. The project drops every column containing missing data. This leaves the 12 complete raw columns:

`Date`, `Time`, `Latitude`, `Longitude`, `Type`, `Depth`, `Magnitude`, `ID`, `Source`, `Location Source`, `Magnitude Source`, `Status`.

The three malformed records are corrected before conversion:

| Row | Date | Time |
|---:|---|---|
| 3378 | 02/23/1975 | 02:58:41 |
| 7512 | 04/28/1985 | 02:53:41 |
| 20650 | 03/13/2011 | 02:23:34 |

`Date` is converted to datetime, `Time` to timedelta, and a combined Unix `Timestamp` is created for modeling.

## Distance to tectonic-plate boundary

The model-exploration milestone contains the original implementation:

```python
plate_coords = df_plates[['lon', 'lat']].values
tree = KDTree(plate_coords)
earthquake_coords = df_cleaned[['Longitude', 'Latitude']].values
distances, indices = tree.query(earthquake_coords, k=1)
df_cleaned['distance_to_boundary'] = distances.flatten()
```

This repository reproduces that KDTree method. Because longitude/latitude are passed directly to the Euclidean KDTree, the resulting numeric feature is in coordinate-degree space. One project figure labels the distance axis in km, but the displayed implementation does not convert the values to kilometers; the code here follows the implementation.

## Final-report model configuration

**Inputs**

- Latitude
- Longitude
- Timestamp
- Distance To Boundary
- Earthquake Type

**Outputs**

- Magnitude
- Depth

**Split**

- 60% training: 14,047 samples
- 40% validation: 9,365 samples

Additional non-conflicting settings recovered from the submitted milestones:

- KNN: `n_neighbors=6`, numeric features standardized
- Random Forest: `n_estimators=100`, `random_state=42`
- SVR: RBF kernel, `C=100`, `epsilon=0.1`, numeric features standardized
- Neural network: numeric features standardized; final-report architecture is used

Categorical earthquake `Type` is one-hot encoded because the final report explicitly lists Earthquake Type as an input feature.

## Reproducibility note

This is a reconstruction from the submitted reports, not the lost original notebook. The final report does not explicitly print every code-level choice, including the exact train/validation random seed or categorical encoding method. Where the final report is silent, the repository uses non-conflicting settings visible in the earlier submitted project milestones and documents any remaining assumptions.

The exact values printed in the final report are stored in:

```text
docs/report_reference_metrics.csv
```

Running the full analysis produces:

```text
outputs/metrics_vs_report.csv
```

so reconstructed results can be compared directly against the submitted report.

## Repository structure

```text
.
├── data/
│   └── raw/
├── docs/
│   ├── DATA_SOURCES.md
│   ├── REPORT_ALIGNMENT.md
│   └── report_reference_metrics.csv
├── notebooks/
│   └── Earthquake_Tectonic_Plates_Analysis.ipynb
├── outputs/
├── scripts/
│   ├── download_data.py
│   └── smoke_test.py
├── src/
│   ├── data_processing.py
│   ├── features.py
│   ├── modeling.py
│   └── visualization.py
├── tests/
│   └── test_core.py
├── data_manifest.json
├── requirements.txt
└── run_analysis.py
```

## Scientific scope

This project analyzes historical seismic records and models magnitude/depth from engineered historical features. It should not be presented as an operational earthquake-warning or future-earthquake forecasting system.
