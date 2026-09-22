# Data sources

The submitted final project report uses two datasets.

## Significant Earthquakes, 1965–2016

- Kaggle handle: `usgs/earthquake-database`
- Expected file: `database.csv`
- Expected shape: 23,412 rows × 21 columns
- URL: https://www.kaggle.com/datasets/usgs/earthquake-database

## Tectonic Plate Boundaries

- Kaggle handle: `cwthompson/tectonic-plate-boundaries`
- Expected file: `all.csv`
- Expected shape: 12,321 rows × 3 columns
- Columns: `plate`, `lat`, `lon`
- URL: https://www.kaggle.com/datasets/cwthompson/tectonic-plate-boundaries

Run `python scripts/download_data.py` to fetch both public datasets with `kagglehub`.
