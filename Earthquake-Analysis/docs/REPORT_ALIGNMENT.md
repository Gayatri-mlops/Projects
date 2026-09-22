# Report alignment

## Primary source

The repository follows the **Earthquakes and Tectonic Plates Analysis – Final Project Report, Group 10, submitted 04/17/2025** as its primary specification.

The final report determines:

- the two data sources
- the cleaning workflow
- exploratory visualizations
- final input features
- output targets
- 60/40 train-validation split
- the five final model families
- the final neural-network architecture
- the metrics used for comparison

## Earlier milestones used only to fill missing implementation details

Where the final report does not specify code-level details, the following submitted milestones are used only when they do not contradict the final report:

- Data Collection and Processing
- Data Exploration and Visualization
- Model Exploration
- Model Implementation
- Performance Evaluation and Interpretation

Recovered details include:

- nearest-boundary feature calculated with scikit-learn `KDTree` on `[lon, lat]`
- KNN `n_neighbors=6`
- StandardScaler for KNN and SVR
- Random Forest `n_estimators=100`, `random_state=42`
- SVR `kernel='rbf'`, `C=100`, `epsilon=0.1`
- exact corrections for the three malformed Date/Time records

## Known evolution across milestones

The project evolved during the semester. Earlier milestones used an 80/20 split and in one stage excluded `Type` because of class imbalance. The final report supersedes those choices: it states a 60/40 split and lists Earthquake Type as an input feature. This repository therefore follows the final report for those items.

The earlier neural-network milestone mentions GridSearchCV, while the final report specifies the selected final architecture (one hidden layer, 10 neurons, ReLU, Adam, maximum 1000 iterations). This repository implements the final architecture rather than reproducing the exploratory grid search.
