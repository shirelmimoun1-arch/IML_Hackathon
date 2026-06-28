# Bike Demand Forecasting – IML Hackathon

**Team**

Yaniv Hayoun, Shirel Mimoun, Michal Segal, Ori Tzairi

---

# Overview

This project was developed as part of the **Introduction to Machine Learning Hackathon** at the Hebrew University of Jerusalem.

The objective of the challenge was to predict the **hourly bike demand** for every rental station. 
Given a city, station and hour, the model predicts how many bike rides are expected to start from that station during that hour.

The challenge was designed to evaluate not only prediction accuracy, but also the ability to 
**generalize to future time periods and previously unseen cities**.

The evaluation metric was **Mean Absolute Error (MAE)**.

---

# The Challenge

The original dataset consisted of **ride-level data**, where each row represented a single bike ride.

The available data included:

* Approximately **two months of historical rides** from **City 1** and **City 2**.
* Only a **small amount of historical data** from **City 3**.

The final competition evaluated the model on **future time periods**, and also included predictions for 
**new cities with limited historical information**.

Therefore, simply memorizing stations or learning city-specific patterns was unlikely to generalize well.

---

# Our Approach

Rather than treating the problem as a standard regression task, we built a complete machine 
learning pipeline that transforms raw ride data into hourly station-demand predictions.

The pipeline includes:

1. Aggregating individual rides into station-hour demand.
2. Creating missing demand=0 examples.
3. Cleaning and imputing missing values.
4. Feature engineering.
5. Training a Histogram Gradient Boosting model.
6. Saving all preprocessing artifacts together with the trained model for inference.

---

# Data Processing Pipeline

The original dataset contains one row per bike ride.

Our preprocessing converts it into one row per **station-hour**.

```
Raw Bike Rides
        │
        ▼
Aggregate rides per station-hour
        │
        ▼
Create missing demand=0 rows
        │
        ▼
Merge weather information
        │
        ▼
Merge station metadata
        │
        ▼
Feature Engineering
        │
        ▼
Gradient Boosting Regressor
        │
        ▼
Hourly Bike Demand Prediction
```

---

# Feature Engineering

The final model uses several groups of features.

## Time Features

* hour
* weekday
* weekend
* working_day

These capture the strong temporal patterns that naturally exist in commuting behavior.

---

## Holiday Features

Holiday names were one-hot encoded.

This allows the model to distinguish between regular weekdays, weekends and different holidays.

---

## Historical Demand Features (Target Encoding)

To capture recurring demand patterns we computed:

* city_hour_mean
* city_weekday_mean

For example:

* Average demand in City 1 at 08:00.
* Average demand in City 2 on Mondays.

If the model encounters an unseen city, it automatically falls back to global averages computed from the training data.

---

## Weather Features

* temperature
* relative humidity
* apparent temperature
* precipitation
* rain
* snowfall
* cloud cover
* wind speed

Weather strongly affects bike usage and therefore provides valuable predictive information.

---

## Station & Infrastructure Features

Each station is described by nearby infrastructure:

* bike lane length
* park area
* university count
* office POI count
* retail POI count
* restaurant & cafe count
* transit stop count
* distance to nearest railway station
* distance to city center

These features describe how attractive or busy the station's surroundings are.

---

# Design Decisions

## Time-Based Train / Test Split

One of the most important design decisions was **how to evaluate the model locally**.

Since the competition requires predicting **future demand**, randomly splitting the data would leak 
future information into the training set.

Instead, we:

* Sorted the data chronologically.
* Used the **first 80% of the timeline for training**.
* Used the **last 20% for testing**.

This setup better simulates the real deployment scenario, where the model is always asked to predict 
future hours using only past observations.

---

## City 3 Generalization

Another important design decision concerned **City 3**.

Because only a small amount of City 3 data was provided, and the competition emphasized generalization to unseen cities, 
we decided to initially keep City 3 **completely separate** from the main training process.

This allowed us to evaluate how well our model generalized to a city it had never seen before.

In later iterations of the project, we planned to use this held-out city to develop a more robust model 
capable of transferring knowledge between cities and improving performance on completely new urban environments.

---

## Why We Did Not Use Station ID

Station IDs were intentionally **excluded** from the feature set.

Using station identifiers directly would encourage the model to memorize specific stations rather than learn general demand patterns.

Instead, we represented stations using descriptive metadata such as nearby infrastructure, weather and historical demand statistics.

This improves the model's ability to generalize to stations and cities that were not observed during training.

---

## Creating Demand = 0 Examples

The original ride dataset only contains hours during which rides actually occurred.

However, the evaluation dataset also includes hours where **no rides happened**.

To match the evaluation format, we reconstructed each station's active timeline.

For every station we:

* Found its first recorded ride.
* Found its last recorded ride.
* Added a 24-hour buffer before and after.
* Generated every missing hour within this interval.
* Assigned demand = 0 whenever no rides occurred.

This significantly improved the realism of the training data.

---

## Missing Values

Missing infrastructure and location features were imputed using:

1. City median
2. Global median (fallback)

Both statistics were computed exclusively from the training data to prevent data leakage.

---

## Outlier Handling

Values outside physically reasonable ranges (temperature, humidity, precipitation, etc.) were replaced using training-set medians.

---

## Hyperparameter Search

Multiple combinations of Gradient Boosting hyperparameters were evaluated using the local validation framework.

The best performing configuration was:

* max_iter = 1000
* learning_rate = 0.05
* max_depth = 10
* max_leaf_nodes = 127
* min_samples_leaf = 20

---

# Model

We selected **Histogram-based Gradient Boosting Regressor** (`HistGradientBoostingRegressor` from scikit-learn).

This model was chosen because it:

* captures complex nonlinear relationships,
* handles heterogeneous numerical features,
* requires minimal preprocessing,
* trains efficiently on large datasets,
* performs well on structured tabular data.

---

# Results

Local validation performance:

| Dataset |       MAE |
| ------- | --------: |
| Overall | **0.964** |
| City 1  |     1.228 |
| City 2  |     0.654 |
| City 3  |     0.995 |

The results demonstrate that the model successfully captures temporal demand patterns while 
maintaining good generalization across different cities.

---

# Repository Structure

```
train.py
```

Builds the complete preprocessing pipeline, trains the model from scratch and saves all learned artifacts.

```
model.py
```

Contains feature engineering, preprocessing, imputation logic and prediction code.

```
predict.py
```

Competition inference wrapper.

```
weights.joblib
```

Serialized trained model together with all preprocessing artifacts.

---

# Reproducing the Model

From the submission directory:

```bash
cd submissions/my_team
python train.py
```

The script:

* reads the training data,
* performs preprocessing,
* trains the Gradient Boosting model,
* computes all required artifacts,
* saves the trained model to `weights.joblib`.

Training takes approximately **5 minutes**.

---

# Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* HistGradientBoostingRegressor
* Joblib

---

## Future Improvements

Given additional development time, we would explore further feature engineering to improve the model's 
ability to capture complex demand patterns. Possible additions include:

* **Cyclical time features** (sine/cosine encoding of the hour).
* **Feature interactions**, such as combining temperature and rainfall, since the same temperature can lead 
to different demand under different weather conditions.
* **Historical demand features**, such as demand from the previous day or the same day in the previous week.

We would also perform a more thorough **feature correlation analysis** (e.g., temperature vs. apparent temperature, 
precipitation vs. rain). Due to the limited duration of the hackathon, we did not remove correlated features, 
as tree-based models such as Histogram Gradient Boosting are generally robust to multicollinearity and can naturally 
select the most informative feature during training.

---
