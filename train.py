#!/usr/bin/env python3
"""
Training script for the bike-demand submission.

Workflow:
    1. Read local_train_set.csv (ride-level data, each row = one bike ride)
    2. Aggregate to station-hour demand table INCLUDING demand=0 rows
       (active-window grid: first_ride - 24h buffer to last_ride + 24h buffer, all hours 0-23)
    3. Compute artifacts from training data:
       - Demand averages: city_hour_mean, city_weekday_mean, global_hour_mean, global_weekday_mean
       - Imputation values: city median + global median for POI/location features
       - Column medians: for replacing out-of-range values
    4. Build features using model.create_features()
    5. Train HistGradientBoostingRegressor (1000 iter, lr=0.05, depth=10, leaves=127)
    6. Save model + all artifacts to weights.joblib

Run from this folder:
    cd submissions/my_team
    python train.py


Current validation MAE: ~0.964
"""

from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from model import BikeDemandModel, VALID_RANGES, FEATURES_TO_IMPUTE


DATA_ROOT = Path("../../dataset")
TRAIN_CSV = DATA_ROOT / "local_train_set.csv"
OUTPUT_WEIGHTS = "weights.joblib"

KEYS = ["city", "start_station_id", "hour_ts"]

RIDE_ONLY_COLS = {
    "started_at",
    "ended_at",
    "end_station_id",
    "usage_time_minutes",
    "distance_meters",
    "user_type",
}


# ---------------------------------------------------------------------------
# Helper functions for data preparation
# ---------------------------------------------------------------------------

def normalize_station_id(s):
    """Normalize station IDs: 31631.0 → '31631', missing → '__missing_station__'."""
    raw = s.astype("string").str.strip()
    num = pd.to_numeric(raw, errors="coerce")
    is_int_like = num.notna() & np.isfinite(num) & (num % 1 == 0)
    out = raw.copy()
    out.loc[is_int_like] = num.loc[is_int_like].astype("int64").astype("string")
    return out.fillna("__missing_station__")


def first_non_null(x):
    """Return first non-null value in a group."""
    x = x.dropna()
    if len(x) == 0:
        return pd.NA
    return x.iloc[0]


def prepare_training_data():
    """
    Aggregate ride-level data -> station-hour demand table, INCLUDING demand=0 rows.

    Steps:
        1. Read rides, clean timestamps, normalize station IDs
        2. Compute demand (count rides per station-hour)
        3. Build station metadata (POI, location - one row per station)
        4. Build city-hour metadata (weather, calendar - one row per city+hour)
        5. Build active-window grid: for each station, all hours between its
           first and last ride (filtered to 6:00-22:00)
        6. Left-join demand onto grid -> missing = demand 0
        7. Left-join station metadata + city-hour metadata onto grid

    This ensures the model sees demand=0 examples during training,
    matching the evaluation format.
    """
    df = pd.read_csv(TRAIN_CSV, low_memory=False)

    # Clean hour_ts to hourly granularity
    df["hour_ts"] = pd.to_datetime(df["hour_ts"], errors="coerce").dt.floor("h")

    # Cannot create training examples without these
    df = df.dropna(subset=["city", "start_station_id", "hour_ts"])

    # Normalize station ids: 1072.0 -> "1072"
    df["start_station_id"] = normalize_station_id(df["start_station_id"])

    # ---------------------------------------------------------------
    # Step 2: Compute demand (rides per station-hour)
    # ---------------------------------------------------------------
    demand = (
        df.groupby(KEYS, dropna=False)
        .size()
        .reset_index(name="demand")
    )

    # ---------------------------------------------------------------
    # Step 3: Station metadata (static per station - POI, location)
    # ---------------------------------------------------------------
    station_meta_cols = [
        "start_lat", "start_lng", "bike_lane_length_500m", "park_area_500m",
        "university_count_1000m", "office_poi_count_1000m",
        "retail_poi_count_1000m", "restaurant_cafe_count_500m",
        "transit_stop_count_500m", "distance_to_nearest_rail_station",
        "distance_to_city_center",
    ]
    existing_station_cols = [c for c in station_meta_cols if c in df.columns]

    station_meta = (
        df.groupby(["city", "start_station_id"], dropna=False)[existing_station_cols]
        .median()
        .reset_index()
    )

    # ---------------------------------------------------------------
    # Step 4: City-hour metadata (weather + calendar, varies by hour)
    # ---------------------------------------------------------------
    weather_calendar_cols = [
        "temperature_2m", "relative_humidity_2m", "apparent_temperature",
        "precipitation", "rain", "snowfall", "cloud_cover", "wind_speed_10m",
        "holiday", "holiday_name", "working_day",
    ]
    existing_wc_cols = [c for c in weather_calendar_cols if c in df.columns]

    # Aggregation: numeric -> median, non-numeric -> first non-null
    wc_agg = {}
    for col in existing_wc_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            wc_agg[col] = "median"
        else:
            wc_agg[col] = first_non_null

    city_hour_meta = (
        df.groupby(["city", "hour_ts"], dropna=False)[existing_wc_cols]
        .agg(wc_agg)
        .reset_index()
    )

    # ---------------------------------------------------------------
    # Step 5: Build active-window grid with demand=0 rows
    # For each station: all hours from (first_ride - 24h) to (last_ride + 24h),
    # all 24 hours of the day included (0-23)
    # ---------------------------------------------------------------
    BUFFER_HOURS = 24

    station_windows = (
        df.groupby(["city", "start_station_id"], dropna=False)["hour_ts"]
        .agg(first_hour="min", last_hour="max")
        .reset_index()
    )

    grid_parts = []
    for _, row in station_windows.iterrows():
        start = row["first_hour"] - pd.Timedelta(hours=BUFFER_HOURS)
        end = row["last_hour"] + pd.Timedelta(hours=BUFFER_HOURS)
        hours = pd.date_range(start, end, freq="h")
        part = pd.DataFrame({
            "city": row["city"],
            "start_station_id": row["start_station_id"],
            "hour_ts": hours,
        })
        grid_parts.append(part)

    grid = pd.concat(grid_parts, ignore_index=True)
    print(f"  Grid rows (all station-hours in active windows + 24h buffer): {len(grid)}")

    # ---------------------------------------------------------------
    # Step 6: Attach demand (missing = 0)
    # ---------------------------------------------------------------
    train_df = grid.merge(demand, on=KEYS, how="left")
    train_df["demand"] = train_df["demand"].fillna(0).astype(int)

    print(f"  Rows with demand=0: {(train_df['demand']==0).sum()} ({(train_df['demand']==0).mean()*100:.1f}%)")
    print(f"  Rows with demand>0: {(train_df['demand']>0).sum()}")

    # ---------------------------------------------------------------
    # Step 7: Attach features
    # ---------------------------------------------------------------
    # Station metadata (static per station)
    train_df = train_df.merge(
        station_meta,
        on=["city", "start_station_id"],
        how="left",
    )

    # City-hour metadata (weather/calendar per city+hour)
    train_df = train_df.merge(
        city_hour_meta,
        on=["city", "hour_ts"],
        how="left",
    )

    # Derive time features from hour_ts
    train_df["hour"] = train_df["hour_ts"].dt.hour
    train_df["weekday"] = train_df["hour_ts"].dt.weekday
    train_df["date"] = train_df["hour_ts"].dt.date.astype(str)
    train_df["month"] = train_df["hour_ts"].dt.month
    train_df["weekend"] = train_df["weekday"].isin([5, 6]).astype(int)

    if "holiday_name" in train_df.columns:
        train_df["holiday_name"] = train_df["holiday_name"].fillna("no_holiday")

    return train_df


# ---------------------------------------------------------------------------
# Compute artifacts from training data
# ---------------------------------------------------------------------------

def compute_imputation_values(train_df):
    """Compute city median + global median for features with missing values."""
    values = {}
    for col in FEATURES_TO_IMPUTE:
        if col not in train_df.columns:
            continue
        col_data = pd.to_numeric(train_df[col], errors="coerce")
        values[col] = {
            "city_medians": {
                str(city): float(val)
                for city, val in train_df.groupby("city")[col].median().dropna().items()
            },
            "global_median": float(col_data.median()) if col_data.notna().any() else 0.0,
        }
    return values


def compute_column_medians(train_df):
    """
    Compute median of each numeric feature (within valid range only).
    Used to replace out-of-range values at predict time.
    """
    medians = {}
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in train_df.columns:
            continue
        s = pd.to_numeric(train_df[col], errors="coerce")
        # Only consider values within valid range for median calculation
        if lo is not None:
            s = s.where(s >= lo, np.nan)
        if hi is not None:
            s = s.where(s <= hi, np.nan)
        medians[col] = float(s.median()) if s.notna().any() else 0.0
    return medians


def compute_demand_averages(train_df):
    """
    Compute demand averages for target encoding:
        - city_hour_mean: average demand per (city, hour)
        - city_weekday_mean: average demand per (city, weekday)
        - global_hour_mean: average demand per hour (all cities)
        - global_weekday_mean: average demand per weekday (all cities)
    """
    hour_ts = pd.to_datetime(train_df["hour_ts"], errors="coerce")
    train_df = train_df.copy()
    train_df["hour"] = hour_ts.dt.hour
    train_df["weekday"] = hour_ts.dt.weekday

    city_hour_mean = (
        train_df.groupby(["city", "hour"])["demand"].mean().to_dict()
    )
    city_weekday_mean = (
        train_df.groupby(["city", "weekday"])["demand"].mean().to_dict()
    )
    global_hour_mean = (
        train_df.groupby("hour")["demand"].mean().to_dict()
    )
    global_weekday_mean = (
        train_df.groupby("weekday")["demand"].mean().to_dict()
    )

    return city_hour_mean, city_weekday_mean, global_hour_mean, global_weekday_mean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Step 1-2: Prepare station-hour training data ---
    print("Preparing training data...")
    train_df = prepare_training_data()
    print(f"  Rows: {len(train_df)}")
    print(f"  Cities: {train_df['city'].value_counts().to_dict()}")

    # --- Step 3: Compute all artifacts ---
    print("Computing artifacts...")

    # Imputation values (city median + global median for missing features)
    imputation_values = compute_imputation_values(train_df)

    # Column medians (for replacing out-of-range values)
    column_medians = compute_column_medians(train_df)

    # Demand averages (city-hour, city-weekday, global-hour, global-weekday)
    city_hour_mean, city_weekday_mean, global_hour_mean, global_weekday_mean = \
        compute_demand_averages(train_df)

    print(f"  Global mean demand: {train_df['demand'].mean():.4f}")
    print(f"  City-hour pairs: {len(city_hour_mean)}")
    print(f"  City-weekday pairs: {len(city_weekday_mean)}")

    # --- Step 4: Build features ---
    print("Building features...")
    bike_model = BikeDemandModel()
    bike_model.artifacts = {
        "model": None,
        "feature_columns": [],
        "city_hour_mean": city_hour_mean,
        "city_weekday_mean": city_weekday_mean,
        "global_hour_mean": global_hour_mean,
        "global_weekday_mean": global_weekday_mean,
        "imputation_values": imputation_values,
        "column_medians": column_medians,
    }

    X_train = bike_model.create_features(train_df)
    y_train = train_df["demand"].to_numpy()

    feature_columns = list(X_train.columns)
    print(f"  Features: {len(feature_columns)}")
    print(f"  Feature names: {feature_columns}")

    # --- Step 5: Train ---
    print("Training HistGradientBoostingRegressor...")
    model = HistGradientBoostingRegressor(
        max_iter=1000,        # converges at ~1000, more iterations don't help
        learning_rate=0.05,   # slow learning + many trees = better generalization
        max_depth=10,         # deep trees capture complex interactions
        max_leaf_nodes=127,   # fine-grained splits per tree
        min_samples_leaf=20,  # prevents overfitting to tiny groups
        random_state=42,
        verbose=0,
    )
    model.fit(X_train, y_train)

    # In-sample MAE
    train_preds = model.predict(X_train)
    train_mae = np.mean(np.abs(y_train - np.maximum(0, train_preds)))
    print(f"  In-sample MAE: {train_mae:.4f}")

    # --- Step 6: Save artifacts ---
    artifacts = {
        "model": model,
        "feature_columns": feature_columns,
        "city_hour_mean": city_hour_mean,
        "city_weekday_mean": city_weekday_mean,
        "global_hour_mean": global_hour_mean,
        "global_weekday_mean": global_weekday_mean,
        "imputation_values": imputation_values,
        "column_medians": column_medians,
    }

    joblib.dump(artifacts, OUTPUT_WEIGHTS)
    print(f"\nSaved {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()
