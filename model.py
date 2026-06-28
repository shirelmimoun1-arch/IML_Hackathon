"""
model.py — Bike Demand Prediction Model
========================================

This file contains the BikeDemandModel class used by predict.py.
It defines feature engineering and prediction logic.

Model: HistGradientBoostingRegressor (sklearn)
    - 1000 iterations, learning_rate=0.05, max_depth=10, max_leaf_nodes=127

Feature pipeline (create_features):
    1. Time features: hour (0-23), weekday (0-6), weekend (0/1), working_day (0/1)
    2. Holiday one-hot: one column per holiday name + "no_holiday"
    3. Demand averages (target encoding by city):
       - city_hour_mean: average demand for (city, hour)
       - city_weekday_mean: average demand for (city, weekday)
       - Fallback: global_hour_mean / global_weekday_mean for unknown cities
    4. Weather: temperature, humidity, precipitation, rain, snowfall, cloud_cover, wind
    5. POI / infrastructure: bike lanes, parks, universities, offices, retail,
       restaurants, transit stops, rail distance, city center distance
    6. Range validation: values outside valid range replaced with column median
    7. Missing value imputation: city median → global median fallback

Columns NOT used as features (dropped):
    - start_station_id (identifier — not generalizable to new cities)
    - city (used only for lookups, not as a direct feature)
    - start_lat, start_lng (location identifiers)

Valid ranges for sanity checking:
    - temperature_2m: -50 to 60
    - relative_humidity_2m: 0 to 100
    - apparent_temperature: -60 to 70
    - precipitation: 0 to 500
    - rain: 0 to 500
    - snowfall: 0 to 200
    - cloud_cover: 0 to 100
    - wind_speed_10m: 0 to 200
    - all distance/area/count features: >= 0
    - working_day: 0 or 1
"""

import numpy as np
import pandas as pd
from base_model import BaseModel


# ---------------------------------------------------------------------------
# Valid ranges — values outside these are replaced with column median
# ---------------------------------------------------------------------------

VALID_RANGES = {
    "temperature_2m": (-50, 60),
    "relative_humidity_2m": (0, 100),
    "apparent_temperature": (-60, 70),
    "precipitation": (0, 500),
    "rain": (0, 500),
    "snowfall": (0, 200),
    "cloud_cover": (0, 100),
    "wind_speed_10m": (0, 200),
    "bike_lane_length_500m": (0, None),
    "park_area_500m": (0, None),
    "university_count_1000m": (0, None),
    "office_poi_count_1000m": (0, None),
    "retail_poi_count_1000m": (0, None),
    "restaurant_cafe_count_500m": (0, None),
    "transit_stop_count_500m": (0, None),
    "distance_to_nearest_rail_station": (0, None),
    "distance_to_city_center": (0, None),
    "working_day": (0, 1),
}

# Features that have missing values and should be imputed with city/global median
FEATURES_TO_IMPUTE = [
    "distance_to_city_center",
    "bike_lane_length_500m",
    "park_area_500m",
    "office_poi_count_1000m",
    "retail_poi_count_1000m",
    "restaurant_cafe_count_500m",
    "transit_stop_count_500m",
    "university_count_1000m",
    "distance_to_nearest_rail_station",
]

# Columns removed from model input (kept in data for reference only)
COLUMNS_TO_DROP = ["start_station_id", "city", "start_lat", "start_lng"]


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class BikeDemandModel(BaseModel):
    """
    Bike demand forecasting model.

    Artifacts (stored in weights.joblib):
        - model: trained HistGradientBoostingRegressor
        - feature_columns: list of feature names in training order
        - city_hour_mean: dict {(city, hour): mean_demand}
        - city_weekday_mean: dict {(city, weekday): mean_demand}
        - global_hour_mean: dict {hour: mean_demand}
        - global_weekday_mean: dict {weekday: mean_demand}
        - imputation_values: dict {feature: {city_medians: {...}, global_median: float}}
        - column_medians: dict {feature: median} — for replacing out-of-range values
    """

    def __init__(self):
        self.artifacts = None

    def load(self, weights_path: str) -> None:
        """Load weights from disk. Required by BaseModel ABC."""
        import joblib
        artifacts = joblib.load(weights_path)
        self.load_artifacts(artifacts)

    def load_artifacts(self, artifacts: dict) -> None:
        """Store pre-computed artifacts from train.py."""
        self.artifacts = artifacts

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict bike demand for each row in test_df.

        Parameters
        ----------
        test_df : pd.DataFrame
            Station-hour test features. Does NOT contain 'demand'.

        Returns
        -------
        np.ndarray
            One non-negative prediction per row.
        """
        if self.artifacts is None:
            raise RuntimeError("Model is not loaded. Call load_artifacts() first.")

        # Build features
        X = self.create_features(test_df)

        # Align columns to training order
        feature_columns = self.artifacts["feature_columns"]
        X = X.reindex(columns=feature_columns, fill_value=0)

        # Predict and clip negatives
        model = self.artifacts["model"]
        preds = model.predict(X)
        return np.maximum(0.0, preds)

    # -------------------------------------------------------------------
    # Feature engineering
    # -------------------------------------------------------------------

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the feature matrix from station-hour data.

        Works identically on training data and test/validation targets.
        Uses hour_ts (train) or target_hour_start (test) as time source.

        Feature groups produced:
            - hour, weekday, weekend, working_day (time)
            - holiday_* columns (one-hot from holiday_name)
            - city_hour_mean, city_weekday_mean (demand averages with global fallback)
            - weather columns (8 features, range-validated)
            - POI/infrastructure columns (9 features, imputed + range-validated)

        Parameters
        ----------
        df : pd.DataFrame
            Station-hour data. Must have hour_ts or target_hour_start.

        Returns
        -------
        pd.DataFrame
            Feature matrix (no identifiers, no NaN).
        """
        out = pd.DataFrame(index=df.index)

        # ===== 1. TIME FEATURES =====
        # Determine timestamp from available columns
        time_col = "hour_ts" if "hour_ts" in df.columns else "target_hour_start"
        hour_ts = pd.to_datetime(df[time_col], errors="coerce")

        # Hour of day (0-23). NaN if timestamp missing.
        out["hour"] = hour_ts.dt.hour

        # Day of week (Monday=0 ... Sunday=6). NaN if timestamp missing.
        out["weekday"] = hour_ts.dt.weekday

        # Weekend flag. Derived from weekday.
        out["weekend"] = out["weekday"].isin([5, 6]).astype(float)
        out.loc[out["weekday"].isna(), "weekend"] = np.nan

        # Working day (from data, validated 0/1)
        out["working_day"] = pd.to_numeric(df.get("working_day"), errors="coerce")
        # Out of range → NaN (will be filled by range check later if needed)

        # Holiday one-hot encoding
        if "holiday_name" in df.columns:
            holiday_name = df["holiday_name"].fillna("no_holiday")
        else:
            holiday_name = pd.Series("no_holiday", index=df.index)
        holiday_dummies = pd.get_dummies(holiday_name, prefix="holiday", dtype=int)
        out = pd.concat([out, holiday_dummies], axis=1)

        # ===== 2. DEMAND AVERAGES (target encoding by city) =====
        if self.artifacts is not None:
            city_hour_mean = self.artifacts.get("city_hour_mean", {})
            city_weekday_mean = self.artifacts.get("city_weekday_mean", {})
            global_hour_mean = self.artifacts.get("global_hour_mean", {})
            global_weekday_mean = self.artifacts.get("global_weekday_mean", {})

            city = df["city"] if "city" in df.columns else pd.Series("unknown", index=df.index)
            hour = out["hour"]
            weekday = out["weekday"]

            # City-hour average demand (fallback: global-hour average)
            out["city_hour_mean"] = [
                city_hour_mean.get((c, h), global_hour_mean.get(h, np.nan))
                for c, h in zip(city, hour)
            ]

            # City-weekday average demand (fallback: global-weekday average)
            out["city_weekday_mean"] = [
                city_weekday_mean.get((c, w), global_weekday_mean.get(w, np.nan))
                for c, w in zip(city, weekday)
            ]

        # ===== 3. WEATHER FEATURES (with range validation) =====
        weather_cols = [
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "precipitation", "rain", "snowfall", "cloud_cover", "wind_speed_10m",
        ]
        for col in weather_cols:
            if col in df.columns:
                out[col] = self._clean_column(df[col], col)

        # ===== 4. POI / INFRASTRUCTURE (imputed + range validated) =====
        df_imputed = self._impute_missing(df)

        poi_cols = [
            "bike_lane_length_500m", "park_area_500m", "university_count_1000m",
            "office_poi_count_1000m", "retail_poi_count_1000m",
            "restaurant_cafe_count_500m", "transit_stop_count_500m",
            "distance_to_nearest_rail_station", "distance_to_city_center",
        ]
        for col in poi_cols:
            if col in df_imputed.columns:
                out[col] = self._clean_column(df_imputed[col], col)

        # ===== 5. WORKING_DAY — validate range =====
        out["working_day"] = self._clean_column(
            out["working_day"], "working_day"
        )

        # ===== 6. FILL REMAINING NaN =====
        # After all cleaning, remaining NaN filled with 0
        out = out.fillna(0)

        return out

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _clean_column(self, series: pd.Series, col_name: str) -> pd.Series:
        """
        Convert to numeric. Replace out-of-range values with column median.

        Valid ranges defined in VALID_RANGES dict.
        If no range defined for column, just convert to numeric.
        """
        s = pd.to_numeric(series, errors="coerce")

        if col_name in VALID_RANGES:
            lo, hi = VALID_RANGES[col_name]

            # Get median from artifacts (computed on train) or compute on the fly
            if self.artifacts and "column_medians" in self.artifacts:
                median_val = self.artifacts["column_medians"].get(col_name, 0.0)
            else:
                median_val = s.median() if s.notna().any() else 0.0

            # Mark out-of-range as NaN
            if lo is not None:
                s = s.where(s >= lo, np.nan)
            if hi is not None:
                s = s.where(s <= hi, np.nan)

            # Replace NaN (missing + out-of-range) with median
            s = s.fillna(median_val)

        return s

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in POI/location features.
        Strategy: city median → global median (from training artifacts).
        """
        out = df.copy()

        imputation_values = self.artifacts.get("imputation_values") if self.artifacts else None
        if imputation_values is None:
            for col in FEATURES_TO_IMPUTE:
                if col in out.columns:
                    out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
            return out

        city = out["city"].astype(str) if "city" in out.columns else pd.Series("unknown", index=out.index)

        for col in FEATURES_TO_IMPUTE:
            if col not in out.columns:
                continue
            out[col] = pd.to_numeric(out[col], errors="coerce")

            if col in imputation_values:
                city_medians = imputation_values[col]["city_medians"]
                global_median = imputation_values[col]["global_median"]
                fill_values = city.map(city_medians).fillna(global_median)
                out[col] = out[col].fillna(fill_values)

        return out
