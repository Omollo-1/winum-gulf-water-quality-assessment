"""Small-sample, cross-validated calibration against collocated field readings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, LeaveOneOut


@dataclass(frozen=True)
class FieldCalibration:
    """Cross-validated field calibration metrics and out-of-fold predictions."""

    metrics: dict[str, float | int | str]
    predictions: pd.DataFrame


def evaluate_field_calibration(observations: pd.DataFrame) -> FieldCalibration:
    """Evaluate a univariate satellite proxy against matched field observations.

    Required columns: ``site_id``, ``field_variable``, ``observed_value``,
    ``field_unit``, ``satellite_method``, and ``satellite_proxy``. If the
    dataset covers at least three sites, each site is held out once; otherwise
    leave-one-observation-out cross-validation is used and flagged.
    """
    required = {
        "site_id",
        "field_variable",
        "observed_value",
        "field_unit",
        "satellite_method",
        "satellite_proxy",
    }
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Missing field validation columns: {sorted(missing)}")
    if observations.empty:
        raise ValueError("Field validation input must contain observations.")

    data = observations.copy()
    for column in ("site_id", "field_variable", "field_unit", "satellite_method"):
        data[column] = data[column].astype("string").str.strip()
        if data[column].isna().any() or data[column].eq("").any():
            raise ValueError(f"{column} must be present on every field observation.")
    if data["field_variable"].nunique() != 1 or data["field_unit"].nunique() != 1:
        raise ValueError("Evaluate one field variable and one measurement unit per calibration.")
    if data["satellite_method"].nunique() != 1:
        raise ValueError("Evaluate one satellite processing method per calibration.")

    data["observed_value"] = pd.to_numeric(data["observed_value"], errors="raise")
    data["satellite_proxy"] = pd.to_numeric(data["satellite_proxy"], errors="raise")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["observed_value", "satellite_proxy"]
    )
    if len(data) < 3:
        raise ValueError("At least three matched field/satellite observations are required.")

    x = data[["satellite_proxy"]].to_numpy(dtype=float)
    y = data["observed_value"].to_numpy(dtype=float)
    unique_sites = data["site_id"].nunique()
    site_held_out = unique_sites >= 3
    if site_held_out:
        splitter = LeaveOneGroupOut() if unique_sites == len(data) else GroupKFold(n_splits=unique_sites)
        splits = splitter.split(x, y, groups=data["site_id"].to_numpy())
        validation_method = "leave-one-site-out"
    else:
        splits = LeaveOneOut().split(x, y)
        validation_method = "leave-one-observation-out"

    predictions = np.full(len(data), np.nan, dtype=float)
    for train_indices, test_indices in splits:
        model = LinearRegression().fit(x[train_indices], y[train_indices])
        predictions[test_indices] = model.predict(x[test_indices])
    if not np.isfinite(predictions).all():
        raise RuntimeError("Cross-validation did not produce a prediction for every observation.")

    full_model = LinearRegression().fit(x, y)
    metrics: dict[str, float | int | str] = {
        "n_observations": int(len(data)),
        "n_sites": int(unique_sites),
        "field_variable": str(data["field_variable"].iloc[0]),
        "field_unit": str(data["field_unit"].iloc[0]),
        "satellite_method": str(data["satellite_method"].iloc[0]),
        "validation_method": validation_method,
        "site_held_out": site_held_out,
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "r2_out_of_fold": float(r2_score(y, predictions)),
        "calibration_intercept": float(full_model.intercept_),
        "calibration_slope_per_proxy_unit": float(full_model.coef_[0]),
    }
    prediction_frame = data.reset_index(drop=True).copy()
    prediction_frame["out_of_fold_prediction"] = predictions
    prediction_frame["out_of_fold_residual"] = y - predictions
    return FieldCalibration(metrics=metrics, predictions=prediction_frame)
