"""Time-ordered Random Forest analysis for matched environmental drivers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


DRIVER_COLUMNS = (
    "rainfall_mm",
    "wind_speed_ms",
    "nyando_inflow_m3s",
    "sondu_miriu_inflow_m3s",
)


@dataclass(frozen=True)
class DriverAttribution:
    """Time-held-out performance, permutation importance, and predictions."""

    metrics: dict[str, object]
    permutation_importances: pd.DataFrame
    predictions: pd.DataFrame
    model: Pipeline


def _monthly_climatology(values: pd.Series, months: pd.Index) -> np.ndarray:
    climatology = values.groupby(values.index.month).median()
    if set(months.month) - set(climatology.index):
        raise ValueError("Training period does not cover every test calendar month.")
    return np.asarray([climatology.loc[month] for month in months.month], dtype=float)


def train_driver_attribution(
    observations: pd.DataFrame,
    *,
    target_column: str = "bloom_metric",
    month_column: str = "month",
    driver_columns: tuple[str, ...] = DRIVER_COLUMNS,
    train_fraction: float = 0.75,
    random_state: int = 42,
    permutation_repeats: int = 30,
    calculate_shap: bool = False,
) -> DriverAttribution:
    """Fit a chronological holdout Random Forest and test-set permutation importance.

    The reported importances are predictive associations, not causal effects.
    Set ``calculate_shap=True`` only when the optional ``shap`` package is
    installed. Same-month and prior-month driver measurements are both
    represented; the intended analysis is retrospective attribution.
    """
    if not 0.5 <= train_fraction < 0.9:
        raise ValueError("train_fraction must be between 0.5 and 0.9.")
    if permutation_repeats < 1:
        raise ValueError("permutation_repeats must be positive.")
    required = {month_column, target_column, *driver_columns}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Missing driver attribution columns: {sorted(missing)}")

    data = observations.loc[:, [month_column, target_column, *driver_columns]].copy()
    data[month_column] = pd.to_datetime(data[month_column], errors="raise").dt.to_period("M").dt.to_timestamp()
    for column in (target_column, *driver_columns):
        data[column] = pd.to_numeric(data[column], errors="raise")
    data = data.replace([np.inf, -np.inf], np.nan)
    if data[month_column].duplicated().any():
        raise ValueError("Driver input must contain at most one row per month.")
    data = data.sort_values(month_column).set_index(month_column)
    data = data.dropna(subset=[target_column])
    minimum_rows = max(36, len(driver_columns) * 8)
    if len(data) < minimum_rows:
        raise ValueError(
            f"Only {len(data)} target months; at least {minimum_rows} are required for a time-ordered analysis."
        )

    features = pd.DataFrame(index=data.index)
    for column in driver_columns:
        for lag in (0, 1, 2):
            features[f"{column}_lag{lag}"] = data[column].shift(lag)
    aligned = pd.concat([data[[target_column]], features], axis=1).dropna(
        subset=list(features.columns), how="all"
    )
    split_at = int(len(aligned) * train_fraction)
    if split_at < 24 or len(aligned) - split_at < 12:
        raise ValueError("Need at least 24 training and 12 held-out monthly observations.")
    train = aligned.iloc[:split_at]
    test = aligned.iloc[split_at:]
    x_train = train[features.columns]
    x_test = test[features.columns]
    all_missing_features = x_train.columns[x_train.isna().all()].tolist()
    if all_missing_features:
        raise ValueError(f"Training period has no values for features: {all_missing_features}")
    y_train = train[target_column].to_numpy(dtype=float)
    y_test = test[target_column].to_numpy(dtype=float)

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=250,
                    min_samples_leaf=2,
                    max_features=0.8,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    baseline = _monthly_climatology(train[target_column], test.index)
    metrics: dict[str, object] = {
        "n_training_months": int(len(train)),
        "n_holdout_months": int(len(test)),
        "train_start": train.index.min().strftime("%Y-%m"),
        "train_end": train.index.max().strftime("%Y-%m"),
        "holdout_start": test.index.min().strftime("%Y-%m"),
        "holdout_end": test.index.max().strftime("%Y-%m"),
        "random_forest_mae": float(mean_absolute_error(y_test, predicted)),
        "random_forest_rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "random_forest_r2": float(r2_score(y_test, predicted)),
        "climatology_mae": float(mean_absolute_error(y_test, baseline)),
        "climatology_rmse": float(np.sqrt(mean_squared_error(y_test, baseline))),
        "predictors_are_causal": False,
    }

    importance = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=permutation_repeats,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
    )
    importance_frame = pd.DataFrame(
        {
            "feature": x_test.columns,
            "permutation_importance_mean": importance.importances_mean,
            "permutation_importance_std": importance.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False, ignore_index=True)

    if calculate_shap:
        try:
            import shap
        except ImportError as error:
            raise ImportError(
                "SHAP output requested; install the optional dependency with "
                "`python -m pip install shap`."
            ) from error
        matrix = model.named_steps["imputer"].transform(x_test)
        explainer = shap.TreeExplainer(model.named_steps["random_forest"])
        shap_values = explainer.shap_values(matrix)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_importance = np.mean(np.abs(np.asarray(shap_values)), axis=0)
        feature_names = list(x_test.columns)
        indicator = model.named_steps["imputer"].indicator_
        indicator_features = indicator.features_ if indicator is not None else []
        feature_names.extend(
            f"missing:{x_test.columns[index]}" for index in indicator_features
        )
        if len(feature_names) != len(shap_importance):
            raise RuntimeError("SHAP output dimensions do not match imputed model features.")
        metrics["shap_available"] = True
        metrics["mean_absolute_shap_values"] = dict(
            zip(feature_names, map(float, shap_importance), strict=True)
        )
    else:
        metrics["shap_available"] = False

    prediction_frame = pd.DataFrame(
        {
            "observed": y_test,
            "random_forest_prediction": predicted,
            "seasonal_climatology": baseline,
        },
        index=test.index,
    )
    prediction_frame.index.name = month_column
    return DriverAttribution(
        metrics=metrics,
        permutation_importances=importance_frame,
        predictions=prediction_frame,
        model=model,
    )
