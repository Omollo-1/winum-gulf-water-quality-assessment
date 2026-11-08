"""Temporal holdout skill test against a same-calendar-month climatology."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class ForecastBacktest:
    """Temporal holdout predictions and candidate-vs-climatology metrics."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame


def backtest_seasonal_forecast(
    observations: pd.DataFrame,
    *,
    target_column: str = "NDCI_chla",
    month_column: str = "month",
    train_end: str = "2022-12-01",
    holdout_start: str = "2023-01-01",
    holdout_end: str = "2024-12-01",
    minimum_training_months: int = 36,
    output_plot: str | None = None,
) -> ForecastBacktest:
    """Compare trend-plus-season regression with train-only monthly climatology.

    The default split reserves 2023–2024 for evaluation. Both methods use only
    observations before ``holdout_start`` for fitting. The comparison is an
    out-of-time skill test, not a claim that the candidate is production-ready.
    """
    required = {month_column, target_column}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Missing forecast backtest columns: {sorted(missing)}")
    if minimum_training_months < 24:
        raise ValueError("minimum_training_months must be at least 24.")

    data = observations.loc[:, [month_column, target_column]].copy()
    data[month_column] = pd.to_datetime(data[month_column], errors="raise").dt.to_period("M").dt.to_timestamp()
    data[target_column] = pd.to_numeric(data[target_column], errors="raise")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_column])
    if data[month_column].duplicated().any():
        raise ValueError("Input must contain at most one target observation per calendar month.")
    data = data.sort_values(month_column).set_index(month_column)

    train_cutoff = pd.Timestamp(train_end).to_period("M").to_timestamp()
    test_start = pd.Timestamp(holdout_start).to_period("M").to_timestamp()
    test_end = pd.Timestamp(holdout_end).to_period("M").to_timestamp()
    if test_start <= train_cutoff or test_end < test_start:
        raise ValueError("Holdout must start after train_end and end on or after holdout_start.")

    train = data.loc[data.index <= train_cutoff, target_column]
    test = data.loc[(data.index >= test_start) & (data.index <= test_end), target_column]
    if len(train) < minimum_training_months:
        raise ValueError(
            f"Only {len(train)} training months; at least {minimum_training_months} are required."
        )
    if len(test) < 12:
        raise ValueError(f"Only {len(test)} holdout observations; at least 12 are required.")

    climatology = train.groupby(train.index.month).median()
    missing_calendar_months = set(test.index.month) - set(climatology.index)
    if missing_calendar_months:
        raise ValueError(f"Training data lacks climatology for months: {sorted(missing_calendar_months)}")
    baseline = np.asarray([climatology.loc[month] for month in test.index.month], dtype=float)

    train_month_number = (
        (train.index.year - train.index[0].year) * 12 + train.index.month - train.index[0].month
    ).to_numpy(dtype=float)
    test_month_number = (
        (test.index.year - train.index[0].year) * 12 + test.index.month - train.index[0].month
    ).to_numpy(dtype=float)
    train_calendar = pd.get_dummies(train.index.month, prefix="month", dtype=float)
    test_calendar = pd.get_dummies(test.index.month, prefix="month", dtype=float).reindex(
        columns=train_calendar.columns, fill_value=0.0
    )
    train_design = np.column_stack([train_month_number, train_calendar.to_numpy()])
    test_design = np.column_stack([test_month_number, test_calendar.to_numpy()])
    candidate_model = LinearRegression().fit(train_design, train.to_numpy(dtype=float))
    candidate = candidate_model.predict(test_design)

    prediction_frame = pd.DataFrame(
        {
            "observed": test.to_numpy(dtype=float),
            "seasonal_climatology": baseline,
            "trend_season_candidate": candidate,
        },
        index=test.index,
    )
    prediction_frame.index.name = "month"
    metric_records = []
    for model_name in ("seasonal_climatology", "trend_season_candidate"):
        predicted = prediction_frame[model_name].to_numpy(dtype=float)
        actual = prediction_frame["observed"].to_numpy(dtype=float)
        metric_records.append(
            {
                "model": model_name,
                "holdout_start": test.index.min().strftime("%Y-%m"),
                "holdout_end": test.index.max().strftime("%Y-%m"),
                "n_holdout_months": int(len(test)),
                "mae": float(mean_absolute_error(actual, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                "bias_predicted_minus_observed": float(np.mean(predicted - actual)),
                "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else float("nan"),
            }
        )
    metrics = pd.DataFrame(metric_records)
    baseline_mae = float(metrics.loc[metrics["model"] == "seasonal_climatology", "mae"].iloc[0])
    candidate_mae = float(metrics.loc[metrics["model"] == "trend_season_candidate", "mae"].iloc[0])
    metrics["mae_improvement_vs_climatology"] = np.where(
        metrics["model"] == "trend_season_candidate", baseline_mae - candidate_mae, 0.0
    )
    metrics["beats_climatology_on_mae"] = np.where(
        metrics["model"] == "trend_season_candidate", candidate_mae < baseline_mae, False
    )

    if output_plot is not None:
        figure, axis = plt.subplots(figsize=(11, 5))
        prediction_frame["observed"].plot(ax=axis, color="black", marker="o", label="Observed")
        prediction_frame["seasonal_climatology"].plot(
            ax=axis, color="steelblue", linestyle="--", marker="s", label="Training climatology"
        )
        prediction_frame["trend_season_candidate"].plot(
            ax=axis, color="darkorange", linestyle=":", marker="^", label="Trend + season candidate"
        )
        axis.set_title("2023–2024 Temporal Holdout: Seasonal Baseline Skill")
        axis.set_ylabel(target_column)
        axis.set_xlabel("Month")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_plot, dpi=180, bbox_inches="tight")
        plt.close(figure)

    return ForecastBacktest(predictions=prediction_frame, metrics=metrics)
