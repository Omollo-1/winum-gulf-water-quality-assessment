"""Paired monthly comparison of atmospheric-correction NDCI time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class CorrectionComparison:
    """Paired monthly observations and correction-method summary metrics."""

    paired_months: pd.DataFrame
    summary: pd.DataFrame


def _trend_metrics(series: pd.Series) -> tuple[float, float]:
    valid = series.dropna()
    if len(valid) < 3 or valid.nunique() < 2:
        return float("nan"), float("nan")
    elapsed_months = (
        (valid.index - valid.index[0]).days.to_numpy(dtype=float) / (365.2425 / 12)
    )
    tau = float(stats.kendalltau(elapsed_months, valid.to_numpy()).statistic)
    slope = float(stats.theilslopes(valid.to_numpy(), elapsed_months).slope)
    return tau, slope


def compare_correction_methods(
    observations: pd.DataFrame,
    *,
    baseline_method: str = "gee_s2_sr",
) -> CorrectionComparison:
    """Compare methods on paired calendar months and compare their NDCI trends.

    Required columns are ``month``, ``method``, and ``ndci``. Each row is an
    already-computed monthly NDCI statistic from a consistently sampled scene
    subset. Duplicate method-month observations are reduced by their median.
    """
    required = {"month", "method", "ndci"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Missing correction comparison columns: {sorted(missing)}")
    if observations.empty:
        raise ValueError("Correction comparison input must contain observations.")

    data = observations.loc[:, ["month", "method", "ndci"]].copy()
    data["month"] = pd.to_datetime(data["month"], errors="raise").dt.to_period("M").dt.to_timestamp()
    data["method"] = data["method"].astype(str).str.strip()
    data["ndci"] = pd.to_numeric(data["ndci"], errors="raise")
    if data["method"].eq("").any() or not np.isfinite(data["ndci"]).all():
        raise ValueError("Correction methods must be named and NDCI values must be finite.")
    if baseline_method not in set(data["method"]):
        raise ValueError(f"Baseline method {baseline_method!r} is absent from the input.")

    monthly = (
        data.groupby(["month", "method"], as_index=False)["ndci"]
        .median()
        .pivot(index="month", columns="method", values="ndci")
        .sort_index()
    )
    baseline = monthly[baseline_method]
    paired_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []

    for method in sorted(column for column in monthly.columns if column != baseline_method):
        paired = monthly[[baseline_method, method]].dropna()
        if paired.empty:
            continue
        baseline_values = paired[baseline_method].to_numpy(dtype=float)
        method_values = paired[method].to_numpy(dtype=float)
        differences = method_values - baseline_values
        correlation = (
            float(stats.pearsonr(baseline_values, method_values).statistic)
            if len(paired) >= 2 and np.ptp(baseline_values) > 0 and np.ptp(method_values) > 0
            else float("nan")
        )
        baseline_tau, baseline_slope = _trend_metrics(paired[baseline_method])
        method_tau, method_slope = _trend_metrics(paired[method])

        summary_records.append(
            {
                "baseline_method": baseline_method,
                "method": method,
                "paired_months": int(len(paired)),
                "mean_bias_method_minus_baseline": float(np.mean(differences)),
                "mae": float(np.mean(np.abs(differences))),
                "rmse": float(np.sqrt(np.mean(np.square(differences)))),
                "pearson_r": correlation,
                "baseline_kendall_tau": baseline_tau,
                "method_kendall_tau": method_tau,
                "baseline_sen_slope_ndci_per_month": baseline_slope,
                "method_sen_slope_ndci_per_month": method_slope,
                "sen_slope_difference_ndci_per_month": method_slope - baseline_slope,
            }
        )
        for month, baseline_value, method_value in zip(
            paired.index, baseline_values, method_values, strict=True
        ):
            paired_records.append(
                {
                    "month": month,
                    "baseline_method": baseline_method,
                    "method": method,
                    "baseline_ndci": baseline_value,
                    "method_ndci": method_value,
                    "method_minus_baseline": method_value - baseline_value,
                }
            )

    if not summary_records:
        raise ValueError("No correction method has paired months with the baseline.")
    return CorrectionComparison(
        paired_months=pd.DataFrame(paired_records),
        summary=pd.DataFrame(summary_records),
    )
