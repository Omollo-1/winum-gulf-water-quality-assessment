"""Robustness analyses for monthly and event-scale water-quality observations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


DEFAULT_BLOOM_THRESHOLDS = tuple(np.round(np.arange(0.10, 0.301, 0.05), 2))
EXPECTED_GULF_ZONES = (
    "Kisumu Bay",
    "Inner Gulf",
    "Homa Bay",
    "Kendu Bay",
    "Outer Gulf",
)
REQUIRED_HYACINTH_CLASSES = {"hyacinth", "open_water", "turbid_water"}


def calibrate_hyacinth_thresholds(
    observations: pd.DataFrame,
    *,
    polygon_column: str = "polygon_id",
    label_column: str = "class_label",
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, object]:
    """Fit NDVI/FAI candidate thresholds on polygon-separated labeled data.

    Thresholds are selected on training polygons by balanced accuracy. Holdout
    polygons are used only for the reported evaluation. Polygon-level medians
    prevent multiple pixels from one hand-labeled polygon leaking across splits.
    """
    required = {polygon_column, label_column, "ndvi", "fai"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Missing hyacinth calibration columns: {sorted(missing)}")
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5.")

    data = observations.loc[:, [polygon_column, label_column, "ndvi", "fai"]].copy()
    data[polygon_column] = data[polygon_column].astype("string").str.strip()
    data[label_column] = data[label_column].astype("string").str.strip().str.lower()
    if (
        data[polygon_column].isna().any()
        or data[polygon_column].eq("").any()
        or data[label_column].isna().any()
        or data[label_column].eq("").any()
    ):
        raise ValueError("Polygon IDs and class labels must be present on every sample.")
    data["ndvi"] = pd.to_numeric(data["ndvi"], errors="raise")
    data["fai"] = pd.to_numeric(data["fai"], errors="raise")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["ndvi", "fai"])
    if data["ndvi"].lt(-1).any() or data["ndvi"].gt(1).any():
        raise ValueError("NDVI values must be between -1 and 1.")
    polygon_labels = data.groupby(polygon_column)[label_column].nunique()
    if polygon_labels.gt(1).any():
        raise ValueError("Each polygon_id must have exactly one class label.")
    polygons = data.groupby(polygon_column, as_index=False).agg(
        class_label=(label_column, "first"),
        ndvi=("ndvi", "median"),
        fai=("fai", "median"),
    )
    labels = set(polygons["class_label"])
    if not REQUIRED_HYACINTH_CLASSES.issubset(labels):
        raise ValueError(
            "Calibration requires hyacinth, open_water, and turbid_water polygon labels."
        )
    per_class = polygons["class_label"].value_counts()
    if per_class.lt(2).any():
        raise ValueError("At least two labeled polygons per class are needed for a holdout split.")

    train_ids, test_ids = train_test_split(
        polygons["polygon_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=polygons["class_label"],
    )
    training = polygons.set_index("polygon_id").loc[train_ids]
    testing = polygons.set_index("polygon_id").loc[test_ids]
    training_labels = training["class_label"].eq("hyacinth").to_numpy()
    testing_labels = testing["class_label"].eq("hyacinth").to_numpy()
    ndvi_candidates = np.unique(
        np.quantile(training["ndvi"], np.linspace(0.05, 0.95, 19))
    )
    fai_candidates = np.unique(
        np.quantile(training["fai"], np.linspace(0.05, 0.95, 19))
    )
    best: tuple[float, float, float, float, float] | None = None
    for ndvi_threshold in ndvi_candidates:
        for fai_threshold in fai_candidates:
            predicted = (
                training["ndvi"].ge(ndvi_threshold)
                & training["fai"].ge(fai_threshold)
            ).to_numpy()
            score = (
                float(balanced_accuracy_score(training_labels, predicted)),
                float(f1_score(training_labels, predicted, zero_division=0)),
                float(precision_score(training_labels, predicted, zero_division=0)),
                float(ndvi_threshold),
                float(fai_threshold),
            )
            if best is None or score > best:
                best = score
    if best is None:
        raise RuntimeError("No NDVI/FAI threshold candidates were evaluated.")

    ndvi_threshold, fai_threshold = best[3], best[4]
    predicted = (
        testing["ndvi"].ge(ndvi_threshold) & testing["fai"].ge(fai_threshold)
    ).to_numpy()
    tn, fp, fn, tp = confusion_matrix(
        testing_labels, predicted, labels=[False, True]
    ).ravel()
    return {
        "ndvi_threshold": ndvi_threshold,
        "fai_threshold": fai_threshold,
        "positive_class": "hyacinth",
        "negative_classes": sorted(labels - {"hyacinth"}),
        "threshold_selection": "training polygon balanced accuracy",
        "n_polygons": int(len(polygons)),
        "recommended_label_count": "50-100 polygons",
        "meets_recommended_label_count": bool(len(polygons) >= 50),
        "n_training_polygons": int(len(training)),
        "n_holdout_polygons": int(len(testing)),
        "training_class_counts": {
            str(label): int(count) for label, count in training["class_label"].value_counts().items()
        },
        "holdout_class_counts": {
            str(label): int(count) for label, count in testing["class_label"].value_counts().items()
        },
        "holdout_metrics": {
            "balanced_accuracy": float(balanced_accuracy_score(testing_labels, predicted)),
            "precision": float(precision_score(testing_labels, predicted, zero_division=0)),
            "recall": float(recall_score(testing_labels, predicted, zero_division=0)),
            "f1": float(f1_score(testing_labels, predicted, zero_division=0)),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "interpretation": "Candidate floating-vegetation mask; validate on independent dates or locations before use.",
    }


def hamed_rao_mann_kendall(
    observations: pd.Series, *, alpha: float = 0.05
) -> dict[str, float | int | bool | str]:
    """Run Hamed-Rao variance-corrected Mann-Kendall on a complete monthly series.

    Missing months are rejected rather than interpolated or treated as adjacent
    observations. The serial-correlation correction is estimated from ranks after
    removing the Sen trend.
    """
    if not isinstance(observations.index, pd.DatetimeIndex):
        raise TypeError("Mann-Kendall input must have a DatetimeIndex.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    series = observations.sort_index().astype(float)
    if series.empty:
        raise ValueError("Mann-Kendall input must not be empty.")
    if series.index.has_duplicates:
        raise ValueError("Mann-Kendall input must contain at most one value per date.")
    if series.isna().any() or not np.isfinite(series.to_numpy()).all():
        raise ValueError("Mann-Kendall input must contain finite values without gaps.")
    monthly_index = series.index.to_period("M")
    if monthly_index.has_duplicates:
        raise ValueError("Mann-Kendall input must contain at most one value per month.")
    expected_months = pd.period_range(monthly_index.min(), monthly_index.max(), freq="M")
    if not monthly_index.equals(expected_months):
        raise ValueError("Hamed-Rao requires a complete monthly series; do not interpolate missing months.")

    deseasonalized = series - series.groupby(series.index.month).transform("median")
    values = deseasonalized.to_numpy(dtype=float)
    n = len(values)
    if n < 24:
        raise ValueError("At least 24 complete monthly observations are required.")

    difference = values[np.newaxis, :] - values[:, np.newaxis]
    upper_triangle = difference[np.triu_indices(n, k=1)]
    score = float(np.sign(upper_triangle).sum())
    _, tie_counts = np.unique(values, return_counts=True)
    tie_term = float(np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5)))
    variance = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if variance <= 0:
        return {
            "trend": "no significant trend",
            "kendall_tau": 0.0,
            "sen_slope_per_month": 0.0,
            "sen_slope_ndci_per_month": 0.0,
            "hamed_rao_z": 0.0,
            "p_value": 1.0,
            "alpha": alpha,
            "significant": False,
            "n_observations": n,
            "variance_correction_factor": 1.0,
            "serial_correlation_adjusted": True,
            "seasonally_deseasonalized": True,
        }

    elapsed_months = np.arange(n, dtype=float)
    sen_slope = float(stats.theilslopes(values, elapsed_months).slope)
    detrended_ranks = stats.rankdata(values - sen_slope * elapsed_months)
    critical_rho = float(stats.norm.ppf(1 - alpha / 2) / np.sqrt(n))
    correction_sum = 0.0
    for lag in range(1, n - 2):
        left = detrended_ranks[:-lag]
        right = detrended_ranks[lag:]
        rho = float(stats.pearsonr(left, right).statistic)
        if np.isfinite(rho) and abs(rho) > critical_rho:
            correction_sum += (
                (n - lag)
                * (n - lag - 1)
                * (n - lag - 2)
                * rho
            )
    variance_factor = 1 + 2 * correction_sum / (n * (n - 1) * (n - 2))
    if variance_factor <= 0:
        raise ValueError("Hamed-Rao adjusted variance is not positive.")
    adjusted_variance = variance * variance_factor

    if score > 0:
        z_score = (score - 1) / np.sqrt(adjusted_variance)
    elif score < 0:
        z_score = (score + 1) / np.sqrt(adjusted_variance)
    else:
        z_score = 0.0
    p_value = float(2 * stats.norm.sf(abs(z_score)))
    tau = float(score / (n * (n - 1) / 2))
    if p_value < alpha:
        trend = "increasing" if score > 0 else "decreasing"
    else:
        trend = "no significant trend"

    return {
        "trend": trend,
        "kendall_tau": tau,
        "sen_slope_per_month": sen_slope,
        "sen_slope_ndci_per_month": sen_slope,
        "hamed_rao_z": float(z_score),
        "p_value": p_value,
        "alpha": alpha,
        "significant": bool(p_value < alpha),
        "n_observations": n,
        "variance_correction_factor": float(variance_factor),
        "serial_correlation_adjusted": True,
        "seasonally_deseasonalized": True,
    }


def bloom_threshold_sensitivity(
    observations: pd.DataFrame,
    *,
    hotspot_frequency: float,
    thresholds: Sequence[float] = DEFAULT_BLOOM_THRESHOLDS,
    reference_threshold: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare per-pixel exceedance frequency and hotspot overlap across NDCI cutoffs.

    Input is long-form, one row per ``pixel_id`` and ``month``. The stability
    score is the Jaccard overlap of hotspots against the reference cutoff.
    """
    required = {"pixel_id", "month", "ndci"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Missing bloom sensitivity columns: {sorted(missing)}")
    if not 0 <= hotspot_frequency <= 1:
        raise ValueError("hotspot_frequency must be between 0 and 1.")
    cutoffs = np.asarray(thresholds, dtype=float)
    if cutoffs.size == 0 or not np.isfinite(cutoffs).all() or np.any(np.diff(cutoffs) <= 0):
        raise ValueError("thresholds must be finite, non-empty, and strictly increasing.")
    if reference_threshold not in cutoffs:
        raise ValueError("thresholds must include the reference_threshold.")

    data = observations.loc[:, ["pixel_id", "month", "ndci"]].copy()
    data["pixel_id"] = data["pixel_id"].astype("string").str.strip()
    if data["pixel_id"].isna().any() or data["pixel_id"].eq("").any():
        raise ValueError("pixel_id must be present on every row.")
    data["month"] = pd.to_datetime(data["month"], errors="raise").dt.to_period("M")
    data["ndci"] = pd.to_numeric(data["ndci"], errors="raise")
    if data.duplicated(["pixel_id", "month"]).any():
        raise ValueError("Input must contain at most one NDCI value per pixel and month.")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["ndci"])
    if data.empty:
        raise ValueError("At least one finite NDCI observation is required.")

    records: list[dict[str, object]] = []
    valid_counts = data.groupby("pixel_id").size()
    for cutoff in cutoffs:
        frequencies = data.assign(exceeds=data["ndci"] >= cutoff).groupby("pixel_id")[
            "exceeds"
        ].mean()
        records.extend(
            {
                "pixel_id": pixel_id,
                "ndci_threshold": float(cutoff),
                "valid_months": int(valid_counts.loc[pixel_id]),
                "bloom_frequency": float(frequency),
            }
            for pixel_id, frequency in frequencies.items()
        )
    frequency_table = pd.DataFrame.from_records(records)
    pivoted = frequency_table.pivot(
        index="pixel_id", columns="ndci_threshold", values="bloom_frequency"
    )
    reference_hotspots = pivoted[reference_threshold] >= hotspot_frequency
    stability_records = []
    for cutoff in cutoffs:
        hotspots = pivoted[float(cutoff)] >= hotspot_frequency
        union = int((reference_hotspots | hotspots).sum())
        intersection = int((reference_hotspots & hotspots).sum())
        stability_records.append(
            {
                "ndci_threshold": float(cutoff),
                "hotspot_frequency_cutoff": hotspot_frequency,
                "reference_threshold": reference_threshold,
                "reference_hotspots": int(reference_hotspots.sum()),
                "hotspots": int(hotspots.sum()),
                "jaccard_vs_reference": float(intersection / union) if union else 1.0,
            }
        )
    return frequency_table, pd.DataFrame.from_records(stability_records)


def summarize_gulf_zones(
    observations: pd.DataFrame, *, shoreline_buffer_m: float = 80.0
) -> pd.DataFrame:
    """Summarize monthly observations by supplied gulf zone, excluding shoreline pixels.

    Zone labels and shoreline distances must be produced from documented,
    authoritative polygons and a projected coordinate system before this step.
    """
    required = {"month", "zone", "pixel_id", "distance_to_shore_m", "ndci"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Missing zonal summary columns: {sorted(missing)}")
    if not np.isfinite(shoreline_buffer_m) or shoreline_buffer_m < 0:
        raise ValueError("shoreline_buffer_m must be finite and non-negative.")
    data = observations.copy()
    data["month"] = pd.to_datetime(data["month"], errors="raise").dt.to_period("M").dt.to_timestamp()
    data["zone"] = data["zone"].astype("string").str.strip()
    data["pixel_id"] = data["pixel_id"].astype("string").str.strip()
    data["distance_to_shore_m"] = pd.to_numeric(data["distance_to_shore_m"], errors="raise")
    data["ndci"] = pd.to_numeric(data["ndci"], errors="raise")
    if data["zone"].isna().any() or data["zone"].eq("").any():
        raise ValueError("zone must be present on every observation.")
    if data["pixel_id"].isna().any() or data["pixel_id"].eq("").any():
        raise ValueError("pixel_id must be present on every observation.")
    if data.duplicated(["pixel_id", "month"]).any():
        raise ValueError("Each pixel must have at most one zone and NDCI value per month.")
    observed_zones = set(data["zone"].unique())
    if observed_zones != set(EXPECTED_GULF_ZONES):
        raise ValueError(
            "Zone labels must contain exactly: " + ", ".join(EXPECTED_GULF_ZONES)
        )
    if not np.isfinite(data["distance_to_shore_m"]).all():
        raise ValueError("distance_to_shore_m must contain finite distances in metres.")
    if data["distance_to_shore_m"].lt(0).any():
        raise ValueError("distance_to_shore_m cannot be negative.")

    offshore = data.loc[data["distance_to_shore_m"] >= shoreline_buffer_m].copy()
    if offshore.empty:
        raise ValueError("No observations remain outside the shoreline buffer.")
    missing_zones = set(EXPECTED_GULF_ZONES) - set(offshore["zone"].unique())
    if missing_zones:
        raise ValueError(
            f"No observations remain outside the shoreline buffer for: {sorted(missing_zones)}"
        )
    metrics = [column for column in ("ndci", "tss_proxy", "cdom_proxy") if column in offshore]
    for column in metrics:
        offshore[column] = pd.to_numeric(offshore[column], errors="raise").replace(
            [np.inf, -np.inf], np.nan
        )
    summary = (
        offshore.groupby(["zone", "month"], as_index=False, observed=True)
        .agg(
            **{
                **{f"{column}_median": (column, "median") for column in metrics},
                "pixel_count": ("ndci", "count"),
                "shoreline_buffer_m": ("distance_to_shore_m", lambda _: shoreline_buffer_m),
            }
        )
        .sort_values(["zone", "month"], ignore_index=True)
    )
    return summary


def event_rainfall_lags(
    daily_observations: pd.DataFrame,
    *,
    target_columns: Sequence[str] = ("ndci", "tss_proxy"),
    lag_days: Sequence[int] = (7, 14, 21, 28),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match daily rainfall totals preceding satellite overpasses to proxy values.

    The input has one row per day; target values are populated only on
    satellite-overpass days. Antecedent windows exclude the overpass day.
    """
    required = {"date", "rainfall_mm", *target_columns}
    missing = required - set(daily_observations.columns)
    if missing:
        raise ValueError(f"Missing event-rainfall columns: {sorted(missing)}")
    lags = tuple(int(lag) for lag in lag_days)
    if not target_columns:
        raise ValueError("target_columns must not be empty.")
    if not lags or any(lag < 1 for lag in lags) or len(set(lags)) != len(lags):
        raise ValueError("lag_days must contain unique positive integers.")
    data = daily_observations.loc[:, ["date", "rainfall_mm", *target_columns]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    if data.empty or data["date"].isna().any():
        raise ValueError("Daily rainfall input must contain valid dates.")
    if data["date"].duplicated().any():
        raise ValueError("Daily rainfall input must contain at most one row per date.")
    data["rainfall_mm"] = pd.to_numeric(data["rainfall_mm"], errors="raise")
    for target in target_columns:
        data[target] = pd.to_numeric(data[target], errors="raise")
    if not np.isfinite(data["rainfall_mm"].dropna()).all():
        raise ValueError("rainfall_mm must contain only finite values or missing values.")
    data[list(target_columns)] = data[list(target_columns)].replace([np.inf, -np.inf], np.nan)
    if data["rainfall_mm"].dropna().lt(0).any():
        raise ValueError("rainfall_mm cannot be negative.")
    data = data.sort_values("date").set_index("date")
    complete_dates = pd.date_range(data.index.min(), data.index.max(), freq="D")
    daily_rain = data["rainfall_mm"].reindex(complete_dates)
    antecedent: dict[int, pd.Series] = {}
    for days in lags:
        antecedent[days] = daily_rain.shift(1).rolling(days, min_periods=days).sum()

    sample_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for target in target_columns:
        target_series = data[target].dropna()
        for days, rain_series in antecedent.items():
            matched = pd.concat(
                [target_series.rename("target"), rain_series.rename("rainfall_mm")],
                axis=1,
            ).dropna()
            if (
                len(matched) < 3
                or matched["rainfall_mm"].nunique() < 2
                or matched["target"].nunique() < 2
            ):
                correlation = float("nan")
                spearman = float("nan")
            else:
                correlation = float(stats.pearsonr(matched["rainfall_mm"], matched["target"]).statistic)
                spearman = float(stats.spearmanr(matched["rainfall_mm"], matched["target"]).statistic)
            summary_records.append(
                {
                    "target": target,
                    "lag_days": days,
                    "n_matchups": int(len(matched)),
                    "pearson_r": correlation,
                    "spearman_rho": spearman,
                    "causal_interpretation": "not established",
                }
            )
            sample_records.extend(
                {
                    "date": date,
                    "target": target,
                    "target_value": float(row["target"]),
                    "lag_days": days,
                    "antecedent_rainfall_mm": float(row["rainfall_mm"]),
                }
                for date, row in matched.iterrows()
            )
    return pd.DataFrame.from_records(sample_records), pd.DataFrame.from_records(summary_records)
