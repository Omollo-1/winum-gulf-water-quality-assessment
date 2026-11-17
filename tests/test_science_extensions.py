import numpy as np
import pandas as pd
import pytest

from winam_gulf.corrections import compare_correction_methods
from winam_gulf.drivers import train_driver_attribution
from winam_gulf.field_validation import evaluate_field_calibration
from winam_gulf.flowpath import analyze_flowpath_distance
from winam_gulf.forecast import backtest_seasonal_forecast
from winam_gulf.robustness import (
    bloom_threshold_sensitivity,
    calibrate_hyacinth_thresholds,
    event_rainfall_lags,
    hamed_rao_mann_kendall,
    summarize_gulf_zones,
)
from winam_gulf.spectral import (
    compute_fai,
    compute_ndvi,
    compute_olci_indices,
    compute_three_band_line_height,
    hyacinth_candidate_mask,
)


def test_spectral_indices_and_explicit_mask_thresholds():
    np.testing.assert_allclose(compute_ndvi([0.6], [0.2]), [0.5])
    np.testing.assert_allclose(
        compute_fai([0.1], [0.3], [0.2]),
        [0.3 - (0.1 + (0.2 - 0.1) * ((842 - 665) / (1610 - 665)))],
    )
    assert np.isnan(compute_ndvi([0.0], [0.0])[0])
    line_height = compute_three_band_line_height(
        [0.1], [0.2], [0.3], wavelengths_nm=(681, 709, 754)
    )
    assert line_height[0] > 0
    olci = compute_olci_indices([0.1], [0.2], [0.3])
    assert set(olci) == {"ndci_709_681", "line_height_709"}
    mask = hyacinth_candidate_mask(
        [0.7, 0.2, np.nan],
        [0.03, 0.05, 0.2],
        ndvi_threshold=0.5,
        fai_threshold=0.02,
    )
    np.testing.assert_array_equal(mask, [True, False, False])
    with pytest.raises(ValueError, match="thresholds must be finite"):
        hyacinth_candidate_mask([0.6], [0.1], ndvi_threshold=np.nan, fai_threshold=0.0)


def test_hyacinth_threshold_calibration_holds_out_whole_polygons():
    rows = []
    for label, ndvi, fai in (
        ("hyacinth", 0.75, 0.12),
        ("open_water", 0.10, -0.02),
        ("turbid_water", 0.25, 0.01),
    ):
        for index in range(20):
            polygon_id = f"{label}-{index}"
            rows.extend(
                [
                    {"polygon_id": polygon_id, "class_label": label, "ndvi": ndvi, "fai": fai},
                    {
                        "polygon_id": polygon_id,
                        "class_label": label,
                        "ndvi": ndvi + 0.01,
                        "fai": fai + 0.001,
                    },
                ]
            )
    result = calibrate_hyacinth_thresholds(pd.DataFrame(rows))
    assert result["n_polygons"] == 60
    assert result["n_training_polygons"] + result["n_holdout_polygons"] == 60
    assert result["meets_recommended_label_count"] is True
    assert result["holdout_metrics"]["balanced_accuracy"] == 1.0
    assert np.isfinite(result["ndvi_threshold"])
    assert np.isfinite(result["fai_threshold"])


def test_correction_comparison_uses_paired_months_and_trend_slopes():
    months = pd.date_range("2020-01-01", periods=30, freq="MS")
    base = np.linspace(0.0, 0.3, len(months))
    frame = pd.DataFrame(
        {
            "month": list(months) * 2,
            "method": ["gee_s2_sr"] * len(months) + ["acolite"] * len(months),
            "ndci": list(base) + list(base * 1.1 + 0.02),
        }
    )
    result = compare_correction_methods(frame)
    assert result.summary.loc[0, "paired_months"] == 30
    assert result.summary.loc[0, "mae"] > 0
    assert result.summary.loc[0, "baseline_sen_slope_ndci_per_month"] > 0
    assert result.summary.loc[0, "method_sen_slope_ndci_per_month"] > result.summary.loc[
        0, "baseline_sen_slope_ndci_per_month"
    ]


def test_field_calibration_holds_out_sites():
    frame = pd.DataFrame(
        {
            "site_id": np.repeat(["A", "B", "C"], 4),
            "field_variable": "turbidity",
            "observed_value": np.arange(12, dtype=float) * 2 + 1,
            "field_unit": "NTU",
            "satellite_method": "acolite",
            "satellite_proxy": np.arange(12, dtype=float) * 0.5,
        }
    )
    result = evaluate_field_calibration(frame)
    assert result.metrics["validation_method"] == "leave-one-site-out"
    assert result.metrics["n_sites"] == 3
    assert np.isfinite(result.predictions["out_of_fold_prediction"]).all()


def test_forecast_backtest_uses_pre_holdout_training():
    months = pd.date_range("2018-01-01", "2024-12-01", freq="MS")
    values = np.asarray([0.1 + 0.002 * i + 0.03 * np.sin(month.month * np.pi / 6) for i, month in enumerate(months)])
    data = pd.DataFrame({"month": months, "NDCI_chla": values})
    result = backtest_seasonal_forecast(data)
    assert result.metrics["n_holdout_months"].eq(24).all()
    assert result.predictions.index.min() == pd.Timestamp("2023-01-01")
    assert result.predictions.index.max() == pd.Timestamp("2024-12-01")
    assert result.metrics["mae"].notna().all()


def test_driver_model_has_chronological_holdout_and_baseline():
    months = pd.date_range("2018-01-01", periods=72, freq="MS")
    rng = np.random.default_rng(42)
    rainfall = rng.uniform(0, 200, len(months))
    wind = rng.uniform(1, 6, len(months))
    nyando = rng.uniform(5, 50, len(months))
    sondu = rng.uniform(3, 35, len(months))
    bloom = 0.001 * rainfall + 0.02 * wind + 0.003 * nyando + 0.004 * sondu
    data = pd.DataFrame(
        {
            "month": months,
            "bloom_metric": bloom,
            "rainfall_mm": rainfall,
            "wind_speed_ms": wind,
            "nyando_inflow_m3s": nyando,
            "sondu_miriu_inflow_m3s": sondu,
        }
    )
    result = train_driver_attribution(data, permutation_repeats=3)
    assert result.metrics["n_training_months"] == 54
    assert result.metrics["n_holdout_months"] == 18
    assert result.metrics["train_end"] < result.metrics["holdout_start"]
    assert result.metrics["n_holdout_months"] >= 12
    assert "climatology_mae" in result.metrics
    assert not result.permutation_importances.empty


def test_flowpath_analysis_requires_measured_path_distances():
    frame = pd.DataFrame(
        {
            "distance_along_flow_m": [0, 500, 1000, 1500],
            "bloom_frequency": [0.8, 0.6, 0.4, 0.2],
        }
    )
    result = analyze_flowpath_distance(frame)
    assert result["distance_basis"].startswith("caller-supplied")
    assert result["linear_slope_response_per_km"] < 0
    assert result["causal_interpretation"] == "not established"
    with pytest.raises(ValueError, match="Missing flow-path columns"):
        analyze_flowpath_distance(pd.DataFrame({"straight_line_distance_m": [1, 2, 3]}))


def test_hamed_rao_trend_adjusts_variance_and_rejects_missing_months():
    months = pd.date_range("2018-01-01", periods=48, freq="MS")
    values = np.linspace(0.1, 0.4, len(months)) + 0.03 * np.sin(
        2 * np.pi * np.arange(len(months)) / 12
    )
    result = hamed_rao_mann_kendall(pd.Series(values, index=months))
    assert result["serial_correlation_adjusted"] is True
    assert result["trend"] == "increasing"
    assert result["sen_slope_ndci_per_month"] > 0
    with pytest.raises(ValueError, match="complete monthly series"):
        hamed_rao_mann_kendall(pd.Series(values[:-1], index=months[:-1]).drop(months[10]))


def test_bloom_threshold_sensitivity_reports_pixel_frequency_and_hotspot_overlap():
    observations = pd.DataFrame(
        {
            "pixel_id": ["shore"] * 4 + ["open"] * 4,
            "month": list(pd.date_range("2020-01-01", periods=4, freq="MS")) * 2,
            "ndci": [0.12, 0.22, 0.14, 0.24, 0.08, 0.12, 0.10, 0.16],
        }
    )
    frequencies, stability = bloom_threshold_sensitivity(
        observations,
        hotspot_frequency=0.5,
    )
    assert set(frequencies["ndci_threshold"]) == {0.1, 0.15, 0.2, 0.25, 0.3}
    assert frequencies.loc[
        (frequencies["pixel_id"] == "shore") & (frequencies["ndci_threshold"] == 0.15),
        "bloom_frequency",
    ].item() == 0.5
    assert stability.loc[
        stability["ndci_threshold"] == 0.15, "jaccard_vs_reference"
    ].item() == 1.0


def test_zonal_summary_applies_shoreline_buffer_and_requires_all_gulf_zones():
    zones = ("Kisumu Bay", "Inner Gulf", "Homa Bay", "Kendu Bay", "Outer Gulf")
    frame = pd.DataFrame(
        {
            "month": ["2024-01-01"] * 10,
            "zone": [zone for zone in zones for _ in range(2)],
            "pixel_id": [f"{zone}-{index}" for zone in zones for index in range(2)],
            "distance_to_shore_m": [40, 100] * len(zones),
            "ndci": np.arange(10, dtype=float),
        }
    )
    result = summarize_gulf_zones(frame)
    assert result["zone"].nunique() == 5
    assert result["pixel_count"].eq(1).all()
    assert result["shoreline_buffer_m"].eq(80).all()
    with pytest.raises(ValueError, match="exactly"):
        summarize_gulf_zones(frame[frame["zone"] != "Kisumu Bay"])


def test_event_rainfall_lags_use_only_prior_days_and_satellite_matchups():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    rain = np.zeros(len(dates))
    rain[[5, 15, 25]] = [20, 40, 60]
    target = np.full(len(dates), np.nan)
    target[[12, 22, 32]] = [1, 2, 3]
    frame = pd.DataFrame(
        {"date": dates, "rainfall_mm": rain, "ndci": target, "tss_proxy": target * 2}
    )
    matchups, summary = event_rainfall_lags(frame, target_columns=("ndci",), lag_days=(7,))
    assert matchups["antecedent_rainfall_mm"].tolist() == [20, 40, 60]
    assert summary.loc[0, "n_matchups"] == 3
    assert summary.loc[0, "spearman_rho"] == 1.0
