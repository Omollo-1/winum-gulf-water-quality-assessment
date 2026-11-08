import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from winam_gulf.regional_chla import (
    evaluate_regional_chla_calibration,
    load_deirmendjian_surface_samples,
    select_nearest_valid_sentinel2_matches,
)


DISCRETE_HEADERS = [
    "Mission",
    "Nom_station",
    "sampling depth (m)",
    "Surface VS Bottom",
    "Longitude (°E)",
    "Latitude (°N)",
    "chl_a (µg/L)",
]
PROBE_HEADERS = [
    "Mission",
    "Station",
    "Longitude (°E)",
    "Latitude (°N)",
    "Date (MM/DD/YYYY)",
]


def test_load_deirmendjian_surface_samples_joins_dates_and_skips_depths(tmp_path):
    workbook = Workbook()
    profiles = workbook.active
    profiles.title = "Discrete-Profiles"
    profiles.append(DISCRETE_HEADERS)
    profiles.append(["LAVIGAS_1", "UL1", 1, "Surface", 33.25, 0.4, 12.5])
    profiles.append(["LAVIGAS_1", "UL1", 10, "NA", 33.25, 0.4, 30.0])
    profiles.append(["LAVIGAS_1", "UL2", 1, "Surface", 33.3, 0.5, 8.0])

    probes = workbook.create_sheet("Profile_exo_probe")
    probes.append(PROBE_HEADERS)
    probes.append(["Lavigas_1", "bay_1", 33.25, 0.4, "03/29/2018"])
    probes.append(["Lavigas_1", "bay_1", None, None, "03/29/2018"])
    workbook.create_sheet("Continuous_measurements_")
    path = tmp_path / "regional.xlsx"
    workbook.save(path)

    samples = load_deirmendjian_surface_samples(path)

    assert len(samples) == 2
    assert samples.loc[0, "chlorophyll_a_mg_m3"] == 12.5
    assert samples.loc[0, "sample_date"] == pd.Timestamp("2018-03-29")
    assert pd.isna(samples.loc[1, "sample_date"])
    assert samples.loc[0, "site_id"] == "UG-UL1"


def test_nearest_valid_scene_is_selected_within_three_day_window():
    observations = pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "sample_date": [
                pd.Timestamp("2018-03-29"),
                pd.Timestamp("2018-03-30"),
                pd.Timestamp("2018-03-31"),
            ],
            "site_id": ["UG-UL1", "UG-UL2", "UG-UL3"],
            "mission": ["LAVIGAS_1"] * 3,
            "chlorophyll_a_mg_m3": [12.0, 8.0, 10.0],
        }
    )
    scene_samples = pd.DataFrame(
        {
            "sample_id": [1, 1, 1, 1, 2],
            "ndci": [0.2, 0.3, np.nan, 0.5, 0.4],
            "scene_time_ms": [
                int(pd.Timestamp("2018-03-28").value // 1_000_000),
                int(pd.Timestamp("2018-03-31").value // 1_000_000),
                int(pd.Timestamp("2018-03-29").value // 1_000_000),
                int(pd.Timestamp("2018-04-02").value // 1_000_000),
                int(pd.Timestamp("2018-04-02 20:00").value // 1_000_000),
            ],
            "scene_date": [
                "2018-03-28",
                "2018-03-31",
                "2018-03-29",
                "2018-04-02",
                "2018-04-02",
            ],
            "scene_cloud_percent": [20.0, 5.0, 1.0, 3.0, 10.0],
        }
    )

    result = select_nearest_valid_sentinel2_matches(observations, scene_samples)

    assert result.matches["sample_id"].tolist() == [1, 2]
    assert result.matches.loc[0, "scene_date"] == pd.Timestamp("2018-03-28")
    assert result.matches.loc[0, "offset_days"] == 1
    assert result.matches.loc[1, "offset_days"] == 3
    assert result.unmatched.loc[0, "sample_id"] == 3
    assert result.unmatched.loc[0, "match_status"] == "no_sentinel2_scene_in_window"


def test_missing_ndci_reduction_property_is_treated_as_masked_scene():
    observations = pd.DataFrame(
        {
            "sample_id": [1],
            "sample_date": [pd.Timestamp("2018-03-29")],
            "site_id": ["UG-UL1"],
            "mission": ["LAVIGAS_1"],
            "chlorophyll_a_mg_m3": [12.0],
        }
    )
    scene_samples = pd.DataFrame(
        {
            "sample_id": [1],
            "scene_time_ms": [
                int(pd.Timestamp("2018-03-29").value // 1_000_000)
            ],
            "scene_date": ["2018-03-29"],
            "scene_cloud_percent": [5.0],
        }
    )

    result = select_nearest_valid_sentinel2_matches(observations, scene_samples)

    assert result.matches.empty
    assert result.unmatched.loc[0, "match_status"] == (
        "no_valid_clear_water_pixel_in_window"
    )


def test_regional_chla_calibration_uses_site_held_out_metrics():
    ndci = np.tile(np.array([0.05, 0.1, 0.15, 0.2]), 3)
    samples = pd.DataFrame(
        {
            "site_id": np.repeat(["UL1", "UL2", "UL3"], 4),
            "ndci": ndci,
            "chlorophyll_a_mg_m3": 3 + 40 * ndci,
        }
    )

    result = evaluate_regional_chla_calibration(samples)

    assert result.metrics["validation_method"] == "leave-one-site-out"
    assert result.metrics["field_unit"] == "mg/m³"
    assert result.metrics["field_variable"] == "chlorophyll-a"
    assert result.metrics["n_sites"] == 3
    assert np.isfinite(result.predictions["out_of_fold_prediction"]).all()
    with pytest.raises(ValueError, match="At least three distinct source station IDs"):
        evaluate_regional_chla_calibration(samples.loc[samples["site_id"] != "UL3"])
