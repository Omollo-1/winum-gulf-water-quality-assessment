"""Regional Lake Victoria chlorophyll calibration using the Deirmendjian dataset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from openpyxl import load_workbook

from winam_gulf.field_validation import FieldCalibration, evaluate_field_calibration


ZENODO_RECORD_URL = "https://zenodo.org/records/4247352"
ZENODO_FILE_URL = (
    "https://zenodo.org/records/4247352/files/"
    "Deirmendjian_et_al_2020_Ecological_Changes_Lake_Victoria.xlsx?download=1"
)
DEFAULT_DATASET_PATH = Path("deirmendjian_lake_victoria_2020.xlsx")
DEFAULT_MATCH_WINDOW_DAYS = 3
SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
SENTINEL2_SCENE_CLOUD_LIMIT = 40
SENTINEL2_NDCI_SCALE_METERS = 20
STATION_BUFFER_METERS = 20
_MILLISECONDS_PER_DAY = 24 * 60 * 60 * 1000
_EMPTY_ROW_LIMIT = 100


@dataclass(frozen=True)
class RegionalSceneMatches:
    """Matched and unmatched Ugandan field observations."""

    matches: pd.DataFrame
    unmatched: pd.DataFrame


def download_deirmendjian_dataset(
    destination: str | Path = DEFAULT_DATASET_PATH,
    *,
    timeout_seconds: float = 120,
) -> Path:
    """Download the public Zenodo workbook if it is not already present."""
    path = Path(destination)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with requests.get(
            ZENODO_FILE_URL,
            stream=True,
            timeout=(15, timeout_seconds),
        ) as response:
            response.raise_for_status()
            with NamedTemporaryFile(
                mode="wb",
                prefix=f"{path.name}.",
                suffix=".part",
                dir=path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        temporary_file.write(chunk)
        temporary_path.replace(path)
    except (OSError, requests.RequestException):
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return path


def _column_positions(
    header: Iterable[Any],
    required: Iterable[str],
    *,
    sheet_name: str,
) -> dict[str, int]:
    positions = {str(name).strip(): index for index, name in enumerate(header) if name is not None}
    missing = set(required) - set(positions)
    if missing:
        raise ValueError(
            f"Zenodo sheet {sheet_name!r} is missing required columns: {sorted(missing)}"
        )
    return {name: positions[name] for name in required}


def _rows_until_empty_streak(sheet: Any, *, min_row: int) -> Iterable[tuple[Any, ...]]:
    empty_streak = 0
    for row in sheet.iter_rows(min_row=min_row, values_only=True):
        if all(value is None for value in row):
            empty_streak += 1
            if empty_streak >= _EMPTY_ROW_LIMIT:
                break
            continue
        empty_streak = 0
        yield row


def _parse_visit_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text or text.casefold() == "no ctd":
        return None
    try:
        return datetime.strptime(text, "%m/%d/%Y").date()
    except ValueError:
        try:
            return datetime.fromisoformat(text).date()
        except ValueError:
            return None


def _numeric_value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _visit_date_lookup(workbook: Any) -> dict[tuple[str, float, float], date]:
    sheet_name = "Profile_exo_probe"
    if sheet_name not in workbook.sheetnames:
        raise ValueError(f"Zenodo workbook is missing the {sheet_name!r} sheet.")
    sheet = workbook[sheet_name]
    required = (
        "Mission",
        "Station",
        "Longitude (°E)",
        "Latitude (°N)",
        "Date (MM/DD/YYYY)",
    )
    columns = _column_positions(
        next(sheet.iter_rows(min_row=1, max_row=1, values_only=True)),
        required,
        sheet_name=sheet_name,
    )
    station_coordinates: dict[tuple[str, str], tuple[float, float]] = {}
    dates_by_location: dict[tuple[str, float, float], set[date]] = {}

    for row in _rows_until_empty_streak(sheet, min_row=2):
        mission_value = row[columns["Mission"]]
        station_value = row[columns["Station"]]
        if mission_value is None or station_value is None:
            continue
        visit_key = (str(mission_value).strip().upper(), str(station_value).strip())
        longitude = _numeric_value(row[columns["Longitude (°E)"]])
        latitude = _numeric_value(row[columns["Latitude (°N)"]])
        if longitude is not None and latitude is not None:
            station_coordinates[visit_key] = (longitude, latitude)
        coordinates = station_coordinates.get(visit_key)
        sample_date = _parse_visit_date(row[columns["Date (MM/DD/YYYY)"]])
        if coordinates is None or sample_date is None:
            continue
        location_key = (
            visit_key[0],
            round(coordinates[0], 6),
            round(coordinates[1], 6),
        )
        dates_by_location.setdefault(location_key, set()).add(sample_date)

    ambiguous = {key: values for key, values in dates_by_location.items() if len(values) > 1}
    if ambiguous:
        raise ValueError(
            "The probe sheet maps a mission and location to multiple dates; "
            "resolve the field visit dates before calibration."
        )
    return {key: next(iter(values)) for key, values in dates_by_location.items()}


def load_deirmendjian_surface_samples(
    workbook_path: str | Path,
) -> pd.DataFrame:
    """Read surface chlorophyll and reconcile visit dates from the probe sheet.

    The workbook stores chlorophyll as µg/L, numerically equivalent to mg/m³.
    Rows with no unambiguous date remain in the returned frame with ``NaT``.
    """
    path = Path(workbook_path)
    if not path.is_file():
        raise FileNotFoundError(f"Deirmendjian workbook does not exist: {path}")

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet_name = "Discrete-Profiles"
        if sheet_name not in workbook.sheetnames:
            raise ValueError(f"Zenodo workbook is missing the {sheet_name!r} sheet.")
        sheet = workbook[sheet_name]
        required = (
            "Mission",
            "Nom_station",
            "sampling depth (m)",
            "Surface VS Bottom",
            "Longitude (°E)",
            "Latitude (°N)",
            "chl_a (µg/L)",
        )
        columns = _column_positions(
            next(sheet.iter_rows(min_row=1, max_row=1, values_only=True)),
            required,
            sheet_name=sheet_name,
        )
        dates = _visit_date_lookup(workbook)
        rows: list[dict[str, Any]] = []

        for row in _rows_until_empty_streak(sheet, min_row=2):
            mission_value = row[columns["Mission"]]
            if mission_value is None:
                continue
            if str(row[columns["Surface VS Bottom"]]).strip().casefold() != "surface":
                continue

            station_value = row[columns["Nom_station"]]
            longitude = _numeric_value(row[columns["Longitude (°E)"]])
            latitude = _numeric_value(row[columns["Latitude (°N)"]])
            chlorophyll = _numeric_value(row[columns["chl_a (µg/L)"]])
            if (
                station_value is None
                or longitude is None
                or latitude is None
                or chlorophyll is None
                or chlorophyll < 0
            ):
                continue

            mission = str(mission_value).strip().upper()
            station_id = str(station_value).strip().upper()
            sample_date = dates.get(
                (mission, round(longitude, 6), round(latitude, 6))
            )
            rows.append(
                {
                    "sample_id": len(rows),
                    "mission": mission,
                    "station_id": station_id,
                    "site_id": f"UG-{station_id}",
                    "region": "Ugandan waters, Lake Victoria",
                    "longitude": longitude,
                    "latitude": latitude,
                    "sample_date": pd.Timestamp(sample_date) if sample_date else pd.NaT,
                    "sampling_depth_m": _numeric_value(
                        row[columns["sampling depth (m)"]]
                    ),
                    "chlorophyll_a_mg_m3": chlorophyll,
                    "field_unit_original": "µg/L",
                }
            )
    finally:
        workbook.close()

    if not rows:
        raise ValueError("No valid surface chlorophyll observations were found in the workbook.")
    return pd.DataFrame(rows)


def select_nearest_valid_sentinel2_matches(
    observations: pd.DataFrame,
    scene_samples: pd.DataFrame,
    *,
    max_match_days: int = DEFAULT_MATCH_WINDOW_DAYS,
) -> RegionalSceneMatches:
    """Choose the nearest valid NDCI scene sample for each dated field visit."""
    if max_match_days < 0:
        raise ValueError("max_match_days must be non-negative.")
    if not scene_samples.empty and "ndci" not in scene_samples:
        scene_samples = scene_samples.copy()
        scene_samples["ndci"] = scene_samples.get("ndci_median", np.nan)
    required_observation_columns = {
        "sample_id",
        "sample_date",
        "site_id",
        "mission",
        "chlorophyll_a_mg_m3",
    }
    missing = required_observation_columns - set(observations.columns)
    if missing:
        raise ValueError(f"Missing field sample columns: {sorted(missing)}")
    required_scene_columns = {
        "sample_id",
        "ndci",
        "scene_time_ms",
        "scene_date",
        "scene_cloud_percent",
    }
    missing = required_scene_columns - set(scene_samples.columns)
    if missing and not scene_samples.empty:
        raise ValueError(f"Missing Sentinel-2 sample columns: {sorted(missing)}")

    field = observations.copy()
    field["sample_date"] = pd.to_datetime(field["sample_date"], errors="coerce")
    date_missing = field["sample_date"].isna()
    undated = field.loc[date_missing].copy()
    if not undated.empty:
        undated["match_status"] = "field_visit_date_unavailable"
    dated = field.loc[~date_missing].copy()
    if scene_samples.empty:
        unmatched = dated.copy()
        unmatched["match_status"] = "no_sentinel2_scene_in_window"
        return RegionalSceneMatches(
            matches=pd.DataFrame(),
            unmatched=pd.concat([undated, unmatched], ignore_index=True),
        )

    scenes = scene_samples.copy()
    scenes["ndci"] = pd.to_numeric(scenes["ndci"], errors="coerce")
    scenes["scene_time_ms"] = pd.to_numeric(scenes["scene_time_ms"], errors="coerce")
    scenes["sample_id"] = pd.to_numeric(scenes["sample_id"], errors="coerce")
    scenes["scene_date"] = pd.to_datetime(scenes["scene_date"], errors="coerce")
    scenes["scene_cloud_percent"] = pd.to_numeric(
        scenes["scene_cloud_percent"], errors="coerce"
    )
    scenes = scenes.replace([np.inf, -np.inf], np.nan)
    field["sample_id"] = pd.to_numeric(field["sample_id"], errors="raise")
    date_lookup = {
        int(row["sample_id"]): pd.Timestamp(row["sample_date"]).normalize()
        for row in field.loc[
            field["sample_date"].notna(), ["sample_id", "sample_date"]
        ].to_dict(orient="records")
    }
    scenes["sample_date"] = scenes["sample_id"].map(date_lookup)
    scenes["offset_days"] = (
        scenes["scene_date"].dt.normalize() - scenes["sample_date"]
    ).abs().dt.days
    valid_scenes = scenes.loc[
        scenes["ndci"].notna()
        & scenes["scene_time_ms"].notna()
        & scenes["scene_date"].notna()
        & scenes["offset_days"].le(max_match_days)
    ].copy()
    valid_scenes["scene_cloud_percent"] = valid_scenes[
        "scene_cloud_percent"
    ].fillna(float("inf"))
    nearest = (
        valid_scenes.sort_values(
            ["sample_id", "offset_days", "scene_cloud_percent", "scene_time_ms"],
            kind="stable",
        )
        .drop_duplicates("sample_id", keep="first")
    )
    scene_columns = [
        "sample_id",
        "ndci",
        "scene_date",
        "scene_cloud_percent",
        "offset_days",
    ]
    matches = dated.merge(nearest[scene_columns], on="sample_id", how="inner")

    valid_ids = set(nearest["sample_id"].astype(int))
    window_candidate_ids = set(
        scenes.loc[scenes["offset_days"].le(max_match_days), "sample_id"]
        .dropna()
        .astype(int)
    )
    unmatched = dated.loc[~dated["sample_id"].isin(valid_ids)].copy()
    unmatched["match_status"] = np.where(
        unmatched["sample_id"].isin(window_candidate_ids),
        "no_valid_clear_water_pixel_in_window",
        "no_sentinel2_scene_in_window",
    )
    unmatched = pd.concat([undated, unmatched], ignore_index=True)
    return RegionalSceneMatches(matches=matches, unmatched=unmatched)


def match_deirmendjian_samples_to_sentinel2(
    observations: pd.DataFrame,
    ee_module: Any,
    *,
    max_match_days: int = DEFAULT_MATCH_WINDOW_DAYS,
) -> RegionalSceneMatches:
    """Sample the nearest clear-water Sentinel-2 NDCI within the selected window."""
    if max_match_days < 0:
        raise ValueError("max_match_days must be non-negative.")
    required = {
        "sample_id",
        "mission",
        "site_id",
        "sample_date",
        "longitude",
        "latitude",
        "chlorophyll_a_mg_m3",
    }
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Missing regional field sample columns: {sorted(missing)}")

    field = observations.copy()
    field["sample_date"] = pd.to_datetime(field["sample_date"], errors="coerce")
    undated = field.loc[field["sample_date"].isna()].copy()
    dated = field.loc[field["sample_date"].notna()].copy()
    if dated.empty:
        return select_nearest_valid_sentinel2_matches(
            field, pd.DataFrame(), max_match_days=max_match_days
        )

    point_features = []
    for row in dated.to_dict(orient="records"):
        sample_date = pd.Timestamp(row["sample_date"]).normalize()
        point_features.append(
            ee_module.Feature(
                ee_module.Geometry.Point(
                    [float(row["longitude"]), float(row["latitude"])]
                ).buffer(STATION_BUFFER_METERS),
                {
                    "sample_id": int(row["sample_id"]),
                    "mission": row["mission"],
                    "site_id": row["site_id"],
                    "sample_date": sample_date.strftime("%Y-%m-%d"),
                    "sample_time_ms": int(sample_date.value // 1_000_000),
                    "chlorophyll_a_mg_m3": float(row["chlorophyll_a_mg_m3"]),
                },
            )
        )
    points = ee_module.FeatureCollection(point_features)
    candidate_window_days = max_match_days + 1
    start_date = dated["sample_date"].min().date() - timedelta(
        days=candidate_window_days
    )
    end_date_exclusive = (
        dated["sample_date"].max().date()
        + timedelta(days=candidate_window_days + 1)
    )

    def add_ndci(image: Any) -> Any:
        image = ee_module.Image(image)
        scl = image.select("SCL")
        clear = (
            scl.neq(3)
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )
        reflectance = image.select(["B3", "B4", "B5", "B8"]).multiply(0.0001)
        open_water = reflectance.normalizedDifference(["B3", "B8"]).gt(0)
        ndci = (
            reflectance.updateMask(clear)
            .updateMask(open_water)
            .normalizedDifference(["B5", "B4"])
            .rename("ndci")
        )
        return ndci.copyProperties(
            image, ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"]
        )

    images = (
        ee_module.ImageCollection(SENTINEL2_COLLECTION)
        .filterBounds(points.geometry())
        .filterDate(start_date.isoformat(), end_date_exclusive.isoformat())
        .filter(ee_module.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", SENTINEL2_SCENE_CLOUD_LIMIT))
        .map(add_ndci)
    )
    image_count = int(images.size().getInfo())
    if image_count == 0:
        unmatched = dated.copy()
        unmatched["match_status"] = "no_sentinel2_scene_in_window"
        if not undated.empty:
            undated["match_status"] = "field_visit_date_unavailable"
            unmatched = pd.concat([undated, unmatched], ignore_index=True)
        return RegionalSceneMatches(matches=pd.DataFrame(), unmatched=unmatched)

    property_names = [
        "sample_id",
        "mission",
        "site_id",
        "sample_date",
        "sample_time_ms",
        "chlorophyll_a_mg_m3",
        "ndci",
        "scene_time_ms",
        "scene_date",
        "scene_cloud_percent",
    ]

    def sample_image(image_value: Any) -> Any:
        image = ee_module.Image(image_value)
        scene_time_ms = ee_module.Number(image.get("system:time_start"))
        within_window = ee_module.Filter.And(
            ee_module.Filter.gte(
                "sample_time_ms",
                scene_time_ms.subtract(candidate_window_days * _MILLISECONDS_PER_DAY),
            ),
            ee_module.Filter.lte(
                "sample_time_ms",
                scene_time_ms.add(candidate_window_days * _MILLISECONDS_PER_DAY),
            ),
        )
        scene_points = points.filter(within_window)
        sampled = image.reduceRegions(
            collection=scene_points,
            reducer=ee_module.Reducer.median(),
            scale=SENTINEL2_NDCI_SCALE_METERS,
            tileScale=4,
            maxPixelsPerRegion=100,
        )

        def annotate(feature: Any) -> Any:
            feature = ee_module.Feature(feature)
            return feature.set(
                {
                    "scene_time_ms": scene_time_ms,
                    "scene_date": ee_module.Date(scene_time_ms).format("YYYY-MM-dd"),
                    "scene_cloud_percent": image.get("CLOUDY_PIXEL_PERCENTAGE"),
                }
            )

        return sampled.map(annotate).select(property_names, retainGeometry=False)

    candidate_samples = ee_module.FeatureCollection(
        ee_module.List(images.toList(image_count)).map(sample_image)
    ).flatten()
    candidate_info = candidate_samples.getInfo()
    candidate_rows = []
    for feature in candidate_info.get("features", []):
        properties = feature["properties"]
        if "ndci" not in properties:
            properties["ndci"] = properties.get("ndci_median")
        candidate_rows.append(properties)
    candidate_frame = pd.DataFrame(candidate_rows)
    if not candidate_frame.empty and "ndci" not in candidate_frame:
        # Earth Engine omits a reduction property when every pixel is masked.
        candidate_frame["ndci"] = np.nan
    selected = select_nearest_valid_sentinel2_matches(
        dated,
        candidate_frame,
        max_match_days=max_match_days,
    )
    if undated.empty:
        return selected
    undated["match_status"] = "field_visit_date_unavailable"
    return RegionalSceneMatches(
        matches=selected.matches,
        unmatched=pd.concat([undated, selected.unmatched], ignore_index=True),
    )


def evaluate_regional_chla_calibration(
    matched_observations: pd.DataFrame,
) -> FieldCalibration:
    """Fit and site-holdout validate the regional NDCI-to-chlorophyll equation."""
    required = {"site_id", "chlorophyll_a_mg_m3", "ndci"}
    missing = required - set(matched_observations.columns)
    if missing:
        raise ValueError(f"Missing matched regional calibration columns: {sorted(missing)}")
    if matched_observations["site_id"].nunique() < 3:
        raise ValueError(
            "At least three distinct source station IDs are required for "
            "leave-one-site-out regional calibration."
        )
    calibration_data = matched_observations.assign(
        field_variable="chlorophyll-a",
        observed_value=matched_observations["chlorophyll_a_mg_m3"],
        field_unit="mg/m³",
        satellite_method="Sentinel-2 SR Harmonized (Sen2Cor), NDCI",
        satellite_proxy=matched_observations["ndci"],
    )
    return evaluate_field_calibration(calibration_data)
