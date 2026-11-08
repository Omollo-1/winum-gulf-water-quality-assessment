"""Google Earth Engine access for the FastAPI service."""

from __future__ import annotations

import math
import os
from datetime import date
from typing import Any


START_DATE = date(2018, 1, 1)
SEARCH_BOUNDS = [34.30, -0.55, 35.05, 0.15]
GULF_CUTLINE_LONGITUDE = 34.22
MAX_TIMESERIES_MONTHS = 120


class EarthEngineConfigurationError(RuntimeError):
    """Earth Engine cannot be initialized with the available project credentials."""


class EarthEngineRequestError(RuntimeError):
    """A Google Earth Engine operation failed."""


class EarthEngineWaterQualityService:
    """Build monthly water-quality products from GEE Sentinel-2 SR imagery."""

    def __init__(self, ee_module: Any | None = None) -> None:
        if ee_module is None:
            import ee as ee_module

        self.ee = ee_module
        self.project_id: str | None = None
        self._aoi: Any | None = None

    def initialize(self) -> None:
        project_id = os.getenv("EARTH_ENGINE_PROJECT", "").strip()
        if not project_id:
            raise EarthEngineConfigurationError(
                "Set EARTH_ENGINE_PROJECT to a Google Cloud project enabled for Earth Engine."
            )
        try:
            self.ee.Initialize(project=project_id)
        except self.ee.EEException as error:
            raise EarthEngineConfigurationError(
                "Earth Engine initialization failed. Authenticate with "
                "`earthengine authenticate` or configure Application Default Credentials, "
                "then verify serviceusage.services.use for EARTH_ENGINE_PROJECT."
            ) from error
        self.project_id = project_id

    @property
    def initialized(self) -> bool:
        return self.project_id is not None

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise EarthEngineConfigurationError(
                "Earth Engine is not initialized. Check the FastAPI startup logs."
            )

    def _get_aoi(self) -> Any:
        self._require_initialized()
        if self._aoi is not None:
            return self._aoi
        ee = self.ee
        gulf_side = ee.Geometry.Rectangle(
            [GULF_CUTLINE_LONGITUDE, SEARCH_BOUNDS[1], SEARCH_BOUNDS[2], SEARCH_BOUNDS[3]]
        )
        permanent_water = (
            ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
            .select("occurrence")
            .gt(75)
            .clip(gulf_side)
            .selfMask()
        )
        water_vectors = permanent_water.reduceToVectors(
            geometry=gulf_side,
            scale=100,
            geometryType="polygon",
            eightConnected=True,
            maxPixels=1e10,
            bestEffort=True,
        )
        water_vectors = water_vectors.map(
            lambda feature: feature.set("area", feature.geometry().area(maxError=100))
        )
        largest_feature = water_vectors.sort("area", False).first()
        self._aoi = ee.Feature(largest_feature).geometry()
        return self._aoi

    def _sentinel_collection(
        self,
        start: date,
        end_exclusive: date,
        *,
        ndvi_threshold: float | None = None,
        fai_threshold: float | None = None,
    ) -> Any:
        ee = self.ee
        aoi = self._get_aoi()
        floating_thresholds = self._validate_hyacinth_thresholds(
            ndvi_threshold, fai_threshold
        )

        def mask_clouds(image: Any) -> Any:
            image = ee.Image(image)
            scl = image.select("SCL")
            cloud_mask = (
                scl.neq(3)
                .And(scl.neq(8))
                .And(scl.neq(9))
                .And(scl.neq(10))
                .And(scl.neq(11))
            )
            reflectance = (
                image.updateMask(cloud_mask)
                .select(["B2", "B3", "B4", "B5", "B8", "B11"])
                .multiply(0.0001)
                .clamp(0, 1)
                .copyProperties(image, ["system:time_start"])
            )
            return reflectance

        def add_indices(image: Any) -> Any:
            image = ee.Image(image)
            ndwi = image.normalizedDifference(["B3", "B8"])
            open_water = image.updateMask(ndwi.gt(0))
            red = image.select("B4")
            nir = image.select("B8")
            fai = nir.subtract(
                red.add(
                    image.select("B11")
                    .subtract(red)
                    .multiply((842 - 665) / (1610 - 665))
                )
            ).rename("fai")
            ndci = open_water.normalizedDifference(["B5", "B4"]).rename("ndci")
            ndvi = image.normalizedDifference(["B8", "B4"]).rename("ndvi")
            tss = open_water.normalizedDifference(["B4", "B3"]).rename("tss_proxy")
            cdom = (
                open_water.select("B2")
                .divide(open_water.select("B3"))
                .rename("cdom_proxy")
            )
            index_bands = [ndci, ndvi, fai, tss, cdom]
            if floating_thresholds is not None:
                candidate_mask = (
                    ndvi.gte(floating_thresholds[0]).And(fai.gte(floating_thresholds[1]))
                )
                ndci_excluding_candidates = (
                    open_water.updateMask(candidate_mask.Not())
                    .normalizedDifference(["B5", "B4"])
                    .rename("ndci_excluding_candidate_vegetation")
                )
                index_bands.append(ndci_excluding_candidates)
            return image.addBands(index_bands)

        return (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start.isoformat(), end_exclusive.isoformat())
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
            .map(mask_clouds)
            .map(add_indices)
        )

    @staticmethod
    def _month_range(start: date, end: date) -> tuple[date, date, int]:
        start_month = start.replace(day=1)
        end_month = end.replace(day=1)
        if start_month < START_DATE:
            raise ValueError(f"Data requests must start on or after {START_DATE:%Y-%m}.")
        current_month = date.today().replace(day=1)
        latest_complete_month = (
            date(current_month.year - 1, 12, 1)
            if current_month.month == 1
            else date(current_month.year, current_month.month - 1, 1)
        )
        if end_month > latest_complete_month:
            raise ValueError(
                f"Data requests must end on or before {latest_complete_month:%Y-%m}, "
                "the latest complete month."
            )
        month_count = (end_month.year - start_month.year) * 12 + end_month.month - start_month.month + 1
        if month_count < 1:
            raise ValueError("The end month must be on or after the start month.")
        if month_count > MAX_TIMESERIES_MONTHS:
            raise ValueError(f"Requests are limited to {MAX_TIMESERIES_MONTHS} months.")
        next_month = (
            date(end_month.year + 1, 1, 1)
            if end_month.month == 12
            else date(end_month.year, end_month.month + 1, 1)
        )
        return start_month, next_month, month_count

    @staticmethod
    def _month_starts(start_month: date, count: int) -> list[date]:
        months = [start_month]
        for _ in range(1, count):
            previous = months[-1]
            next_year = previous.year + (previous.month == 12)
            next_month = 1 if previous.month == 12 else previous.month + 1
            months.append(date(next_year, next_month, 1))
        return months

    @staticmethod
    def _validate_hyacinth_thresholds(
        ndvi_threshold: float | None, fai_threshold: float | None
    ) -> tuple[float, float] | None:
        if (ndvi_threshold is None) != (fai_threshold is None):
            raise ValueError("Both locally calibrated NDVI and FAI thresholds must be supplied together.")
        if ndvi_threshold is None:
            return None
        if fai_threshold is None:
            raise ValueError("Both locally calibrated NDVI and FAI thresholds must be supplied together.")
        if (
            not math.isfinite(ndvi_threshold)
            or not math.isfinite(fai_threshold)
            or not -1 <= ndvi_threshold <= 1
        ):
            raise ValueError("NDVI must be finite and between -1 and 1; FAI must be finite.")
        return ndvi_threshold, fai_threshold

    def timeseries(
        self,
        start: date,
        end: date,
        *,
        ndvi_threshold: float | None = None,
        fai_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Return monthly AOI means and scene counts from one GEE feature request."""
        self._require_initialized()
        floating_thresholds = self._validate_hyacinth_thresholds(ndvi_threshold, fai_threshold)
        ee = self.ee
        start_month, end_exclusive, month_count = self._month_range(start, end)
        months = self._month_starts(start_month, month_count)
        aoi = self._get_aoi()
        collection = self._sentinel_collection(
            start_month,
            end_exclusive,
            ndvi_threshold=ndvi_threshold,
            fai_threshold=fai_threshold,
        )
        features = []

        for month_start in months:
            next_month = (
                date(month_start.year + 1, 1, 1)
                if month_start.month == 12
                else date(month_start.year, month_start.month + 1, 1)
            )
            scenes = collection.filterDate(month_start.isoformat(), next_month.isoformat())
            composite = scenes.median()
            selected_bands = ["ndci", "ndvi", "fai", "tss_proxy", "cdom_proxy"]
            if floating_thresholds is not None:
                selected_bands.append("ndci_excluding_candidate_vegetation")
            statistics = composite.select(selected_bands).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=20,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=4,
            )
            feature_properties = (
                ee.Dictionary(statistics)
                .set("month", month_start.strftime("%Y-%m"))
                .set("scene_count", scenes.size())
            )
            features.append(ee.Feature(None, feature_properties))

        try:
            result = ee.FeatureCollection(features).getInfo()
        except self.ee.EEException as error:
            raise EarthEngineRequestError("Earth Engine monthly summary request failed.") from error
        return {
            "source": "COPERNICUS/S2_SR_HARMONIZED",
            "correction": "Sen2Cor surface reflectance (GEE SR product)",
            "aoi": "JRC Global Surface Water permanent-water polygon clipped at the gulf cutline",
            "floating_vegetation_mask": (
                {
                    "ndvi_threshold": ndvi_threshold,
                    "fai_threshold": fai_threshold,
                    "interpretation": "Candidate floating vegetation mask; locally validate thresholds and class labels.",
                }
                if floating_thresholds is not None
                else None
            ),
            "months": [feature["properties"] for feature in result["features"]],
        }

    def map_layer(
        self,
        month: date,
        layer: str,
        *,
        ndvi_threshold: float | None = None,
        fai_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Return an ephemeral Earth Engine map tile URL for the requested layer."""
        self._require_initialized()
        ee = self.ee
        floating_thresholds = self._validate_hyacinth_thresholds(ndvi_threshold, fai_threshold)
        month_start = month.replace(day=1)
        current_month = date.today().replace(day=1)
        latest_complete_month = (
            date(current_month.year - 1, 12, 1)
            if current_month.month == 1
            else date(current_month.year, current_month.month - 1, 1)
        )
        if month_start < START_DATE or month_start > latest_complete_month:
            raise ValueError(
                f"Maps are available only for complete months from {START_DATE:%Y-%m} "
                f"through {latest_complete_month:%Y-%m}."
            )
        month_end = (
            date(month_start.year + 1, 1, 1)
            if month_start.month == 12
            else date(month_start.year, month_start.month + 1, 1)
        )
        aoi = self._get_aoi()
        scenes = self._sentinel_collection(
            month_start,
            month_end,
            ndvi_threshold=ndvi_threshold,
            fai_threshold=fai_threshold,
        )
        try:
            scene_count = int(scenes.size().getInfo())
            if scene_count == 0:
                raise LookupError(f"No Sentinel-2 scenes are available for {month_start:%Y-%m}.")
            composite = scenes.median().clip(aoi)
            if layer == "ndci":
                image = composite.select("ndci")
                if floating_thresholds is not None:
                    image = composite.select("ndci_excluding_candidate_vegetation")
                visualization = {
                    "min": -0.3,
                    "max": 0.6,
                    "palette": ["313695", "74add1", "ffffbf", "fdae61", "a50026"],
                }
            elif layer == "ndvi":
                image = composite.select("ndvi")
                visualization = {
                    "min": -0.2,
                    "max": 0.9,
                    "palette": ["f7fcf5", "74c476", "00441b"],
                }
            elif layer == "fai":
                image = composite.select("fai")
                visualization = {
                    "min": -0.05,
                    "max": 0.12,
                    "palette": ["313695", "ffffbf", "d7191c"],
                }
            elif layer == "hyacinth_candidate":
                if ndvi_threshold is None or fai_threshold is None:
                    raise ValueError(
                        "hyacinth_candidate requires locally calibrated ndvi_threshold and fai_threshold."
                    )
                ndvi = composite.select("ndvi")
                fai = composite.select("fai")
                image = (
                    ndvi.gte(ndvi_threshold)
                    .And(fai.gte(fai_threshold))
                    .selfMask()
                    .rename("hyacinth_candidate")
                )
                visualization = {"min": 0, "max": 1, "palette": ["39ff14"]}
            else:
                raise ValueError("layer must be ndci, ndvi, fai, or hyacinth_candidate.")
            map_id = image.getMapId(visualization)
        except (ValueError, LookupError):
            raise
        except self.ee.EEException as error:
            raise EarthEngineRequestError("Earth Engine map-tile request failed.") from error

        tile_url = map_id.get("tile_fetcher").url_format
        return {
            "month": month_start.strftime("%Y-%m"),
            "layer": layer,
            "scene_count": scene_count,
            "source": "COPERNICUS/S2_SR_HARMONIZED",
            "correction": "Sen2Cor surface reflectance (GEE SR product)",
            "interpretation": (
                "Candidate floating vegetation only; locally validate thresholds and class labels."
                if layer == "hyacinth_candidate"
                else "NDVI/FAI threshold mask applied; validate candidate vegetation labels locally."
                if layer == "ndci" and floating_thresholds is not None
                else "Satellite-derived spectral index, not a calibrated concentration."
            ),
            "tile_url": tile_url,
            "bounds": [[-0.55, 34.22], [0.15, 35.05]],
        }
