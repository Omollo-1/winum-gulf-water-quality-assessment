from datetime import date

import pytest
from fastapi.testclient import TestClient

from winam_gulf.api import create_app
from winam_gulf.gee_service import EarthEngineWaterQualityService
from winam_gulf.map_viewer import fetch_gee_timeseries, render_gee_map


class FakeEarthEngineService:
    def __init__(self):
        self.initialized = False
        self.requested_layers = []

    def initialize(self):
        self.initialized = True

    def timeseries(self, start, end, *, ndvi_threshold=None, fai_threshold=None):
        return {
            "source": "COPERNICUS/S2_SR_HARMONIZED",
            "correction": "Sen2Cor surface reflectance (GEE SR product)",
            "aoi": "Winam Gulf",
            "floating_vegetation_mask": (
                {"ndvi_threshold": ndvi_threshold, "fai_threshold": fai_threshold}
                if ndvi_threshold is not None and fai_threshold is not None
                else None
            ),
            "months": [{"month": "2024-06", "scene_count": 3, "ndci": 0.12}],
        }

    def map_layer(self, month, layer, *, ndvi_threshold=None, fai_threshold=None):
        self.requested_layers.append(
            (month, layer, ndvi_threshold, fai_threshold)
        )
        return {
            "month": month.strftime("%Y-%m"),
            "layer": layer,
            "scene_count": 3,
            "source": "COPERNICUS/S2_SR_HARMONIZED",
            "correction": "Sen2Cor surface reflectance (GEE SR product)",
            "interpretation": "Satellite-derived spectral index.",
            "tile_url": "https://earthengine.googleapis.com/v1/test/{z}/{x}/{y}",
            "bounds": [[-0.55, 34.22], [0.15, 35.05]],
        }


class FakeImage:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: self


class FakeImageCollection:
    def __init__(self):
        self.map_calls = 0

    def filterBounds(self, _aoi):
        return self

    def filterDate(self, _start, _end):
        return self

    def filter(self, _filter):
        return self

    def map(self, transform):
        result = transform(FakeImage())
        assert isinstance(result, FakeImage)
        self.map_calls += 1
        return self


class FakeEarthEngineModule:
    EEException = RuntimeError
    Filter = type("Filter", (), {"lt": staticmethod(lambda *_args: object())})

    def __init__(self):
        self.collection = FakeImageCollection()

    @staticmethod
    def Image(image):
        return image

    def ImageCollection(self, _collection_id):
        return self.collection


def test_fastapi_health_timeseries_and_map_use_same_gee_service():
    service = FakeEarthEngineService()
    app = create_app(service)
    with TestClient(app) as client:
        health = client.get("/health")
        series = client.get(
            "/api/v1/timeseries",
            params={"start": "2024-06-01", "end": "2024-06-30"},
        )
        masked_series = client.get(
            "/api/v1/timeseries",
            params={
                "start": "2024-06-01",
                "end": "2024-06-30",
                "ndvi_threshold": 0.6,
                "fai_threshold": 0.02,
            },
        )
        map_result = client.get(
            "/api/v1/maps/fai",
            params={"month": "2024-06-15"},
        )

    assert health.status_code == 200
    assert health.json()["gee_initialized"] is True
    assert series.status_code == 200
    assert series.json()["months"][0]["ndci"] == 0.12
    assert masked_series.json()["floating_vegetation_mask"] == {
        "ndvi_threshold": 0.6,
        "fai_threshold": 0.02,
    }
    assert map_result.status_code == 200
    assert map_result.json()["tile_url"].startswith("https://earthengine.googleapis.com/")
    assert service.requested_layers == [(date(2024, 6, 15), "fai", None, None)]


def test_gee_sentinel_collection_applies_cloud_mask_and_returns_processed_scenes():
    ee_module = FakeEarthEngineModule()
    service = EarthEngineWaterQualityService(ee_module)
    service.project_id = "test-project"
    service._aoi = object()

    collection = service._sentinel_collection(date(2024, 6, 1), date(2024, 7, 1))

    assert collection is ee_module.collection
    assert ee_module.collection.map_calls == 2


def test_fastapi_rejects_bad_dates_layers_and_unvalidated_hyacinth_thresholds():
    service = FakeEarthEngineService()
    with TestClient(create_app(service)) as client:
        assert client.get(
            "/api/v1/timeseries",
            params={"start": "2024-07-01", "end": "2024-06-01"},
        ).status_code == 422
        assert client.get(
            "/api/v1/timeseries",
            params={
                "start": "2024-06-01",
                "end": "2024-06-30",
                "ndvi_threshold": 0.6,
            },
        ).status_code == 422
        assert client.get(
            "/api/v1/maps/unknown",
            params={"month": "2024-06-01"},
        ).status_code == 422
        response = client.get(
            "/api/v1/maps/hyacinth_candidate",
            params={"month": "2024-06-01"},
        )
    assert response.status_code == 422
    assert "locally calibrated" in response.json()["detail"]


def test_ipyleaflet_viewer_uses_map_url_returned_by_api(monkeypatch):
    service = FakeEarthEngineService()
    client = TestClient(create_app(service))
    monkeypatch.setattr(
        "winam_gulf.map_viewer.requests.get",
        lambda url, params, timeout: client.get(url.removeprefix("http://testserver"), params=params),
    )

    water_map = render_gee_map(
        api_url="http://testserver",
        month=date(2024, 6, 1),
        layer="fai",
    )

    assert water_map.center == [-0.15, 34.65]
    assert any("earthengine.googleapis.com" in layer.url for layer in water_map.layers)
    assert any("LegendControl" in control.__class__.__name__ for control in water_map.controls)
    assert service.requested_layers[-1][1] == "fai"

    payload = fetch_gee_timeseries(
        api_url="http://testserver",
        start=date(2024, 6, 1),
        end=date(2024, 6, 30),
        ndvi_threshold=0.6,
        fai_threshold=0.02,
    )
    assert payload["source"] == "COPERNICUS/S2_SR_HARMONIZED"


def test_hyacinth_candidate_map_requires_explicit_thresholds():
    with pytest.raises(ValueError, match="locally calibrated"):
        render_gee_map(
            api_url="http://testserver",
            month=date(2024, 6, 1),
            layer="hyacinth_candidate",
        )
