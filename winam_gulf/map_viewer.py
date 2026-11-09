"""ipyleaflet notebook maps connected to the running Winam Gulf FastAPI service."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import requests
from ipyleaflet import LayersControl, Map, TileLayer, basemaps
from ipyleaflet.leaflet import LegendControl


def _validate_thresholds(
    ndvi_threshold: float | None,
    fai_threshold: float | None,
) -> None:
    if (ndvi_threshold is None) != (fai_threshold is None):
        raise ValueError("Supply both locally calibrated NDVI and FAI thresholds, or neither.")
    if ndvi_threshold is not None:
        if not math.isfinite(ndvi_threshold) or not -1 <= ndvi_threshold <= 1:
            raise ValueError("NDVI threshold must be finite and between -1 and 1.")
        if not math.isfinite(fai_threshold):
            raise ValueError("FAI threshold must be finite.")


LAYER_LEGENDS = {
    "ndci": {"Lower NDCI": "#313695", "Higher NDCI": "#a50026"},
    "ndvi": {"Lower vegetation index": "#f7fcf5", "Higher vegetation index": "#00441b"},
    "fai": {"Lower FAI": "#313695", "Higher FAI": "#d7191c"},
    "hyacinth_candidate": {"Candidate floating vegetation": "#39ff14"},
}


def render_gee_map(
    *,
    api_url: str,
    month: date | datetime,
    layer: str = "ndci",
    ndvi_threshold: float | None = None,
    fai_threshold: float | None = None,
    timeout_seconds: float = 30,
) -> Map:
    """Fetch an ephemeral GEE tile URL from FastAPI and display it in ipyleaflet."""
    if layer not in LAYER_LEGENDS:
        raise ValueError(f"Unsupported map layer: {layer!r}.")
    if layer == "hyacinth_candidate" and (
        ndvi_threshold is None or fai_threshold is None
    ):
        raise ValueError(
            "Supply locally calibrated NDVI and FAI thresholds before rendering candidate vegetation."
        )
    _validate_thresholds(ndvi_threshold, fai_threshold)

    parameters: dict[str, str | float] = {"month": month.strftime("%Y-%m-%d")}
    if ndvi_threshold is not None:
        parameters["ndvi_threshold"] = ndvi_threshold
    if fai_threshold is not None:
        parameters["fai_threshold"] = fai_threshold
    endpoint = f"{api_url.rstrip('/')}/api/v1/maps/{layer}"
    try:
        response = requests.get(
            endpoint,
            params=parameters,
            timeout=timeout_seconds,
        )
    except requests.RequestException as error:
        raise ConnectionError(
            f"Cannot reach the Winam Gulf API at {endpoint}; start the API and check WINAM_GULF_API_URL."
        ) from error
    if response.status_code >= 400:
        detail = response.text[:500]
        raise RuntimeError(f"Winam Gulf API returned HTTP {response.status_code}: {detail}")
    payload: dict[str, Any] = response.json()
    tile_url = payload.get("tile_url")
    bounds = payload.get("bounds")
    if not tile_url or not bounds:
        raise ValueError("Map API response is missing tile_url or bounds.")

    layer_label = f"{layer.upper()} · {payload['month']}"
    if layer == "ndci" and ndvi_threshold is not None:
        layer_label += " · floating vegetation screened"
    water_map = Map(
        center=[-0.15, 34.65],
        zoom=9,
        basemap=basemaps.OpenStreetMap.Mapnik,
        scroll_wheel_zoom=True,
    )
    water_map.add(
        TileLayer(
            url=tile_url,
            name=layer_label,
            opacity=0.78,
            bounds=bounds,
            attribution="Google Earth Engine · Sentinel-2 SR Harmonized",
        )
    )
    water_map.add(LayersControl(position="topright"))
    water_map.add(
        LegendControl(
            legend=LAYER_LEGENDS[layer],
            title=layer_label,
            position="bottomright",
        )
    )
    return water_map


def fetch_gee_timeseries(
    *,
    api_url: str,
    start: date,
    end: date,
    ndvi_threshold: float | None = None,
    fai_threshold: float | None = None,
    timeout_seconds: float = 180,
) -> dict[str, Any]:
    """Fetch GEE-backed monthly summaries from the running FastAPI service."""
    endpoint = f"{api_url.rstrip('/')}/api/v1/timeseries"
    _validate_thresholds(ndvi_threshold, fai_threshold)
    parameters: dict[str, str | float] = {
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    if ndvi_threshold is not None and fai_threshold is not None:
        parameters["ndvi_threshold"] = ndvi_threshold
        parameters["fai_threshold"] = fai_threshold
    try:
        response = requests.get(
            endpoint,
            params=parameters,
            timeout=timeout_seconds,
        )
    except requests.RequestException as error:
        raise ConnectionError(
            f"Cannot reach the Winam Gulf API at {endpoint}; start the API and check WINAM_GULF_API_URL."
        ) from error
    if response.status_code >= 400:
        raise RuntimeError(
            f"Winam Gulf API returned HTTP {response.status_code}: {response.text[:500]}"
        )
    return response.json()
