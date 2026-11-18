"""ipyleaflet notebook maps connected to the running Winam Gulf FastAPI service."""

from __future__ import annotations

import math
from datetime import date, datetime
from html import escape
from typing import Any

import requests
from ipywidgets import HTML, Layout
from ipyleaflet import LayersControl, Map, ScaleControl, TileLayer, WidgetControl, basemaps
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
    "ndci": {"Low NDCI (-0.30)": "#313695", "High NDCI (+0.60)": "#a50026"},
    "ndvi": {"Low NDVI (-0.20)": "#f7fcf5", "High NDVI (+0.90)": "#00441b"},
    "fai": {"Low FAI (-0.05)": "#313695", "High FAI (+0.12)": "#d7191c"},
    "hyacinth_candidate": {"Candidate floating vegetation": "#39ff14"},
}

LAYER_TITLES = {
    "ndci": "NDCI (chlorophyll-a proxy)",
    "ndvi": "NDVI (vegetation index)",
    "fai": "Floating Algae Index (FAI)",
    "hyacinth_candidate": "Candidate floating vegetation",
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

    layer_label = f"{LAYER_TITLES[layer]} · {payload['month']}"
    if layer == "ndci" and ndvi_threshold is not None:
        layer_label += " · floating vegetation screened"
    water_map = Map(
        center=[-0.15, 34.65],
        zoom=9,
        basemap=basemaps.OpenStreetMap.Mapnik,
        scroll_wheel_zoom=True,
        layout=Layout(
            width="100%",
            height="600px",
            border="2px solid #334155",
        ),
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

    map_details = " · ".join(
        detail
        for detail in (
            payload.get("interpretation"),
            payload.get("source"),
            payload.get("correction"),
        )
        if detail
    )
    map_title = f"Winam Gulf - {LAYER_TITLES[layer]} - {payload['month']}"
    if layer == "ndci" and ndvi_threshold is not None:
        map_title += " (floating vegetation screened)"
    title_widget = HTML(
        value=(
            '<div style="background:white;padding:8px 12px;border:1px solid #64748b;'
            'border-radius:4px;box-shadow:0 1px 5px #64748b;">'
            f'<strong>{escape(map_title)}</strong>'
            f'<div style="max-width:360px;font-size:11px;color:#475569;">'
            f'{escape(map_details)}</div>'
            "</div>"
        )
    )
    water_map.add(WidgetControl(widget=title_widget, position="topright"))
    compass_widget = HTML(
        value=(
            '<div aria-label="North arrow" style="background:white;padding:5px 8px;'
            'border:1px solid #64748b;border-radius:4px;text-align:center;'
            'box-shadow:0 1px 5px #64748b;">'
            '<strong style="font-size:12px;">N</strong>'
            '<div style="margin:2px auto 0;width:0;height:0;'
            'border-left:7px solid transparent;border-right:7px solid transparent;'
            'border-bottom:16px solid #1e293b;"></div>'
            "</div>"
        )
    )
    water_map.add(WidgetControl(widget=compass_widget, position="topleft"))
    water_map.add(ScaleControl(position="bottomleft", imperial=False))
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
