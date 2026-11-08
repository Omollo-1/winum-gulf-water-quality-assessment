"""FastAPI endpoints backed by Google Earth Engine Sentinel-2 data."""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request

from .gee_service import (
    EarthEngineConfigurationError,
    EarthEngineRequestError,
    EarthEngineWaterQualityService,
)


def _validate_floating_thresholds(
    ndvi_threshold: float | None, fai_threshold: float | None
) -> None:
    if (ndvi_threshold is None) != (fai_threshold is None):
        raise HTTPException(
            status_code=422,
            detail="Supply both locally calibrated NDVI and FAI thresholds, or neither.",
        )
    if fai_threshold is not None and not math.isfinite(fai_threshold):
        raise HTTPException(status_code=422, detail="FAI threshold must be finite.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = app.state.gee_service
    service.initialize()
    yield


def create_app(gee_service: Any | None = None) -> FastAPI:
    """Build the service; inject a fake ``gee_service`` for offline API tests."""
    app = FastAPI(
        title="Winam Gulf Water Quality API",
        description="Live Sentinel-2 water-quality summaries and map tiles from Google Earth Engine.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.gee_service = (
        gee_service if gee_service is not None else EarthEngineWaterQualityService()
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        service = request.app.state.gee_service
        return {
            "status": "ok" if service.initialized else "not_initialized",
            "gee_initialized": bool(service.initialized),
            "source": "COPERNICUS/S2_SR_HARMONIZED",
        }

    @app.get("/api/v1/timeseries")
    def timeseries(
        request: Request,
        start: date = Query(..., description="First requested month; YYYY-MM-DD."),
        end: date = Query(..., description="Last requested month; YYYY-MM-DD."),
        ndvi_threshold: float | None = Query(None, ge=-1, le=1),
        fai_threshold: float | None = Query(None),
    ) -> dict[str, Any]:
        service = request.app.state.gee_service
        _validate_floating_thresholds(ndvi_threshold, fai_threshold)
        if end < start:
            raise HTTPException(status_code=422, detail="end must be on or after start.")
        try:
            return service.timeseries(
                start,
                end,
                ndvi_threshold=ndvi_threshold,
                fai_threshold=fai_threshold,
            )
        except EarthEngineConfigurationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except EarthEngineRequestError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/v1/maps/{layer}")
    def map_layer(
        request: Request,
        layer: str,
        month: date = Query(..., description="Any date in the target month; YYYY-MM-DD."),
        ndvi_threshold: float | None = Query(None, ge=-1, le=1),
        fai_threshold: float | None = Query(None),
    ) -> dict[str, Any]:
        service = request.app.state.gee_service
        _validate_floating_thresholds(ndvi_threshold, fai_threshold)
        if layer not in {"ndci", "ndvi", "fai", "hyacinth_candidate"}:
            raise HTTPException(
                status_code=422,
                detail="layer must be ndci, ndvi, fai, or hyacinth_candidate.",
            )
        if layer == "hyacinth_candidate" and (
            ndvi_threshold is None or fai_threshold is None
        ):
            raise HTTPException(
                status_code=422,
                detail="hyacinth_candidate requires locally calibrated ndvi_threshold and fai_threshold.",
            )
        try:
            return service.map_layer(
                month,
                layer,
                ndvi_threshold=ndvi_threshold,
                fai_threshold=fai_threshold,
            )
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except EarthEngineConfigurationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except EarthEngineRequestError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


app = create_app()
