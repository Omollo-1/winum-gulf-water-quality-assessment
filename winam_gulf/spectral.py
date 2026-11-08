"""Reflectance indices for aquatic vegetation and chlorophyll-proxy comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_float_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def _safe_normalized_difference(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> NDArray[np.float64]:
    first, second = np.broadcast_arrays(first, second)
    denominator = first + second
    result = np.full(first.shape, np.nan, dtype=np.float64)
    np.divide(first - second, denominator, out=result, where=np.isfinite(denominator) & (denominator != 0))
    return result


def compute_ndvi(nir: ArrayLike, red: ArrayLike) -> NDArray[np.float64]:
    """Return NDVI from NIR and red surface reflectance."""
    return _safe_normalized_difference(
        _finite_float_array(nir, "nir"), _finite_float_array(red, "red")
    )


def compute_fai(
    red: ArrayLike,
    nir: ArrayLike,
    swir: ArrayLike,
    *,
    wavelengths_nm: Sequence[float] = (665.0, 842.0, 1610.0),
) -> NDArray[np.float64]:
    """Return the floating-algae index as NIR minus a red-to-SWIR baseline."""
    if len(wavelengths_nm) != 3:
        raise ValueError("wavelengths_nm must contain red, NIR, and SWIR centers.")
    red_nm, nir_nm, swir_nm = map(float, wavelengths_nm)
    if not red_nm < nir_nm < swir_nm:
        raise ValueError("Wavelength centers must satisfy red < NIR < SWIR.")

    red_array, nir_array, swir_array = np.broadcast_arrays(
        _finite_float_array(red, "red"),
        _finite_float_array(nir, "nir"),
        _finite_float_array(swir, "swir"),
    )
    red_to_swir_fraction = (nir_nm - red_nm) / (swir_nm - red_nm)
    baseline = red_array + (swir_array - red_array) * red_to_swir_fraction
    result = nir_array - baseline
    invalid = ~(np.isfinite(red_array) & np.isfinite(nir_array) & np.isfinite(swir_array))
    result[invalid] = np.nan
    return result


def compute_two_band_ndci(
    red_edge: ArrayLike, red: ArrayLike
) -> NDArray[np.float64]:
    """Return a normalized red-edge/red difference (the two-band NDCI form)."""
    return _safe_normalized_difference(
        _finite_float_array(red_edge, "red_edge"),
        _finite_float_array(red, "red"),
    )


def compute_three_band_line_height(
    left: ArrayLike,
    peak: ArrayLike,
    right: ArrayLike,
    *,
    wavelengths_nm: Sequence[float],
) -> NDArray[np.float64]:
    """Return peak reflectance above the straight continuum between two bands."""
    if len(wavelengths_nm) != 3:
        raise ValueError("wavelengths_nm must contain left, peak, and right centers.")
    left_nm, peak_nm, right_nm = map(float, wavelengths_nm)
    if not left_nm < peak_nm < right_nm:
        raise ValueError("Wavelength centers must satisfy left < peak < right.")

    left_array, peak_array, right_array = np.broadcast_arrays(
        _finite_float_array(left, "left"),
        _finite_float_array(peak, "peak"),
        _finite_float_array(right, "right"),
    )
    fraction = (peak_nm - left_nm) / (right_nm - left_nm)
    continuum = left_array + (right_array - left_array) * fraction
    result = peak_array - continuum
    invalid = ~(np.isfinite(left_array) & np.isfinite(peak_array) & np.isfinite(right_array))
    result[invalid] = np.nan
    return result


def compute_olci_indices(
    r681: ArrayLike,
    r709: ArrayLike,
    r754: ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    """Compute OLCI 681/709 nm normalized difference and 709 nm line height."""
    return {
        "ndci_709_681": compute_two_band_ndci(r709, r681),
        "line_height_709": compute_three_band_line_height(
            r681,
            r709,
            r754,
            wavelengths_nm=(681.0, 709.0, 754.0),
        ),
    }


def hyacinth_candidate_mask(
    ndvi: ArrayLike,
    fai: ArrayLike,
    *,
    ndvi_threshold: float,
    fai_threshold: float,
) -> NDArray[np.bool_]:
    """Create a thresholded floating-vegetation candidate mask.

    Thresholds must be fitted against labeled local observations. This function
    deliberately supplies no default thresholds and does not infer a validated
    hyacinth class from spectral indices alone.
    """
    if not np.isfinite(ndvi_threshold) or not np.isfinite(fai_threshold):
        raise ValueError("NDVI and FAI thresholds must be finite.")
    ndvi_array, fai_array = np.broadcast_arrays(
        _finite_float_array(ndvi, "ndvi"),
        _finite_float_array(fai, "fai"),
    )
    return (
        np.isfinite(ndvi_array)
        & np.isfinite(fai_array)
        & (ndvi_array >= ndvi_threshold)
        & (fai_array >= fai_threshold)
    )
