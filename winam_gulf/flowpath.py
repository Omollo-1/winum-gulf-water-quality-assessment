"""Association analysis using supplied water-network flow-path distances."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def analyze_flowpath_distance(
    observations: pd.DataFrame,
    *,
    distance_column: str = "distance_along_flow_m",
    response_column: str = "bloom_frequency",
) -> dict[str, float | int | str]:
    """Quantify monotonic and linear association along user-supplied flow paths.

    This function does not infer hydrodynamic routes from straight-line
    distances. The input distance must be independently measured along the
    documented water-flow network.
    """
    required = {distance_column, response_column}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Missing flow-path columns: {sorted(missing)}")
    data = observations.loc[:, [distance_column, response_column]].copy()
    data[distance_column] = pd.to_numeric(data[distance_column], errors="raise")
    data[response_column] = pd.to_numeric(data[response_column], errors="raise")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3:
        raise ValueError("At least three valid flow-path observations are required.")
    if (data[distance_column] < 0).any():
        raise ValueError("Flow-path distances must be non-negative.")
    if data[distance_column].nunique() < 2 or data[response_column].nunique() < 2:
        raise ValueError("Distance and response must each contain at least two distinct values.")

    slope, intercept, r_value, p_value, standard_error = stats.linregress(
        data[distance_column].to_numpy(dtype=float) / 1000.0,
        data[response_column].to_numpy(dtype=float),
    )
    spearman = stats.spearmanr(data[distance_column], data[response_column])
    kendall = stats.kendalltau(data[distance_column], data[response_column])
    return {
        "n_observations": int(len(data)),
        "distance_basis": "caller-supplied distance along water-flow path",
        "response_column": response_column,
        "linear_slope_response_per_km": float(slope),
        "linear_intercept": float(intercept),
        "linear_r_squared": float(r_value**2),
        "linear_p_value": float(p_value),
        "linear_slope_standard_error": float(standard_error),
        "spearman_rho": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
        "kendall_tau": float(kendall.statistic),
        "kendall_p_value": float(kendall.pvalue),
        "causal_interpretation": "not established",
    }
