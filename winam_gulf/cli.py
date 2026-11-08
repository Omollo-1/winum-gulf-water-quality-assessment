"""Command-line entry points for the offline science-extension analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .corrections import compare_correction_methods
from .drivers import train_driver_attribution
from .field_validation import evaluate_field_calibration
from .flowpath import analyze_flowpath_distance
from .forecast import backtest_seasonal_forecast


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _write_flowpath_plot(data: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(data["distance_along_flow_m"] / 1000, data["bloom_frequency"], alpha=0.8)
    axis.set_title("Bloom-frequency proxy vs. water-flow-path distance")
    axis.set_xlabel("Distance along flow path (km)")
    axis.set_ylabel("Bloom-frequency proxy")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="analysis", required=True)
    for name in ("corrections", "field-validation", "forecast", "drivers", "flowpath"):
        command = subparsers.add_parser(name)
        command.add_argument("--input", type=Path, required=True, help="Input CSV file.")
        command.add_argument("--output-dir", type=Path, required=True, help="Directory for result files.")
    subparsers.choices["corrections"].add_argument("--baseline", default="gee_s2_sr")
    subparsers.choices["forecast"].add_argument("--target", default="NDCI_chla")
    subparsers.choices["forecast"].add_argument("--train-end", default="2022-12-01")
    subparsers.choices["forecast"].add_argument("--holdout-start", default="2023-01-01")
    subparsers.choices["forecast"].add_argument("--holdout-end", default="2024-12-01")
    subparsers.choices["drivers"].add_argument("--target", default="bloom_metric")
    subparsers.choices["drivers"].add_argument("--shap", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)

    if args.analysis == "corrections":
        result = compare_correction_methods(data, baseline_method=args.baseline)
        result.summary.to_csv(args.output_dir / "correction_comparison_metrics.csv", index=False)
        result.paired_months.to_csv(args.output_dir / "correction_comparison_paired_months.csv", index=False)
        figure, axis = plt.subplots(figsize=(10, 5))
        for method, group in data.groupby("method"):
            monthly = group.assign(month=pd.to_datetime(group["month"]).dt.to_period("M").dt.to_timestamp())
            monthly.groupby("month")["ndci"].median().plot(ax=axis, marker="o", label=method)
        axis.set_title("Paired monthly NDCI by atmospheric-correction method")
        axis.set_ylabel("NDCI (unitless)")
        axis.set_xlabel("Month")
        axis.legend()
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(args.output_dir / "correction_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
    elif args.analysis == "field-validation":
        result = evaluate_field_calibration(data)
        _write_json(args.output_dir / "field_calibration_metrics.json", result.metrics)
        result.predictions.to_csv(args.output_dir / "field_calibration_predictions.csv", index=False)
    elif args.analysis == "forecast":
        result = backtest_seasonal_forecast(
            data,
            target_column=args.target,
            train_end=args.train_end,
            holdout_start=args.holdout_start,
            holdout_end=args.holdout_end,
            output_plot=str(args.output_dir / "forecast_backtest.png"),
        )
        result.metrics.to_csv(args.output_dir / "forecast_backtest_metrics.csv", index=False)
        result.predictions.to_csv(args.output_dir / "forecast_backtest_predictions.csv")
    elif args.analysis == "drivers":
        result = train_driver_attribution(data, target_column=args.target, calculate_shap=args.shap)
        _write_json(args.output_dir / "driver_attribution_metrics.json", result.metrics)
        result.permutation_importances.to_csv(
            args.output_dir / "driver_permutation_importance.csv", index=False
        )
        result.predictions.to_csv(args.output_dir / "driver_holdout_predictions.csv")
    elif args.analysis == "flowpath":
        result = analyze_flowpath_distance(data)
        _write_json(args.output_dir / "flowpath_metrics.json", result)
        _write_flowpath_plot(data, args.output_dir / "flowpath_distance.png")
    print(f"Results saved under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
