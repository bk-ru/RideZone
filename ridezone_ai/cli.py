"""Console entry point for RideZone AI."""
from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Sequence

from .config import DEFAULT_CONFIG, PipelineConfig
from .pipeline import RideZonePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RideZone AI - micromobility station optimizer")
    parser.add_argument("--top-k", type=int, default=None, help="How many zones to recommend")
    parser.add_argument(
        "--min-confidence", type=float, default=None, help="Minimum confidence to keep a recommendation"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing the markdown report (useful for quick experiments)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config: PipelineConfig = DEFAULT_CONFIG
    if args.top_k is not None or args.min_confidence is not None:
        rec_cfg = config.recommendations
        if args.top_k is not None:
            rec_cfg = replace(rec_cfg, top_k=args.top_k)
        if args.min_confidence is not None:
            rec_cfg = replace(rec_cfg, min_confidence=args.min_confidence)
        config = replace(config, recommendations=rec_cfg)

    pipeline = RideZonePipeline(config)
    result = pipeline.run(persist_report=not args.no_report)

    print("Model metrics -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}".format(
        mae=result.metrics.mae, rmse=result.metrics.rmse, r2=result.metrics.r2
    ))
    if pipeline.data_health is not None:
        print(f"Rows processed: {pipeline.data_health.row_count}")
        if pipeline.data_health.duplicates:
            print(f"Warning: {pipeline.data_health.duplicates} duplicate zone IDs detected")

    if not result.recommendations:
        print("No high-confidence recommendations were produced.")
        return

    print("\nTop zones:")
    for idx, rec in enumerate(result.recommendations, start=1):
        print(
            f"#{idx:02d} {rec.zone_id} ({rec.district}, {rec.zone_type}) -> "
            f"{rec.predicted_trips:.1f} trips/day | score {rec.composite_score:.3f} | "
            f"confidence {rec.confidence:.2f}"
        )
        print(f"    {rec.rationale}")

    if result.report_path is None:
        print("Report generation skipped.")
    else:
        print(f"\nReport saved to: {result.report_path}")
    print(f"Model artifact stored at: {result.model_path}")


if __name__ == "__main__":
    main()
