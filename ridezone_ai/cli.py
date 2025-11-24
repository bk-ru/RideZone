"""Console entry point for RideZone AI transfer learning."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_CONFIG, PipelineConfig
from .pipeline import RideZonePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RideZone AI - transfer learning from Chicago Divvy to Moscow micromobility demand"
    )
    parser.add_argument("--chicago-dir", type=Path, default=None, help="Directory with Chicago Divvy CSV files")
    parser.add_argument("--output", type=Path, default=None, help="Path to save Moscow predictions CSV")
    parser.add_argument("--simulate-osm", action="store_true", help="Use synthetic OSM-like features (offline)")
    parser.add_argument("--resolution", type=int, default=None, help="H3 resolution for grid generation")
    parser.add_argument("--poi-radius", type=int, default=None, help="Radius in meters for POI density")
    parser.add_argument("--no-report", action="store_true", help="Skip writing the markdown report")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config: PipelineConfig = DEFAULT_CONFIG
    if args.chicago_dir is not None or args.output is not None or args.simulate_osm or args.resolution or args.poi_radius:
        paths = config.paths
        features = config.features
        if args.chicago_dir is not None:
            paths = replace(paths, chicago_dir=args.chicago_dir)
        if args.output is not None:
            paths = replace(paths, moscow_predictions=args.output)
        if args.simulate_osm:
            features = replace(features, simulate_osm=True)
        if args.resolution:
            features = replace(features, h3_resolution=args.resolution)
        if args.poi_radius:
            features = replace(features, poi_radius_m=args.poi_radius)
        config = replace(config, paths=paths, features=features)

    pipeline = RideZonePipeline(config)
    result = pipeline.run(persist_report=not args.no_report)

    print(
        "Transfer model trained on Chicago -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}".format(
            mae=result.metrics.mae, rmse=result.metrics.rmse, r2=result.metrics.r2
        )
    )
    if pipeline.data_health is not None:
        dh = pipeline.data_health
        print(
            f"Rows after cleaning: {dh.row_count} | dropped zeros: {dh.dropped_zero_coords} | "
            f"dropped outliers: {dh.dropped_outliers} | missing coords: {dh.missing_coordinates}"
        )

    print(f"Moscow predictions saved to: {result.moscow_predictions_path}")
    print(f"Model artifact stored at: {result.model_path}")
    if result.report_path is not None:
        print(f"Markdown report stored at: {result.report_path}")


if __name__ == "__main__":
    main()
