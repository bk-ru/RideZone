"""Generate a RideZone-ready dataset for Moscow using OpenStreetMap data."""
from __future__ import annotations

import argparse
from pathlib import Path

from ridezone_ai.osm_moscow_builder import MoscowDatasetConfig, write_moscow_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Moscow dataset from OSM")
    parser.add_argument("--resolution", type=int, default=8, help="H3 resolution (default: 8)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/moscow_hex_zones.csv"),
        help="Path to the CSV file that will be created",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MoscowDatasetConfig(resolution=args.resolution, output=args.output)
    output_path = write_moscow_dataset(cfg)
    print(f"Dataset written to {output_path}")


if __name__ == "__main__":
    main()
