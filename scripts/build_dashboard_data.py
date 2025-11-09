"""Build JSON payload consumed by the interactive dashboard."""
from __future__ import annotations

import argparse
from pathlib import Path

from ridezone_ai.dashboard import write_dashboard_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dashboard_data.json from the RideZone pipeline")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("web/dashboard_data.json"),
        help="Destination file for the JSON payload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = write_dashboard_json(args.output)
    print(f"Dashboard data written to {output_path}")


if __name__ == "__main__":
    main()
