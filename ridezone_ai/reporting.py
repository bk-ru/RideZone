"""Reporting utilities for RideZone AI transfer learning."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import pandas as pd

from .config import DataPaths
from .data_models import DataValidationResult, ModelMetrics


class ReportGenerator:
    """Builds lightweight markdown reports for stakeholders."""

    def __init__(self, paths: DataPaths) -> None:
        self.paths = paths
        self.paths.report_dir.mkdir(parents=True, exist_ok=True)

    def create_report(
        self,
        metrics: ModelMetrics,
        predictions: pd.DataFrame,
        data_health: DataValidationResult | None,
        source_city: str,
        target_city: str,
        top_k: int = 15,
    ) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        report_path = self.paths.report_dir / "latest_report.md"

        header = dedent(
            f"""
            # RideZone AI transfer report ({timestamp})
            Мы используем transfer learning: модель обучена на паттернах {source_city}, потому что для {target_city} нет исторических поездок.
            """
        ).strip()

        lines = [
            header,
            "",
            "## Model quality (Chicago holdout)",
            f"- MAE: {metrics.mae:.2f}",
            f"- RMSE: {metrics.rmse:.2f}",
            f"- R2: {metrics.r2:.3f}",
            "",
            "## Data health (Chicago Divvy)",
        ]

        if data_health is None:
            lines.append("- No validation stats available.")
        else:
            lines.extend(
                [
                    f"- Rows after cleaning: {data_health.row_count}",
                    f"- Dropped zero coords: {data_health.dropped_zero_coords}",
                    f"- Dropped outliers: {data_health.dropped_outliers}",
                    f"- Missing coordinates: {data_health.missing_coordinates}",
                ]
            )

        lines.append("")
        lines.append(f"## Top {top_k} Moscow hexagons by predicted demand")
        lines.append("| Rank | H3 | Pred trips/day | dist_to_subway_m | poi_density_500m | dist_to_center_km |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        top = predictions.sort_values("predicted_demand", ascending=False).head(top_k)
        for idx, row in enumerate(top.itertuples(), start=1):
            lines.append(
                f"| {idx} | {row.h3_index} | {row.predicted_demand:.1f} | "
                f"{row.dist_to_subway_m:.0f} | {row.poi_density_500m:.0f} | {row.dist_to_center_km:.2f} |"
            )

        content = "\n".join(lines) + "\n"
        report_path.write_text(content, encoding="utf-8")
        return report_path
