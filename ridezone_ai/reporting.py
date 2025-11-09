"""Reporting utilities for RideZone AI."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

from .config import DataPaths
from .data_models import ModelMetrics, Recommendation


class ReportGenerator:
    """Builds lightweight markdown reports for stakeholders."""

    def __init__(self, paths: DataPaths) -> None:
        self.paths = paths
        self.paths.report_dir.mkdir(parents=True, exist_ok=True)

    def create_report(self, metrics: ModelMetrics, recommendations: list[Recommendation]) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        report_path = self.paths.report_dir / "latest_report.md"
        lines = [
            f"# RideZone AI report ({timestamp})",
            "",
            "## Model quality",
            f"- MAE: {metrics.mae:.2f}",
            f"- RMSE: {metrics.rmse:.2f}",
            f"- R²: {metrics.r2:.3f}",
            "",
            "## Top recommendations",
        ]

        if not recommendations:
            lines.append("No recommendations cleared the minimum confidence threshold.")
        else:
            header = "| Rank | Zone | District | Type | Pred trips | Score | Confidence | Rationale |"
            divider = "| --- | --- | --- | --- | --- | --- | --- | --- |"
            rows = []
            for idx, rec in enumerate(recommendations, start=1):
                rows.append(
                    f"| {idx} | {rec.zone_id} | {rec.district} | {rec.zone_type} | "
                    f"{rec.predicted_trips:.1f} | {rec.composite_score:.3f} | {rec.confidence:.2f} | {rec.rationale} |"
                )
            lines.extend([header, divider, *rows])

        content = "\n".join(lines) + "\n"
        report_path.write_text(content, encoding="utf-8")
        return report_path

