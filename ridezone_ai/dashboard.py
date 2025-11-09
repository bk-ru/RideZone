"""Helpers for generating JSON payloads consumed by the dashboard UI."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from .pipeline import RideZonePipeline


def _normalize(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()
    if maximum - minimum == 0:
        return pd.Series(0.5, index=series.index)
    return (series - minimum) / (maximum - minimum)


@dataclass
class DashboardDataBuilder:
    """Runs the ML pipeline and converts its output into dashboard-friendly JSON."""

    pipeline: RideZonePipeline | None = None

    def build_payload(self) -> dict[str, Any]:
        pipeline = self.pipeline or RideZonePipeline()
        result = pipeline.run()

        frame = result.processed_frame.copy()
        frame["predicted_trips"] = result.predictions
        frame["demand_score"] = _normalize(frame["predicted_trips"])
        if "micromobility_friendly_score" in frame:
            frame["infrastructure_score"] = frame["micromobility_friendly_score"].clip(0, 1)
        else:
            frame["infrastructure_score"] = 0.5
        if "competition_pressure" in frame:
            frame["competition_penalty"] = _normalize(frame["competition_pressure"])
        else:
            frame["competition_penalty"] = 0.5

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "meta": self._build_meta(result, frame),
            "zones": self._build_zone_entries(frame),
            "recommendations": self._build_recommendations(result.recommendations),
            "heatmap": self._build_heatmap(frame),
            "clusters": self._build_clusters(frame),
            "distribution": self._build_distribution(frame),
            "forecast": self._build_forecast_series(frame),
        }
        return payload

    @staticmethod
    def _build_meta(result, frame: pd.DataFrame) -> dict[str, Any]:
        demand = frame["predicted_trips"]
        high_demand_threshold = float(demand.quantile(0.75))
        high_demand_count = int((demand >= high_demand_threshold).sum())

        meta = {
            "metrics": {
                "mae": result.metrics.mae,
                "rmse": result.metrics.rmse,
                "r2": result.metrics.r2,
            },
            "stats": {
                "total_points": int(len(frame)),
                "high_demand_zones": high_demand_count,
                "recommended_stations": len(result.recommendations),
                "min_predicted": float(demand.min()),
                "max_predicted": float(demand.max()),
            },
            "filters": {
                "districts": sorted(frame["district"].unique().tolist()),
                "zone_types": sorted(frame["zone_type"].unique().tolist()),
            },
        }
        if result.data_health is not None:
            meta["data_health"] = {
                "row_count": result.data_health.row_count,
                "duplicates": result.data_health.duplicates,
            }
        return meta

    @staticmethod
    def _build_zone_entries(frame: pd.DataFrame) -> list[dict[str, Any]]:
        keep_columns = [
            "zone_id",
            "district",
            "zone_type",
            "latitude",
            "longitude",
            "predicted_trips",
            "avg_daily_trips",
            "micromobility_friendly_score",
            "competition_pressure",
            "future_growth_index",
            "distance_to_center_km",
            "bike_infra_km",
            "micro_mobility_lanes_km",
            "transit_stops",
            "population_density",
        ]
        available_cols = [col for col in keep_columns if col in frame.columns]

        zone_entries: list[dict[str, Any]] = []
        for _, row in frame[available_cols].iterrows():
            record = {key: (float(row[key]) if isinstance(row[key], (np.floating, float)) else row[key]) for key in available_cols}
            record["demand_score"] = float(row["predicted_trips"])
            zone_entries.append(record)
        return zone_entries

    @staticmethod
    def _build_recommendations(recommendations) -> list[dict[str, Any]]:
        payload = []
        for rec in recommendations:
            payload.append(
                {
                    "zone_id": rec.zone_id,
                    "district": rec.district,
                    "zone_type": rec.zone_type,
                    "predicted_trips": rec.predicted_trips,
                    "score": rec.composite_score,
                    "confidence": rec.confidence,
                    "rationale": rec.rationale,
                }
            )
        return payload

    @staticmethod
    def _build_heatmap(frame: pd.DataFrame) -> list[list[float]]:
        normalized = _normalize(frame["predicted_trips"])
        return [
            [
                float(row["latitude"]),
                float(row["longitude"]),
                float(normalized.iloc[idx]),
            ]
            for idx, row in frame.iterrows()
        ]

    @staticmethod
    def _build_clusters(frame: pd.DataFrame) -> list[dict[str, Any]]:
        clusters = []
        for district, group in frame.groupby("district"):
            clusters.append(
                {
                    "id": district,
                    "label": district,
                    "center": {
                        "lat": float(group["latitude"].mean()),
                        "lng": float(group["longitude"].mean()),
                    },
                    "radius": float(200 + group["predicted_trips"].mean()),
                    "avgDemand": float(group["predicted_trips"].mean()),
                    "size": int(len(group)),
                }
            )
        return clusters

    @staticmethod
    def _build_distribution(frame: pd.DataFrame) -> dict[str, list[Any]]:
        distribution = frame.groupby("zone_type")["predicted_trips"].sum().sort_values(ascending=False)
        return {
            "labels": distribution.index.tolist(),
            "values": [float(v) for v in distribution.values],
        }

    @staticmethod
    def _build_forecast_series(frame: pd.DataFrame) -> dict[str, list[float]]:
        top = frame.sort_values("predicted_trips", ascending=False).head(5)
        historical = top["avg_daily_trips"] if "avg_daily_trips" in top else top["predicted_trips"] * 0.9
        forecast = top["predicted_trips"] * 1.05
        return {
            "labels": top["zone_id"].tolist(),
            "historical": [float(x) for x in historical],
            "forecast": [float(x) for x in forecast],
        }


def write_dashboard_json(output_path: Path) -> Path:
    """Utility function used by scripts to build dashboard payloads."""

    builder = DashboardDataBuilder()
    payload = builder.build_payload()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
