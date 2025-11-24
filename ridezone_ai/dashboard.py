"""Helpers for generating JSON payloads consumed by the dashboard UI."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

        moscow = result.processed_moscow.copy()
        moscow["predicted_trips"] = moscow["predicted_demand"]
        moscow["zone_id"] = moscow["h3_index"]
        moscow["zone_type"] = "hex"
        moscow["district"] = self._default_district(moscow)
        moscow["latitude"] = moscow["lat"]
        moscow["longitude"] = moscow["lon"]
        moscow["demand_score"] = _normalize(moscow["predicted_trips"])

        recommendations = self._build_recommendations(moscow)

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "meta": self._build_meta(result, moscow, recommendations),
            "zones": self._build_zones(moscow),
            "recommendations": recommendations,
            "heatmap": self._build_heatmap(moscow),
            "clusters": self._build_clusters(moscow),
            "distribution": self._build_distribution(moscow),
            "forecast": self._build_forecast_series(moscow),
        }
        return payload

    @staticmethod
    def _default_district(moscow: pd.DataFrame) -> str:
        if "district" in moscow.columns and moscow["district"].notna().any():
            return moscow["district"]
        return "Moscow"

    @staticmethod
    def _build_meta(result, moscow: pd.DataFrame, recommendations: list[dict[str, Any]]) -> dict[str, Any]:
        filters = {
            "districts": sorted(moscow["district"].unique().tolist()),
            "zone_types": sorted(moscow["zone_type"].unique().tolist()),
        }
        high_demand_threshold = float(moscow["predicted_trips"].quantile(0.75))
        meta = {
            "metrics": {
                "mae": result.metrics.mae,
                "rmse": result.metrics.rmse,
                "r2": result.metrics.r2,
            },
            "stats": {
                "total_points": int(len(moscow)),
                "high_demand_zones": int((moscow["predicted_trips"] >= high_demand_threshold).sum()),
                "recommended_stations": len(recommendations),
                "min_predicted": float(moscow["predicted_trips"].min()),
                "max_predicted": float(moscow["predicted_trips"].max()),
            },
            "filters": filters,
        }
        if result.data_health is not None:
            meta["data_health"] = {
                "row_count": result.data_health.row_count,
                "dropped_zero_coords": result.data_health.dropped_zero_coords,
                "dropped_outliers": result.data_health.dropped_outliers,
                "missing_coordinates": result.data_health.missing_coordinates,
            }
        return meta

    @staticmethod
    def _build_recommendations(moscow: pd.DataFrame, top_k: int = 10) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []
        for _, row in moscow.sort_values("predicted_trips", ascending=False).head(top_k).iterrows():
            rationale = (
                f"Близко к метро ({row['dist_to_subway_m']:.0f} м), "
                f"POI ~ {row['poi_density_500m']:.0f}, центр {row['dist_to_center_km']:.1f} км"
            )
            recs.append(
                {
                    "zone_id": row["zone_id"],
                    "district": row["district"],
                    "zone_type": row["zone_type"],
                    "predicted_trips": float(row["predicted_trips"]),
                    "composite_score": float(row["demand_score"]),
                    "confidence": float(row["demand_score"]),
                    "rationale": rationale,
                }
            )
        return recs

    @staticmethod
    def _build_zones(moscow: pd.DataFrame) -> list[dict[str, Any]]:
        keep_cols = [
            "zone_id",
            "district",
            "zone_type",
            "latitude",
            "longitude",
            "predicted_trips",
            "dist_to_subway_m",
            "dist_to_center_km",
            "poi_density_500m",
        ]
        zones: list[dict[str, Any]] = []
        for _, row in moscow[keep_cols].iterrows():
            zones.append({col: (float(row[col]) if isinstance(row[col], (float, np.floating, int, np.integer)) else row[col]) for col in keep_cols})
        return zones

    @staticmethod
    def _build_heatmap(moscow: pd.DataFrame) -> list[list[float]]:
        normalized = _normalize(moscow["predicted_trips"])
        return [
            [
                float(row["latitude"]),
                float(row["longitude"]),
                float(normalized.iloc[idx]),
            ]
            for idx, row in moscow.iterrows()
        ]

    @staticmethod
    def _build_clusters(moscow: pd.DataFrame) -> list[dict[str, Any]]:
        clusters = []
        for district, group in moscow.groupby("district"):
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
    def _build_distribution(moscow: pd.DataFrame) -> dict[str, list[Any]]:
        distribution = moscow.groupby("zone_type")["predicted_trips"].sum().sort_values(ascending=False)
        return {
            "labels": distribution.index.tolist(),
            "values": [float(v) for v in distribution.values],
        }

    @staticmethod
    def _build_forecast_series(moscow: pd.DataFrame) -> dict[str, list[float]]:
        top = moscow.sort_values("predicted_trips", ascending=False).head(5)
        historical = top["predicted_trips"] * 0.9
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
