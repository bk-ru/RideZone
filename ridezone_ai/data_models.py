"""Dataclasses used across the RideZone AI project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class DataValidationResult:
    """Information about dataset health."""

    row_count: int
    missing_values: dict[str, int]
    duplicates: int


@dataclass(frozen=True)
class ZoneRecord:
    """Container that represents one zone entry from the dataset."""

    zone_id: str
    district: str
    zone_type: str
    latitude: float
    longitude: float
    population_density: float
    daytime_population: float
    avg_income: float
    bike_infra_km: float
    micro_mobility_lanes_km: float
    transit_stops: int
    competitor_density: float
    tourist_index: float
    weather_penalty: float
    event_count: int
    existing_parking_spots: int
    safety_incidents: int
    current_station_count: int
    future_growth_index: float
    micromobility_friendly_score: float
    suitability_score: float
    avg_daily_trips: float

    @classmethod
    def from_series(cls, row: pd.Series) -> "ZoneRecord":
        payload = {field.name: row[field.name] for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**payload)


@dataclass(frozen=True)
class ModelMetrics:
    """Evaluation metrics for the forecasting model."""

    mae: float
    rmse: float
    r2: float


@dataclass(frozen=True)
class Recommendation:
    """Recommended zone for deploying a new station."""

    zone_id: str
    district: str
    zone_type: str
    predicted_trips: float
    composite_score: float
    confidence: float
    rationale: str


@dataclass(frozen=True)
class PipelineResult:
    """Snapshot of the complete pipeline execution."""

    metrics: ModelMetrics
    recommendations: list[Recommendation]
    model_path: Path
    report_path: Path | None
    processed_frame: pd.DataFrame
    predictions: pd.Series
    data_health: DataValidationResult | None = None


def to_zone_records(df: pd.DataFrame) -> Iterable[ZoneRecord]:
    """Helper that converts a dataframe into zone records."""

    for _, row in df.iterrows():
        yield ZoneRecord.from_series(row)
