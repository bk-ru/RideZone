"""Configuration objects for the RideZone AI pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class DataPaths:
    """Centralized storage for important project paths."""

    raw_data: Path = Path("data/raw/moscow_hex_zones.csv")
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")
    feature_cache: Path = Path("data/processed/features.parquet")


@dataclass(frozen=True)
class FeatureConfig:
    """Settings that describe which features are allowed and how to treat them."""

    numeric_features: Sequence[str] = (
        "population_density",
        "daytime_population",
        "avg_income",
        "bike_infra_km",
        "micro_mobility_lanes_km",
        "transit_stops",
        "competitor_density",
        "tourist_index",
        "weather_penalty",
        "event_count",
        "existing_parking_spots",
        "safety_incidents",
        "current_station_count",
        "future_growth_index",
        "micromobility_friendly_score",
        "suitability_score",
    )
    derived_numeric_features: Sequence[str] = (
        "distance_to_center_km",
        "bike_per_stop",
        "micro_lane_ratio",
        "growth_pressure",
        "competition_pressure",
        "available_parking_ratio",
        "weather_resilience",
        "safety_penalty",
    )
    categorical_features: Sequence[str] = ("district", "zone_type")
    target_column: str = "avg_daily_trips"
    latitude_column: str = "latitude"
    longitude_column: str = "longitude"


@dataclass(frozen=True)
class ModelConfig:
    """Parameters for the machine-learning estimator."""

    random_state: int = 42
    test_size: float = 0.2
    n_estimators: int = 250
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1


@dataclass(frozen=True)
class RecommendationConfig:
    """Settings that control how many zones are recommended and scoring weights."""

    top_k: int = 10
    min_confidence: float = 0.15
    demand_weight: float = 0.55
    infrastructure_weight: float = 0.25
    competition_penalty: float = 0.15
    growth_weight: float = 0.20


@dataclass(frozen=True)
class PipelineConfig:
    """Root configuration object that bundles all other configs."""

    paths: DataPaths = field(default_factory=DataPaths)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    recommendations: RecommendationConfig = field(default_factory=RecommendationConfig)


DEFAULT_CONFIG = PipelineConfig()
