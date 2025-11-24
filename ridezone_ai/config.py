"""Configuration objects for the transfer learning RideZone AI pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DataPaths:
    """Centralized storage for important project paths."""

    chicago_dir: Path = Path("data")
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")
    chicago_feature_cache: Path = Path("data/processed/chicago_features.parquet")
    moscow_feature_cache: Path = Path("data/processed/moscow_features.parquet")
    moscow_predictions: Path = Path("moscow_predictions.csv")
    boundary_cache_dir: Path = Path("cache/osm_boundaries")
    poi_cache_dir: Path = Path("cache/osm_pois")
    transit_cache_dir: Path = Path("cache/osm_transit")


@dataclass(frozen=True)
class CityDescriptor:
    """Minimal city metadata required for OSM queries and distance calculations."""

    name: str
    boundary_query: str
    center_lat: float
    center_lon: float


@dataclass(frozen=True)
class FeatureConfig:
    """Settings that describe how we build geo features."""

    h3_resolution: int = 8
    poi_radius_m: int = 500
    drop_zero_coords: bool = True
    simulate_osm: bool = False
    chicago_bounds: tuple[tuple[float, float], tuple[float, float]] = ((41.0, 43.5), (-88.5, -86.0))
    moscow_bounds: tuple[tuple[float, float], tuple[float, float]] = ((55.1, 56.1), (36.0, 38.5))
    feature_columns: tuple[str, ...] = ("dist_to_subway_m", "dist_to_center_km", "poi_density_500m")


@dataclass(frozen=True)
class OSMConfig:
    """Tags and behaviors for OSM data extraction."""

    transit_tags: Mapping[str, Any] = field(
        default_factory=lambda: {"railway": ["station", "subway", "light_rail"], "public_transport": "station"}
    )
    poi_tags: Mapping[str, Any] = field(
        default_factory=lambda: {
            "amenity": ["cafe", "restaurant", "bar", "fast_food", "pub", "biergarten", "food_court", "college"],
            "office": True,
            "shop": ["convenience", "supermarket"],
        }
    )
    boundary_timeout: int = 120
    overpass_settings: Mapping[str, Any] = field(default_factory=lambda: {"timeout": 180})


@dataclass(frozen=True)
class ModelConfig:
    """Parameters for the machine-learning estimator."""

    random_state: int = 42
    test_size: float = 0.2
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1


@dataclass(frozen=True)
class PipelineConfig:
    """Root configuration object that bundles all other configs."""

    paths: DataPaths = field(default_factory=DataPaths)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    osm: OSMConfig = field(default_factory=OSMConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    source_city: CityDescriptor = field(
        default_factory=lambda: CityDescriptor(
            name="Chicago",
            boundary_query="Chicago, Illinois, USA",
            center_lat=41.8781,
            center_lon=-87.6298,
        )
    )
    target_city: CityDescriptor = field(
        default_factory=lambda: CityDescriptor(
            name="Moscow",
            boundary_query="Moscow, Russia",
            center_lat=55.751244,
            center_lon=37.618423,
        )
    )


DEFAULT_CONFIG = PipelineConfig()
