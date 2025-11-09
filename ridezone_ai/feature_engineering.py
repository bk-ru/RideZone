"""Feature engineering module."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import FeatureConfig


@dataclass
class FeatureEngineeringResult:
    """Return payload for engineered features."""

    dataframe: pd.DataFrame
    cache_path: Path | None = None


class FeatureEngineer:
    """Adds domain-specific features that raise model quality."""

    MOSCOW_CENTER = (55.751244, 37.618423)

    def __init__(self, config: FeatureConfig, cache_path: Path) -> None:
        self.config = config
        self.cache_path = cache_path

    def transform(self, df: pd.DataFrame, persist: bool = True) -> FeatureEngineeringResult:
        engineered = df.copy()
        engineered = self._add_distance_to_center(engineered)
        engineered = self._add_infrastructure_ratios(engineered)
        engineered = self._add_competition_features(engineered)
        engineered = self._add_resilience_features(engineered)

        cache_path = None
        if persist:
            cache_path = self._persist(engineered)
        return FeatureEngineeringResult(engineered, cache_path)

    def _persist(self, df: pd.DataFrame) -> Path:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.cache_path, index=False)
        return self.cache_path

    def _add_distance_to_center(self, df: pd.DataFrame) -> pd.DataFrame:
        lat0, lon0 = self.MOSCOW_CENTER

        def haversine(row: pd.Series) -> float:
            return self._haversine_km(lat0, lon0, row[self.config.latitude_column], row[self.config.longitude_column])

        df["distance_to_center_km"] = df.apply(haversine, axis=1)
        return df

    def _add_infrastructure_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bike_per_stop"] = (df["bike_infra_km"] + 1e-3) / (df["transit_stops"] + 1)
        df["micro_lane_ratio"] = (df["micro_mobility_lanes_km"] + 1e-3) / (df["bike_infra_km"] + 1)
        df["growth_pressure"] = df["future_growth_index"] * df["population_density"]
        return df

    def _add_competition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["competition_pressure"] = df["competitor_density"] * (1 + 0.2 * df["current_station_count"])
        df["available_parking_ratio"] = df["existing_parking_spots"] / (df["population_density"] + 1)
        return df

    def _add_resilience_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["weather_resilience"] = (1 - df["weather_penalty"]) * df["micromobility_friendly_score"]
        df["safety_penalty"] = df["safety_incidents"] / (df["transit_stops"] + 1)
        return df

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius = 6371.0
        lat1_rad, lon1_rad = map(math.radians, (lat1, lon1))
        lat2_rad, lon2_rad = map(math.radians, (lat2, lon2))
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return radius * c

