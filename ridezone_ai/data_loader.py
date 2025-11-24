"""Data loading utilities for RideZone AI transfer learning."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h3
import pandas as pd

from .config import CityDescriptor, DataPaths, FeatureConfig
from .data_models import DataValidationResult


class ChicagoDivvyLoader:
    """Loads, cleans, and aggregates Chicago trip logs into H3 demand cells."""

    def __init__(self, paths: DataPaths, feature_config: FeatureConfig, source_city: CityDescriptor) -> None:
        self.paths = paths
        self.feature_config = feature_config
        self.source_city = source_city
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self.paths.model_dir.mkdir(parents=True, exist_ok=True)
        self.paths.report_dir.mkdir(parents=True, exist_ok=True)
        self.paths.chicago_feature_cache.parent.mkdir(parents=True, exist_ok=True)

    def load_raw(self) -> pd.DataFrame:
        files = sorted(self.paths.chicago_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No Divvy CSV files found under {self.paths.chicago_dir.resolve()}")

        frames: list[pd.DataFrame] = []
        for path in files:
            frames.append(self._read_csv(path))
        df = pd.concat(frames, ignore_index=True)
        return df

    def _read_csv(self, path: Path) -> pd.DataFrame:
        # We read minimally required columns to keep memory reasonable.
        possible_columns = [
            "ride_id",
            "started_at",
            "start_time",
            "start_lat",
            "start_lng",
            "start_longitude",
            "start_latitude",
        ]
        header = pd.read_csv(path, nrows=0)
        available = [col for col in possible_columns if col in header.columns]
        if not {"start_lat", "start_lng"}.intersection(header.columns) and not {
            "start_latitude",
            "start_longitude",
        }.intersection(header.columns):
            raise ValueError(f"Latitude/longitude columns not found in {path.name}")
        df = pd.read_csv(path, usecols=available)
        return df[available].copy()

    def clean_and_validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, DataValidationResult]:
        lat_col, lon_col = self._resolve_coordinate_columns(df.columns)
        missing = int(df[lat_col].isna().sum() + df[lon_col].isna().sum())
        df = df.dropna(subset=[lat_col, lon_col])

        zero_mask = (df[lat_col] == 0) | (df[lon_col] == 0)
        dropped_zero = int(zero_mask.sum()) if self.feature_config.drop_zero_coords else 0
        if self.feature_config.drop_zero_coords:
            df = df.loc[~zero_mask]

        lat_bounds, lon_bounds = self.feature_config.chicago_bounds
        outlier_mask = ~df[lat_col].between(lat_bounds[0], lat_bounds[1]) | ~df[lon_col].between(
            lon_bounds[0], lon_bounds[1]
        )
        dropped_outliers = int(outlier_mask.sum())
        df = df.loc[~outlier_mask].copy()
        df.rename(columns={lat_col: "lat", lon_col: "lon"}, inplace=True)

        validation = DataValidationResult(
            row_count=len(df),
            dropped_zero_coords=dropped_zero,
            dropped_outliers=dropped_outliers,
            missing_coordinates=missing,
        )
        return df, validation

    def aggregate_to_h3(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["h3_index"] = df.apply(
            lambda row: self._to_h3(row["lat"], row["lon"], self.feature_config.h3_resolution), axis=1
        )

        date_col = self._resolve_date_column(df.columns)
        if date_col:
            df["trip_day"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
            df = df.dropna(subset=["trip_day"])
            grouped = df.groupby(["h3_index", "trip_day"]).size().reset_index(name="daily_trips")
            agg = grouped.groupby("h3_index")["daily_trips"].mean().reset_index(name="avg_daily_trips")
        else:
            agg = df.groupby("h3_index").size().reset_index(name="avg_daily_trips")

        agg["lat"] = agg["h3_index"].apply(lambda idx: self._to_lat_lon(idx)[0])
        agg["lon"] = agg["h3_index"].apply(lambda idx: self._to_lat_lon(idx)[1])
        return agg[["h3_index", "lat", "lon", "avg_daily_trips"]]

    @staticmethod
    def _resolve_coordinate_columns(columns: Iterable[str]) -> tuple[str, str]:
        lat_candidates = ["start_lat", "start_latitude"]
        lon_candidates = ["start_lng", "start_longitude"]
        lat_col = next((col for col in lat_candidates if col in columns), None)
        lon_col = next((col for col in lon_candidates if col in columns), None)
        if lat_col is None or lon_col is None:
            raise ValueError("Could not locate latitude/longitude columns in Divvy feed")
        return lat_col, lon_col

    @staticmethod
    def _resolve_date_column(columns: Iterable[str]) -> str | None:
        for candidate in ("started_at", "start_time"):
            if candidate in columns:
                return candidate
        return None

    @staticmethod
    def _to_h3(lat: float, lon: float, resolution: int) -> str:
        if hasattr(h3, "geo_to_h3"):
            return h3.geo_to_h3(lat, lon, resolution)
        if hasattr(h3, "latlng_to_cell"):
            return h3.latlng_to_cell(lat, lon, resolution)
        raise AttributeError("No suitable H3 conversion function found")

    @staticmethod
    def _to_lat_lon(h3_index: str) -> tuple[float, float]:
        if hasattr(h3, "h3_to_geo"):
            lat, lon = h3.h3_to_geo(h3_index)
            return lat, lon
        if hasattr(h3, "cell_to_latlng"):
            lat, lon = h3.cell_to_latlng(h3_index)
            return lat, lon
        raise AttributeError("No suitable H3 reverse conversion function found")
