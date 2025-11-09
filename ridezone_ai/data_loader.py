"""Data loading utilities for RideZone AI."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import DataPaths, FeatureConfig
from .data_models import DataValidationResult


class ZoneDataLoader:
    """Loads and validates datasets used in the pipeline."""

    def __init__(self, paths: DataPaths, feature_config: FeatureConfig) -> None:
        self.paths = paths
        self.feature_config = feature_config
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self.paths.model_dir.mkdir(parents=True, exist_ok=True)
        self.paths.report_dir.mkdir(parents=True, exist_ok=True)
        self.paths.feature_cache.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        if not self.paths.raw_data.exists():
            raise FileNotFoundError(f"Dataset not found at {self.paths.raw_data.resolve()}")
        df = pd.read_csv(self.paths.raw_data)
        return df

    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        missing = {column: int(df[column].isna().sum()) for column in df.columns}
        duplicates = int(df.duplicated(subset="zone_id").sum())
        return DataValidationResult(row_count=len(df), missing_values=missing, duplicates=duplicates)

    def prepare_supervised(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Return feature matrix and target vector using the feature config."""

        target = df[self.feature_config.target_column]
        feature_columns = list(self.feature_config.numeric_features)
        feature_columns += list(self.feature_config.derived_numeric_features)
        feature_columns += list(self.feature_config.categorical_features)
        feature_columns += [self.feature_config.latitude_column, self.feature_config.longitude_column]
        X = df[feature_columns].copy()
        return X, target

