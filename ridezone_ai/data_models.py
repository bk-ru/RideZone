"""Dataclasses used across the transfer learning RideZone AI project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataValidationResult:
    """Information about dataset health after cleaning."""

    row_count: int
    dropped_zero_coords: int
    dropped_outliers: int
    missing_coordinates: int


@dataclass(frozen=True)
class ModelMetrics:
    """Evaluation metrics for the forecasting model."""

    mae: float
    rmse: float
    r2: float


@dataclass(frozen=True)
class TrainingArtifacts:
    """Container for model training outputs."""

    metrics: ModelMetrics
    model_path: Path
    train_rows: int
    holdout_rows: int


@dataclass(frozen=True)
class PipelineResult:
    """Snapshot of the complete pipeline execution."""

    metrics: ModelMetrics
    model_path: Path
    moscow_predictions_path: Path
    report_path: Path | None
    processed_chicago: pd.DataFrame
    processed_moscow: pd.DataFrame
    data_health: DataValidationResult | None = None


@dataclass(frozen=True)
class Recommendation:
    """Legacy recommendation placeholder kept for compatibility with reporting utilities."""

    zone_id: str = ""
    district: str = ""
    zone_type: str = ""
    predicted_trips: float = 0.0
    composite_score: float = 0.0
    confidence: float = 0.0
    rationale: str = ""
