"""Top-level orchestration for the RideZone AI transfer-learning workflow."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DEFAULT_CONFIG, PipelineConfig
from .data_loader import ChicagoDivvyLoader
from .data_models import PipelineResult
from .feature_engineering import OSMFeatureEngineer
from .modeling import TransferRegressor
from .reporting import ReportGenerator


class RideZonePipeline:
    """Coordinates data preparation, modeling, and cross-city transfer."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.loader = ChicagoDivvyLoader(self.config.paths, self.config.features, self.config.source_city)
        self.feature_engineer = OSMFeatureEngineer(self.config.features, self.config.osm, self.config.paths)
        self.model = TransferRegressor(self.config.features, self.config.model, self.config.paths.model_dir)
        self._data_health = None
        self.reporter = ReportGenerator(self.config.paths)

    @property
    def data_health(self):
        return self._data_health

    def run(self, persist_report: bool = True) -> PipelineResult:
        # Transfer learning: we train on Chicago patterns because Moscow lacks historical trip logs.
        raw_df = self.loader.load_raw()
        cleaned_df, validation = self.loader.clean_and_validate(raw_df)
        self._data_health = validation
        chicago_agg = self.loader.aggregate_to_h3(cleaned_df)
        chicago_features = self.feature_engineer.build_features(chicago_agg, self.config.source_city)
        self._persist_features(chicago_features, self.config.paths.chicago_feature_cache)

        training_result = self.model.train(chicago_features)

        moscow_grid = self.feature_engineer.build_h3_grid(self.config.target_city)
        moscow_features = self.feature_engineer.build_features(moscow_grid, self.config.target_city)
        predictions = self.model.predict(moscow_features)
        moscow_features["predicted_demand"] = predictions
        self._persist_features(moscow_features, self.config.paths.moscow_feature_cache)
        predictions_path = self._persist_predictions(moscow_features)
        report_path = None
        if persist_report:
            report_path = self.reporter.create_report(
                metrics=training_result.metrics,
                predictions=moscow_features,
                data_health=self._data_health,
                source_city=self.config.source_city.name,
                target_city=self.config.target_city.name,
            )

        return PipelineResult(
            metrics=training_result.metrics,
            model_path=training_result.model_path,
            moscow_predictions_path=predictions_path,
            report_path=report_path,
            processed_chicago=chicago_features,
            processed_moscow=moscow_features,
            data_health=self._data_health,
        )

    def _persist_predictions(self, df: pd.DataFrame) -> Path:
        self.config.paths.moscow_predictions.parent.mkdir(parents=True, exist_ok=True)
        output = df[["h3_index", "lat", "lon", "predicted_demand"]].copy()
        output.to_csv(self.config.paths.moscow_predictions, index=False)
        return self.config.paths.moscow_predictions

    @staticmethod
    def _persist_features(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
