"""Top-level orchestration for the RideZone AI workflow."""
from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_CONFIG, PipelineConfig
from .data_loader import ZoneDataLoader
from .data_models import DataValidationResult, PipelineResult
from .feature_engineering import FeatureEngineer
from .modeling import DemandRegressor
from .optimizer import StationPlacementOptimizer
from .reporting import ReportGenerator


class RideZonePipeline:
    """Coordinates data preparation, modeling, and optimization."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.loader = ZoneDataLoader(self.config.paths, self.config.features)
        self.feature_engineer = FeatureEngineer(self.config.features, self.config.paths.feature_cache)
        self.model = DemandRegressor(self.config.features, self.config.model, self.config.paths.model_dir)
        self.optimizer = StationPlacementOptimizer(self.config.recommendations)
        self.reporter = ReportGenerator(self.config.paths)
        self._data_health: DataValidationResult | None = None

    @property
    def data_health(self) -> DataValidationResult | None:
        return self._data_health

    def run(self, persist_report: bool = True) -> PipelineResult:
        raw_df = self.loader.load()
        self._data_health = self.loader.validate(raw_df)
        engineered = self.feature_engineer.transform(raw_df)
        features, target = self.loader.prepare_supervised(engineered.dataframe)
        training_result = self.model.train(features, target)
        recommendations = self.optimizer.recommend(engineered.dataframe, training_result.predictions)
        report_path = None
        if persist_report:
            report_path = self.reporter.create_report(training_result.metrics, recommendations)

        return PipelineResult(
            metrics=training_result.metrics,
            recommendations=recommendations,
            model_path=training_result.model_path,
            report_path=report_path,
            processed_frame=engineered.dataframe,
            predictions=training_result.predictions,
            data_health=self._data_health,
        )

