from __future__ import annotations

from pathlib import Path

from ridezone_ai.config import DataPaths, PipelineConfig, DEFAULT_CONFIG
from ridezone_ai.pipeline import RideZonePipeline


def test_pipeline_end_to_end(tmp_path):
    paths = DataPaths(
        raw_data=DEFAULT_CONFIG.paths.raw_data,
        model_dir=tmp_path / "models",
        report_dir=tmp_path / "reports",
        feature_cache=tmp_path / "features.parquet",
    )
    config = PipelineConfig(
        paths=paths,
        features=DEFAULT_CONFIG.features,
        model=DEFAULT_CONFIG.model,
        recommendations=DEFAULT_CONFIG.recommendations,
    )

    pipeline = RideZonePipeline(config)
    result = pipeline.run()

    assert result.metrics.mae > 0
    assert result.metrics.rmse > 0
    assert result.recommendations, "Expected at least one recommended zone"
    assert result.model_path.exists()
    assert result.report_path is not None and Path(result.report_path).exists()
    assert not result.processed_frame.empty
    assert len(result.predictions) == len(result.processed_frame)
