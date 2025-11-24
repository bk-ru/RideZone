from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from ridezone_ai.config import DataPaths, DEFAULT_CONFIG, PipelineConfig
from ridezone_ai.pipeline import RideZonePipeline


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    # Minimal Divvy-like sample to keep the test lightweight.
    sample = pd.DataFrame(
        {
            "ride_id": ["a", "b", "c", "d"],
            "started_at": [
                datetime(2023, 1, 1, 8, 0),
                datetime(2023, 1, 1, 9, 0),
                datetime(2023, 1, 2, 8, 0),
                datetime(2023, 1, 2, 10, 0),
            ],
            "start_lat": [41.88, 41.89, 41.90, 41.92],
            "start_lng": [-87.63, -87.64, -87.62, -87.65],
        }
    )
    chicago_dir = tmp_path / "chicago"
    chicago_dir.mkdir(parents=True, exist_ok=True)
    sample_path = chicago_dir / "divvy_sample.csv"
    sample.to_csv(sample_path, index=False)

    paths = DataPaths(
        chicago_dir=chicago_dir,
        model_dir=tmp_path / "models",
        report_dir=tmp_path / "reports",
        chicago_feature_cache=tmp_path / "chicago_features.parquet",
        moscow_feature_cache=tmp_path / "moscow_features.parquet",
        moscow_predictions=tmp_path / "moscow_predictions.csv",
        boundary_cache_dir=tmp_path / "boundaries",
        poi_cache_dir=tmp_path / "pois",
        transit_cache_dir=tmp_path / "transit",
    )
    features = replace(DEFAULT_CONFIG.features, simulate_osm=True, h3_resolution=7)
    config = PipelineConfig(
        paths=paths,
        features=features,
        osm=DEFAULT_CONFIG.osm,
        model=DEFAULT_CONFIG.model,
        source_city=DEFAULT_CONFIG.source_city,
        target_city=DEFAULT_CONFIG.target_city,
    )

    pipeline = RideZonePipeline(config)
    result = pipeline.run()

    assert result.model_path.exists()
    assert result.moscow_predictions_path.exists()
    assert not result.processed_chicago.empty
    assert not result.processed_moscow.empty
