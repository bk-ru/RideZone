"""Modeling utilities for RideZone AI transfer learning."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FeatureConfig, ModelConfig
from .data_models import ModelMetrics, TrainingArtifacts


@dataclass
class TrainingResult:
    """Artifacts returned after training."""

    metrics: ModelMetrics
    model_path: Path
    predictions: pd.Series
    holdout_predictions: pd.Series
    holdout_actuals: pd.Series
    artifacts: TrainingArtifacts


class TransferRegressor:
    """Wrapper around a scikit-learn pipeline that predicts trip demand on log scale."""

    def __init__(self, feature_config: FeatureConfig, model_config: ModelConfig, model_dir: Path) -> None:
        self.feature_config = feature_config
        self.model_config = model_config
        self.model_dir = model_dir
        self.pipeline: Pipeline | None = None

    def train(self, df: pd.DataFrame) -> TrainingResult:
        X = df[list(self.feature_config.feature_columns)]
        y = np.log1p(df["avg_daily_trips"])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.model_config.test_size,
            random_state=self.model_config.random_state,
        )
        pipeline = self._build_pipeline()
        pipeline.fit(X_train, y_train)

        test_pred_log = pipeline.predict(X_test)
        y_pred = np.expm1(test_pred_log)
        y_true = np.expm1(y_test)
        metrics = self._compute_metrics(y_true, y_pred)

        all_predictions = np.expm1(pipeline.predict(X))
        prediction_series = pd.Series(all_predictions, index=df.index, name="predicted_trips")
        holdout_series = pd.Series(y_pred, index=X_test.index, name="predicted_holdout")
        model_path = self._persist_model(pipeline)
        self.pipeline = pipeline

        artifacts = TrainingArtifacts(
            metrics=metrics,
            model_path=model_path,
            train_rows=len(X_train),
            holdout_rows=len(X_test),
        )

        return TrainingResult(
            metrics=metrics,
            model_path=model_path,
            predictions=prediction_series,
            holdout_predictions=holdout_series,
            holdout_actuals=pd.Series(y_true, index=X_test.index, name="actual_holdout"),
            artifacts=artifacts,
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained yet.")
        preds = self.pipeline.predict(features[list(self.feature_config.feature_columns)])
        return pd.Series(np.expm1(preds), index=features.index, name="predicted_demand")

    def _build_pipeline(self) -> Pipeline:
        numeric_columns = list(self.feature_config.feature_columns)

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_columns),
            ]
        )

        regressor = RandomForestRegressor(
            n_estimators=self.model_config.n_estimators,
            max_depth=self.model_config.max_depth,
            min_samples_split=self.model_config.min_samples_split,
            min_samples_leaf=self.model_config.min_samples_leaf,
            random_state=self.model_config.random_state,
            n_jobs=-1,
        )

        return Pipeline(steps=[("preprocess", transformer), ("model", regressor)])

    @staticmethod
    def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> ModelMetrics:
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(mse**0.5)
        r2 = float(r2_score(y_true, y_pred))
        return ModelMetrics(mae=mae, rmse=rmse, r2=r2)

    def _persist_model(self, pipeline: Pipeline) -> Path:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / "transfer_model.pkl"
        joblib.dump(pipeline, model_path)
        return model_path
