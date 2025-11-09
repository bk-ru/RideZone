"""Optimization logic for selecting best deployment zones."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RecommendationConfig
from .data_models import Recommendation


class StationPlacementOptimizer:
    """Ranks candidate zones using multi-factor scoring."""

    def __init__(self, config: RecommendationConfig) -> None:
        self.config = config

    def recommend(self, df: pd.DataFrame, predictions: pd.Series) -> list[Recommendation]:
        if predictions.isna().any():
            raise ValueError("Predictions contain NaNs; aborting recommendation generation")

        enriched = df.copy()
        enriched["predicted_trips"] = predictions
        enriched["demand_score"] = self._normalize(enriched["predicted_trips"])
        enriched["infrastructure_score"] = enriched["micromobility_friendly_score"].clip(0, 1)
        enriched["growth_score"] = enriched["future_growth_index"].clip(0, 1)
        enriched["competition_penalty"] = self._normalize(enriched["competition_pressure"], invert=True)

        enriched["composite_score"] = (
            self.config.demand_weight * enriched["demand_score"]
            + self.config.infrastructure_weight * enriched["infrastructure_score"]
            + self.config.growth_weight * enriched["growth_score"]
            - self.config.competition_penalty * (1 - enriched["competition_penalty"])
        )

        enriched["confidence"] = (
            0.6 * enriched["demand_score"]
            + 0.25 * enriched["infrastructure_score"]
            + 0.15 * enriched["competition_penalty"]
        )

        enriched = enriched.sort_values("composite_score", ascending=False)

        recommendations: list[Recommendation] = []
        for _, row in enriched.head(self.config.top_k).iterrows():
            confidence = float(row["confidence"])
            if confidence < self.config.min_confidence:
                continue
            rationale = self._build_rationale(row)
            recommendations.append(
                Recommendation(
                    zone_id=row["zone_id"],
                    district=row["district"],
                    zone_type=row["zone_type"],
                    predicted_trips=float(row["predicted_trips"]),
                    composite_score=float(row["composite_score"]),
                    confidence=confidence,
                    rationale=rationale,
                )
            )
        return recommendations

    @staticmethod
    def _normalize(series: pd.Series, invert: bool = False) -> pd.Series:
        minimum = series.min()
        maximum = series.max()
        if maximum - minimum == 0:
            normalized = pd.Series(0.5, index=series.index)
        else:
            normalized = (series - minimum) / (maximum - minimum)
        if invert:
            return 1 - normalized
        return normalized.clip(0, 1)

    @staticmethod
    def _build_rationale(row: pd.Series) -> str:
        parts = [
            f"Demand~{row['predicted_trips']:.0f} trips/day",
            f"infra score={row['infrastructure_score']:.2f}",
            f"competition penalty={1 - row['competition_penalty']:.2f}",
        ]
        if row.get("growth_score") is not None:
            parts.append(f"growth={row['growth_score']:.2f}")
        return "; ".join(parts)

