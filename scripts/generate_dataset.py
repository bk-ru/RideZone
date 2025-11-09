"""Generate a synthetic dataset for RideZone AI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DatasetConfig:
    samples: int = 150
    seed: int = 42
    outfile: Path = Path("data/raw/sample_zones.csv")


def simulate_row(rng: np.random.Generator, zone_index: int, districts: list[str], zone_types: list[str]) -> dict:
    district = rng.choice(districts)
    zone_type = rng.choice(zone_types)
    lat = 55.70 + rng.normal(0, 0.04)
    lon = 37.60 + rng.normal(0, 0.07)
    population_density = np.clip(rng.normal(12500, 2500), 3000, 22000)
    daytime_population = population_density * rng.uniform(0.7, 1.4)
    avg_income = np.clip(rng.normal(780, 130), 420, 1200)
    bike_infra_km = np.clip(rng.normal(2.3, 0.8), 0.2, 5.5)
    micro_mobility_lanes_km = np.clip(bike_infra_km * rng.uniform(0.5, 1.2), 0.1, 5.0)
    transit_stops = int(np.clip(rng.normal(22, 8), 3, 55))
    competitor_density = np.clip(rng.normal(1.5, 0.6), 0.1, 4.5)
    tourist_index = np.clip(rng.beta(2, 3), 0.05, 0.95)
    weather_penalty = np.clip(rng.uniform(0.05, 0.35), 0.05, 0.40)
    event_count = int(np.clip(rng.normal(12, 5), 0, 35))
    existing_parking_spots = int(np.clip(rng.normal(18, 6), 0, 45))
    safety_incidents = int(np.clip(rng.normal(4, 2), 0, 15))
    current_station_count = int(np.clip(rng.normal(1.8, 1.0), 0, 6))
    future_growth_index = np.clip(rng.normal(0.55, 0.2), 0.1, 0.95)
    micromobility_friendly_score = np.clip(
        0.5 * bike_infra_km / 5.5 + 0.3 * (1 - weather_penalty) + 0.2 * future_growth_index,
        0,
        1,
    )

    demand_base = (
        0.35 * population_density / 22000
        + 0.25 * daytime_population / 24000
        + 0.15 * bike_infra_km / 5.5
        + 0.1 * micro_mobility_lanes_km / 5.0
        + 0.08 * transit_stops / 55
        + 0.07 * tourist_index
        + 0.05 * event_count / 35
    )

    penalties = 0.25 * competitor_density / 4.5 + 0.1 * safety_incidents / 15 + 0.05 * current_station_count / 6
    avg_daily_trips = np.clip((demand_base - penalties + 0.4) * 220 + rng.normal(0, 8), 8, 320)
    suitability_score = np.clip(
        0.5 * (avg_daily_trips / 320)
        + 0.3 * micromobility_friendly_score
        + 0.2 * (1 - weather_penalty)
        - 0.1 * competitor_density / 4.5,
        0,
        1,
    )

    return {
        "zone_id": f"Z{zone_index:04d}",
        "district": district,
        "zone_type": zone_type,
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "population_density": round(population_density, 2),
        "daytime_population": round(daytime_population, 2),
        "avg_income": round(avg_income, 2),
        "bike_infra_km": round(bike_infra_km, 3),
        "micro_mobility_lanes_km": round(micro_mobility_lanes_km, 3),
        "transit_stops": transit_stops,
        "competitor_density": round(competitor_density, 3),
        "tourist_index": round(tourist_index, 3),
        "weather_penalty": round(weather_penalty, 3),
        "event_count": event_count,
        "existing_parking_spots": existing_parking_spots,
        "safety_incidents": safety_incidents,
        "current_station_count": current_station_count,
        "future_growth_index": round(future_growth_index, 3),
        "micromobility_friendly_score": round(micromobility_friendly_score, 3),
        "avg_daily_trips": round(avg_daily_trips, 2),
        "suitability_score": round(suitability_score, 3),
    }


def main(config: DatasetConfig) -> None:
    rng = np.random.default_rng(config.seed)
    districts = [
        "CAO",
        "SAO",
        "SVAO",
        "VAO",
        "YUVAO",
        "YUAO",
        "YUZAO",
        "ZAO",
        "SZAO",
        "ZelAO",
    ]
    zone_types = [
        "residential",
        "business",
        "transport_hub",
        "tourist",
        "university",
    ]

    rows = [simulate_row(rng, idx + 1, districts, zone_types) for idx in range(config.samples)]
    df = pd.DataFrame(rows)
    config.outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.outfile, index=False)
    print(f"Dataset written to {config.outfile} with {len(df)} rows")


if __name__ == "__main__":
    main(DatasetConfig())
