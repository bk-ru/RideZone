"""Utilities to build a micromobility dataset for Moscow using OSM data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import geopandas as gpd
import h3
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon, mapping
from shapely.geometry.base import BaseGeometry
from h3.api.basic_str import LatLngPoly, polygon_to_cells

ox.settings.use_cache = True
ox.settings.log_console = False
TARGET_COLUMNS = [
    "zone_id",
    "district",
    "zone_type",
    "latitude",
    "longitude",
    "population_density",
    "daytime_population",
    "avg_income",
    "bike_infra_km",
    "micro_mobility_lanes_km",
    "transit_stops",
    "competitor_density",
    "tourist_index",
    "weather_penalty",
    "event_count",
    "existing_parking_spots",
    "safety_incidents",
    "current_station_count",
    "future_growth_index",
    "micromobility_friendly_score",
    "suitability_score",
    "avg_daily_trips",
]


@dataclass
class MoscowDatasetConfig:
    city_name: str = "Москва, Россия"
    resolution: int = 8
    output: Path = Path("data/raw/moscow_hex_zones.csv")
    min_population_density: int = 4000
    max_population_density: int = 22000


class MoscowOSMDatasetBuilder:
    """Collects OSM data for Moscow and converts it to RideZone features."""

    DISTRICTS = {
        "ЦАО": "Центральный административный округ, Москва, Россия",
        "САО": "Северный административный округ, Москва, Россия",
        "СВАО": "Северо-Восточный административный округ, Москва, Россия",
        "ВАО": "Восточный административный округ, Москва, Россия",
        "ЮВАО": "Юго-Восточный административный округ, Москва, Россия",
        "ЮАО": "Южный административный округ, Москва, Россия",
        "ЮЗАО": "Юго-Западный административный округ, Москва, Россия",
        "ЗАО": "Западный административный округ, Москва, Россия",
        "СЗАО": "Северо-Западный административный округ, Москва, Россия",
        "ЗелАО": "Зеленоградский административный округ, Москва, Россия",
        "ТАО": "Троицкий административный округ, Москва, Россия",
        "НАО": "Новомосковский административный округ, Москва, Россия",
    }

    LAYERS = {
        "transit": {
            "public_transport": ["station", "stop_position", "platform"],
            "railway": ["station", "stop", "halt", "tram_stop"],
            "highway": ["bus_stop"],
        },
        "cycleways": {"highway": ["cycleway"], "cycleway": True},
        "paths": {"highway": ["path", "footway"]},
        "commerce": {"shop": True, "amenity": ["bank", "pharmacy", "cafe", "restaurant", "fast_food", "supermarket"]},
        "education": {"amenity": ["school", "university", "college", "kindergarten"]},
        "residential": {"building": ["apartments", "residential", "dormitory"]},
        "office": {"building": ["office", "commercial", "retail"]},
        "tourism": {"tourism": ["attraction", "museum", "gallery", "viewpoint", "zoo"]},
        "leisure": {"leisure": ["park", "stadium", "sports_centre", "pitch", "garden"]},
        "parking": {"amenity": ["bicycle_parking", "parking"]},
        "stations": {"amenity": ["bicycle_rental", "charging_station"]},
        "primary_roads": {"highway": ["primary", "trunk"]},
    }

    def __init__(self, config: MoscowDatasetConfig | None = None) -> None:
        self.config = config or MoscowDatasetConfig()
        self.boundary_gdf: gpd.GeoDataFrame | None = None
        self.boundary_polygon: BaseGeometry | None = None
        self.features: dict[str, gpd.GeoDataFrame] = {}

    def build(self) -> pd.DataFrame:
        self._load_boundary()
        self._load_layers()
        hex_gdf = self._create_hex_grid()
        features_df = self._aggregate_features(hex_gdf)
        districts = self._assign_districts(hex_gdf)
        dataset = self._assemble_dataset(hex_gdf, features_df, districts)
        return dataset

    def _load_boundary(self) -> None:
        boundary = ox.geocode_to_gdf(self.config.city_name)
        self.boundary_gdf = boundary.to_crs("EPSG:4326")
        geom = self.boundary_gdf.geometry.unary_union
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        self.boundary_polygon = geom

    def _load_layers(self) -> None:
        polygon = self.boundary_polygon
        for name, tags in self.LAYERS.items():
            try:
                gdf = ox.features_from_polygon(polygon, tags=tags)
                gdf = gdf.reset_index()
                gdf = gdf[gdf.geometry.notnull()].copy()
                gdf = gdf.set_geometry("geometry").to_crs("EPSG:4326")
                self.features[name] = gdf
            except Exception:
                self.features[name] = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    def _create_hex_grid(self) -> gpd.GeoDataFrame:
        geom = self.boundary_polygon
        shell = [(lat, lon) for lon, lat in geom.exterior.coords]
        holes = [
            [(lat, lon) for lon, lat in interior.coords]
            for interior in geom.interiors
        ]
        latlng_poly = LatLngPoly(shell, *holes)
        cells = polygon_to_cells(latlng_poly, self.config.resolution)
        records = []
        for idx, cell in enumerate(cells):
            boundary_latlon = h3.cell_to_boundary(cell)
            poly = Polygon([(lon, lat) for lat, lon in boundary_latlon])
            lat, lon = h3.cell_to_latlng(cell)
            records.append({"hex_id": cell, "geometry": poly, "latitude": lat, "longitude": lon})
        return gpd.GeoDataFrame(records, crs="EPSG:4326")

    def _aggregate_features(self, hex_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=hex_gdf["hex_id"])
        df.index.name = "hex_id"
        df["transit_stops"] = self._count_points("transit")
        df["commerce_count"] = self._count_points("commerce")
        df["education_count"] = self._count_points("education")
        df["tourism_count"] = self._count_points("tourism")
        df["leisure_count"] = self._count_points("leisure")
        df["parking_count"] = self._count_points("parking")
        df["station_count"] = self._count_points("stations")
        df["office_count"] = self._count_points("office")
        df["residential_area"] = self._sum_area("residential")
        df["bike_km"] = self._sum_length("cycleways")
        df["path_km"] = self._sum_length("paths")
        df["primary_km"] = self._sum_length("primary_roads")
        return df.fillna(0).reset_index()

    def _assign_districts(self, hex_gdf: gpd.GeoDataFrame) -> list[str]:
        results: list[str] = []
        district_polygons: list[tuple[str, BaseGeometry]] = []
        for short_name, query in self.DISTRICTS.items():
            try:
                gdf = ox.geocode_to_gdf(query).to_crs("EPSG:4326")
                district_polygons.append((short_name, gdf.geometry.iloc[0]))
            except Exception:
                continue
        for _, row in hex_gdf.iterrows():
            centroid = row.geometry.centroid
            assigned = None
            for name, poly in district_polygons:
                if poly.contains(centroid):
                    assigned = name
                    break
            results.append(assigned or "Москва")
        return results

    def _assemble_dataset(self, hex_gdf: gpd.GeoDataFrame, features_df: pd.DataFrame, districts: Iterable[str]) -> pd.DataFrame:
        df = hex_gdf[["hex_id", "latitude", "longitude"]].copy()
        df = df.merge(features_df, on="hex_id", how="left").fillna(0)
        df["district"] = list(districts)
        df["zone_type"] = [self._infer_zone_type(row) for _, row in df.iterrows()]

        def scaled(series: pd.Series, low: float, high: float) -> pd.Series:
            if series.max() == series.min():
                return pd.Series(low + (high - low) / 2, index=series.index)
            normalized = (series - series.min()) / (series.max() - series.min())
            return normalized * (high - low) + low

        df["population_density"] = scaled(df["residential_area"], self.config.min_population_density, self.config.max_population_density)
        df["daytime_population"] = scaled(df["commerce_count"] + df["office_count"], 2000, 35000)
        df["avg_income"] = scaled(df["commerce_count"], 550, 1600)
        df["bike_infra_km"] = df["bike_km"].clip(lower=0)
        df["micro_mobility_lanes_km"] = (df["path_km"] * 0.7).clip(lower=0)
        df["transit_stops"] = df["transit_stops"].astype(int)
        df["competitor_density"] = scaled(df["station_count"], 0.1, 4.5)
        df["tourist_index"] = scaled(df["tourism_count"], 0.05, 0.95)
        df["weather_penalty"] = 0.22
        df["event_count"] = scaled(df["leisure_count"], 2, 30)
        df["existing_parking_spots"] = df["parking_count"].astype(int)
        df["safety_incidents"] = scaled(df["primary_km"], 1, 20)
        df["current_station_count"] = df["station_count"].astype(int)
        growth_base = scaled(df["residential_area"] + df["tourism_count"], 0.2, 0.95)
        df["future_growth_index"] = growth_base
        infra_score = (scaled(df["bike_infra_km"], 0, 5.5) + scaled(df["micro_mobility_lanes_km"], 0, 4.0)) / 2
        df["micromobility_friendly_score"] = infra_score.clip(0, 1)
        demand_signal = (
            0.4 * scaled(df["population_density"], 0, 1)
            + 0.25 * scaled(df["transit_stops"], 0, 1)
            + 0.2 * scaled(df["commerce_count"], 0, 1)
            + 0.15 * scaled(df["tourism_count"], 0, 1)
        )
        df["suitability_score"] = (0.6 * demand_signal + 0.4 * df["micromobility_friendly_score"]).clip(0, 1)
        df["avg_daily_trips"] = (df["suitability_score"] * 280 + 60).clip(50, 320)

        df_final = pd.DataFrame({
            "zone_id": [f"MSK_{i:04d}" for i in range(len(df))],
            "district": df["district"],
            "zone_type": df["zone_type"],
            "latitude": df["latitude"],
            "longitude": df["longitude"],
            "population_density": df["population_density"],
            "daytime_population": df["daytime_population"],
            "avg_income": df["avg_income"],
            "bike_infra_km": df["bike_infra_km"],
            "micro_mobility_lanes_km": df["micro_mobility_lanes_km"],
            "transit_stops": df["transit_stops"],
            "competitor_density": df["competitor_density"],
            "tourist_index": df["tourist_index"],
            "weather_penalty": df["weather_penalty"],
            "event_count": df["event_count"],
            "existing_parking_spots": df["existing_parking_spots"],
            "safety_incidents": df["safety_incidents"],
            "current_station_count": df["current_station_count"],
            "future_growth_index": df["future_growth_index"],
            "micromobility_friendly_score": df["micromobility_friendly_score"],
            "suitability_score": df["suitability_score"],
            "avg_daily_trips": df["avg_daily_trips"],
        })
        return df_final[TARGET_COLUMNS]

    def _infer_zone_type(self, row: pd.Series) -> str:
        scores = {
            "residential": row.get("residential_area", 0),
            "business": row.get("commerce_count", 0) + row.get("office_count", 0),
            "transport_hub": row.get("transit_stops", 0),
            "tourist": row.get("tourism_count", 0),
            "university": row.get("education_count", 0),
        }
        labels = {
            "residential": "жилой массив",
            "business": "деловой кластер",
            "transport_hub": "транспортный хаб",
            "tourist": "туристический кластер",
            "university": "университетский квартал",
        }
        best = max(scores, key=scores.get)
        return labels[best]

    def _count_points(self, layer_name: str) -> pd.Series:
        gdf = self.features.get(layer_name)
        if gdf is None or gdf.empty:
            return pd.Series(dtype=float)
        gdf = gdf.copy()
        gdf["centroid"] = gdf.geometry.centroid
        gdf = gdf[gdf["centroid"].notnull()]
        gdf["hex_id"] = gdf["centroid"].apply(lambda geom: h3.latlng_to_cell(geom.y, geom.x, self.config.resolution))
        return gdf.groupby("hex_id").size()

    def _sum_length(self, layer_name: str) -> pd.Series:
        gdf = self.features.get(layer_name)
        if gdf is None or gdf.empty:
            return pd.Series(dtype=float)
        metric = gdf.to_crs("EPSG:3857").copy()
        metric["length_km"] = metric.geometry.length / 1000
        centroids = gdf.to_crs("EPSG:4326").geometry.centroid
        metric = metric.assign(hex_id=[h3.latlng_to_cell(pt.y, pt.x, self.config.resolution) for pt in centroids])
        return metric.groupby("hex_id")["length_km"].sum()

    def _sum_area(self, layer_name: str) -> pd.Series:
        gdf = self.features.get(layer_name)
        if gdf is None or gdf.empty:
            return pd.Series(dtype=float)
        metric = gdf.to_crs("EPSG:3857").copy()
        metric["area_sqkm"] = metric.geometry.area / 1_000_000
        centroids = gdf.to_crs("EPSG:4326").geometry.centroid
        metric = metric.assign(hex_id=[h3.latlng_to_cell(pt.y, pt.x, self.config.resolution) for pt in centroids])
        return metric.groupby("hex_id")["area_sqkm"].sum()


def build_moscow_dataset(config: MoscowDatasetConfig | None = None) -> pd.DataFrame:
    builder = MoscowOSMDatasetBuilder(config)
    return builder.build()


def write_moscow_dataset(config: MoscowDatasetConfig | None = None) -> Path:
    cfg = config or MoscowDatasetConfig()
    df = build_moscow_dataset(cfg)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output, index=False)
    return cfg.output
