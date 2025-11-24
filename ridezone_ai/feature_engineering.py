"""Feature engineering module driven by OSM for transfer learning."""
from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import h3
from h3.api import basic_str as h3_basic
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .config import CityDescriptor, DataPaths, FeatureConfig, OSMConfig


class OSMFeatureEngineer:
    """Adds geo-features aligned between source (Chicago) and target (Moscow)."""

    def __init__(self, features: FeatureConfig, osm_config: OSMConfig, paths: DataPaths) -> None:
        self.config = features
        self.osm_config = osm_config
        self.paths = paths
        self.paths.boundary_cache_dir.mkdir(parents=True, exist_ok=True)
        self.paths.poi_cache_dir.mkdir(parents=True, exist_ok=True)
        self.paths.transit_cache_dir.mkdir(parents=True, exist_ok=True)

    def build_features(self, df: pd.DataFrame, city: CityDescriptor) -> pd.DataFrame:
        """Return dataframe with aligned feature columns for the given city."""
        features = df.copy()
        features["dist_to_center_km"] = features.apply(
            lambda row: self._haversine_km(city.center_lat, city.center_lon, row["lat"], row["lon"]), axis=1
        )
        if self.config.simulate_osm:
            # We simulate OSM-derived features when network is not available, keeping names consistent
            # for transfer learning across domains.
            return self._simulate_osm_features(features)

        boundary = self._get_boundary(city)
        try:
            transit = self._load_transit(boundary, city)
            pois = self._load_pois(boundary, city)
        except Exception:
            # If live OSM queries fail, fall back to synthetic features to keep the transfer workflow running.
            return self._simulate_osm_features(features)

        points = gpd.GeoDataFrame(
            features, geometry=gpd.points_from_xy(features["lon"], features["lat"]), crs="EPSG:4326"
        )
        points_proj = points.to_crs(epsg=3857)
        transit_proj = transit.to_crs(points_proj.crs) if not transit.empty else transit
        pois_proj = pois.to_crs(points_proj.crs) if not pois.empty else pois

        features["dist_to_subway_m"] = self._distance_to_transit(points_proj, transit_proj)
        features["poi_density_500m"] = self._count_pois(points_proj, pois_proj, self.config.poi_radius_m)
        return features.drop(columns="geometry")

    def build_h3_grid(self, city: CityDescriptor) -> pd.DataFrame:
        """Generate an H3 grid inside the city boundary for inference."""
        boundary = self._get_boundary(city)
        polygon = unary_union(boundary.geometry.values)
        hexes = self._polyfill(polygon)
        data = [{"h3_index": h, "lat": self._cell_to_lat_lon(h)[0], "lon": self._cell_to_lat_lon(h)[1]} for h in hexes]
        return pd.DataFrame(data)

    def _get_boundary(self, city: CityDescriptor) -> gpd.GeoDataFrame:
        cache_path = self._cached_path(self.paths.boundary_cache_dir, city, "boundary.parquet")
        if cache_path.exists():
            return gpd.read_parquet(cache_path)
        if self.config.simulate_osm:
            return self._boundary_from_bounds(city)
        try:
            gdf = ox.geocode_to_gdf(city.boundary_query, timeout=self.osm_config.boundary_timeout)
            gdf = gdf[["geometry"]]
            gdf.to_parquet(cache_path)
            return gdf
        except Exception:
            # If the OSM geocoder fails we fall back to a bounding-box polygon to keep the pipeline running.
            return self._boundary_from_bounds(city)

    def _boundary_from_bounds(self, city: CityDescriptor) -> gpd.GeoDataFrame:
        if city.name.lower().startswith("chicago"):
            lat_bounds, lon_bounds = self.config.chicago_bounds
        else:
            lat_bounds, lon_bounds = self.config.moscow_bounds
        polygon = Polygon(
            [
                (lon_bounds[0], lat_bounds[0]),
                (lon_bounds[1], lat_bounds[0]),
                (lon_bounds[1], lat_bounds[1]),
                (lon_bounds[0], lat_bounds[1]),
            ]
        )
        return gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")

    def _load_transit(self, boundary: gpd.GeoDataFrame, city: CityDescriptor) -> gpd.GeoDataFrame:
        cache_path = self._cached_path(self.paths.transit_cache_dir, city, "transit.parquet")
        if cache_path.exists():
            return gpd.read_parquet(cache_path)
        polygon = unary_union(boundary.geometry.values)
        try:
            gdf = ox.geometries_from_polygon(polygon, tags=self.osm_config.transit_tags, **self.osm_config.overpass_settings)
        except Exception:
            if self.config.simulate_osm:
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            raise
        gdf = gdf.reset_index()
        gdf = gdf.loc[gdf["geometry"].notna(), ["geometry"]]
        gdf = gdf.set_crs("EPSG:4326")
        gdf.to_parquet(cache_path)
        return gdf

    def _load_pois(self, boundary: gpd.GeoDataFrame, city: CityDescriptor) -> gpd.GeoDataFrame:
        cache_path = self._cached_path(self.paths.poi_cache_dir, city, "pois.parquet")
        if cache_path.exists():
            return gpd.read_parquet(cache_path)
        polygon = unary_union(boundary.geometry.values)
        try:
            gdf = ox.geometries_from_polygon(polygon, tags=self.osm_config.poi_tags, **self.osm_config.overpass_settings)
        except Exception:
            if self.config.simulate_osm:
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            raise
        gdf = gdf.reset_index()
        gdf = gdf.loc[gdf["geometry"].notna(), ["geometry"]]
        gdf = gdf.set_crs("EPSG:4326")
        gdf.to_parquet(cache_path)
        return gdf

    def _distance_to_transit(self, points: gpd.GeoDataFrame, transit: gpd.GeoDataFrame) -> pd.Series:
        if transit.empty:
            return pd.Series(np.full(len(points), self.config.poi_radius_m * 4), index=points.index)
        tree = STRtree(transit.geometry.values)
        distances = []
        for geom in points.geometry.values:
            nearest_geom = tree.nearest(geom)
            distances.append(geom.distance(nearest_geom))
        return pd.Series(distances, index=points.index, name="dist_to_subway_m")

    def _count_pois(self, points: gpd.GeoDataFrame, pois: gpd.GeoDataFrame, radius_m: int) -> pd.Series:
        if pois.empty:
            return pd.Series(np.zeros(len(points), dtype=int), index=points.index)
        buffers = points.geometry.buffer(radius_m)
        tree = STRtree(pois.geometry.values)
        counts = []
        for buf in buffers:
            matches = tree.query(buf)
            counts.append(len(matches))
        return pd.Series(counts, index=points.index, name="poi_density_500m")

    def _cached_path(self, directory: Path, city: CityDescriptor, suffix: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        safe_name = city.name.lower().replace(" ", "_")
        return directory / f"{safe_name}_{suffix}"

    def _simulate_osm_features(self, features: pd.DataFrame) -> pd.DataFrame:
        # Transfer learning note: when OSM is offline we still keep consistent feature names/shape
        # so the model trained on Chicago patterns can be applied to Moscow without schema drift.
        features["dist_to_subway_m"] = (features["dist_to_center_km"] * 450).clip(lower=100, upper=8000)
        pseudo_poi = 2000 / (features["dist_to_center_km"] + 0.5)
        features["poi_density_500m"] = np.clip(np.round(pseudo_poi), 0, None)
        return features

    def _polyfill(self, polygon) -> list[str]:
        if polygon.geom_type == "MultiPolygon":
            cells: set[str] = set()
            for poly in polygon.geoms:
                cells.update(self._polyfill(poly))
            return list(cells)
        geo_json = mapping(polygon)
        if hasattr(h3, "polyfill"):
            return list(h3.polyfill(geo_json, self.config.h3_resolution, geo_json_conformant=True))
        if hasattr(h3, "polygon_to_cells"):
            latlng_poly = self._to_latlng_poly(polygon)
            return list(h3.polygon_to_cells(latlng_poly, self.config.h3_resolution))
        raise AttributeError("No suitable H3 polyfill function found")

    @staticmethod
    def _cell_to_lat_lon(h3_index: str) -> tuple[float, float]:
        if hasattr(h3, "h3_to_geo"):
            lat, lon = h3.h3_to_geo(h3_index)
            return lat, lon
        if hasattr(h3, "cell_to_latlng"):
            lat, lon = h3.cell_to_latlng(h3_index)
            return lat, lon
        raise AttributeError("No suitable H3 reverse conversion function found")

    @staticmethod
    def _to_latlng_poly(polygon: Polygon) -> h3_basic.LatLngPoly:
        outer = [(lat, lon) for lon, lat in polygon.exterior.coords]
        holes = [[(lat, lon) for lon, lat in interior.coords] for interior in polygon.interiors]
        return h3_basic.LatLngPoly(outer, *holes)

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius = 6371.0
        lat1_rad, lon1_rad = map(math.radians, (lat1, lon1))
        lat2_rad, lon2_rad = map(math.radians, (lat2, lon2))
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return radius * c
