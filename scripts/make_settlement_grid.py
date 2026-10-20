# scripts/make_settlement_grid.py
"""
Builds the model grid for a settlement domain at a fixed cell size in metres.

The original grid (abm/environment/grid_generator.py, id 7d9ce7c720a6) is a
100 x 100 division of a hand-drawn box: ~60 x 51 m cells, households spread
uniformly over every land cell. This builds instead:

* the DOMAIN: a settlement polygon (default abm/environment/domains/
  akuse_settlement.geojson - the contiguous built-up area around Akuse in
  GHS-POP 2025, ~11,600 people) plus `--surroundings-m` (default 1 km) of
  surroundings, so that the rivers, canals, schools and churches its people
  use are on the grid;
* square cells of `--cell-size-m` (default 25 m: in the densest part of
  town about two households share a cell, i.e. a compound);
* layers 0-4 in the same order and names as the original grid so the model
  reads it unchanged: residences (cells inside the settlement polygon, not
  water, not a school/church/sampling point), place_of_worship, school,
  water (the Akuse sampling points), natural_water (OSM natural=water and
  waterway=river/canal/stream, lines buffered to a realistic half-width);
  then layers the model does not use yet: marketplace, health_facility,
  osm_water_point;
* extra arrays: building_area (OSM building footprint m^2 per cell - used by
  scripts/add_population_layer.py to put households on buildings) and
  settlement (cell centre inside the settlement polygon).

Run from the repository root (needs OpenStreetMap access; osmnx, shapely>=2,
numpy, pandas):
    python -m scripts.make_settlement_grid
It prints the grid id and writes grids/<grid_id>/grid.npz and grid_meta.json.
Then: python -m scripts.add_population_layer --grid-id <grid_id> --tiles ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

M_PER_DEG_LAT = 110_574.0          # same constants as abm/environment/water_network.py
M_PER_DEG_LON_EQ = 111_320.0
# Points of interest, fetched within the grid's bounding box. The first two
# are the model's activity locations (layers 1-2, as in the original grid);
# the rest are extra layers (5+) that the model does not use yet: markets
# (adult daytime mixing, food exposure), health facilities (distance to
# care, for the household economy) and mapped water points (wells, taps).
POI_LAYERS = {
    "place_of_worship": {"amenity": ["place_of_worship"]},
    "school": {"amenity": ["school", "kindergarten"]},
    "marketplace": {"amenity": ["marketplace"]},
    "health_facility": {"amenity": ["hospital", "clinic", "doctors", "pharmacy"], "healthcare": True},
    "osm_water_point": {"amenity": ["drinking_water", "water_point"],
                        "man_made": ["water_well", "water_tap", "water_works"]},
}
NATURAL_WATER_TAGS = {"natural": "water", "waterway": ["river", "canal", "stream"]}
# Half-widths (m) for waterways mapped as centre lines. Wide rivers are
# normally also mapped as natural=water polygons; these keep a line-only
# feature at a plausible width. A cell is water if the feature touches it.
WATERWAY_HALF_WIDTH_M = {"river": 30.0, "canal": 5.0, "stream": 2.5}
WATER_SAMPLING_CSV = Path("abm/data/water_sampling_points.csv")
VERSION = "settlement-grid-v1"
LAYERS = ["residences", "place_of_worship", "school", "water", "natural_water",
          "marketplace", "health_facility", "osm_water_point"]
OVERPASS_ATTEMPTS = 6
OVERPASS_FIRST_WAIT_S = 15


def _dms_to_decimal(dms) -> float | None:
    if dms is None or (isinstance(dms, float) and math.isnan(dms)):
        return None
    m = re.search(r"(\d+)\D+(\d+)\D+([0-9.]+)", str(dms))
    if not m:
        return None
    return float(m.group(1)) + float(m.group(2)) / 60.0 + float(m.group(3)) / 3600.0


class LocalFrame:
    """Equirectangular metres around a reference point - the conversion the
    model itself uses, so cells are exactly cell_size_m on its own terms."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
        self.ky = M_PER_DEG_LAT

    def to_m(self, geom):
        import shapely
        return shapely.transform(geom, lambda xy: np.column_stack(
            [(xy[:, 0] - self.lon0) * self.kx, (xy[:, 1] - self.lat0) * self.ky]))

    def to_deg(self, geom):
        import shapely
        return shapely.transform(geom, lambda xy: np.column_stack(
            [xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0]))


def _query(fn, *args, what: str, sleep=time.sleep):
    """Runs an osmnx query, retrying with exponential backoff on server or
    network errors (the public Overpass server answers 502/504 when busy).
    An empty answer (osmnx's InsufficientResponseError) returns None - the
    feature genuinely is not mapped. If every attempt fails the script stops
    WITHOUT writing a grid, rather than saving one with a layer missing."""
    for k in range(OVERPASS_ATTEMPTS):
        try:
            return fn(*args)
        except Exception as e:                    # noqa: BLE001 - osmnx error types vary by version
            if type(e).__name__ == "InsufficientResponseError":
                print(f"  {what}: none mapped")
                return None
            if k == OVERPASS_ATTEMPTS - 1:
                raise SystemExit(
                    f"{what}: OpenStreetMap query failed {OVERPASS_ATTEMPTS} times "
                    f"({type(e).__name__}: {str(e)[:200]}). No grid was written. Try again later, or "
                    f"use another Overpass server with --overpass-url (e.g. "
                    f"https://overpass.private.coffee/api)."
                ) from e
            wait = OVERPASS_FIRST_WAIT_S * 2 ** k
            print(f"  {what}: {type(e).__name__} - retrying in {wait} s ({k + 1}/{OVERPASS_ATTEMPTS})")
            sleep(wait)


def build(domain_path: str, cell_size_m: float, surroundings_m: float, out_root: str = "grids",
          osm=None, overpass_url: str | None = None) -> str:
    import shapely
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union

    if osm is None:
        import osmnx as osm                       # noqa: N813
    if overpass_url and hasattr(osm, "settings"):
        osm.settings.overpass_url = overpass_url
    domain_bytes = Path(domain_path).read_bytes()
    gj = json.loads(domain_bytes)
    settlement_deg = unary_union([shape(f["geometry"]) for f in gj["features"]])
    c = settlement_deg.centroid
    frame = LocalFrame(c.x, c.y)
    settlement = frame.to_m(settlement_deg)
    area = settlement.buffer(surroundings_m)
    minx, miny, maxx, maxy = area.bounds
    cols = int(math.ceil((maxx - minx) / cell_size_m))
    rows = int(math.ceil((maxy - miny) / cell_size_m))
    params = {"version": VERSION, "domain_sha256": hashlib.sha256(domain_bytes).hexdigest(),
              "cell_size_m": cell_size_m, "surroundings_m": surroundings_m,
              "poi_layers": POI_LAYERS, "natural_water_tags": NATURAL_WATER_TAGS,
              "waterway_half_width_m": WATERWAY_HALF_WIDTH_M}
    grid_id = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    print(f"Grid {grid_id}: {rows} rows x {cols} cols of {cell_size_m:g} m "
          f"({(maxx - minx) / 1000:.2f} x {(maxy - miny) / 1000:.2f} km)")

    # Cell boxes in metres, row 0 = south (as grid.npz), and a spatial index.
    xs = minx + np.arange(cols) * cell_size_m
    ys = miny + np.arange(rows) * cell_size_m
    X0, Y0 = np.meshgrid(xs, ys)
    cells = shapely.box(X0.ravel(), Y0.ravel(), X0.ravel() + cell_size_m, Y0.ravel() + cell_size_m)
    tree = shapely.STRtree(cells)
    centres_x, centres_y = X0.ravel() + cell_size_m / 2, Y0.ravel() + cell_size_m / 2

    def cells_touching(geom_m) -> np.ndarray:
        return tree.query(geom_m, predicate="intersects")

    bbox_deg = frame.to_deg(shapely.box(minx, miny, maxx, maxy))

    # --- natural water ---
    print("Fetching natural water (OSM)...")
    water = np.zeros(rows * cols, dtype=bool)
    n_water_features = 0
    wgdf = _query(osm.features_from_polygon, bbox_deg, NATURAL_WATER_TAGS, what="natural water")
    if wgdf is not None:
        for _, f in wgdf.iterrows():
            g = f.geometry
            if g is None or g.is_empty:
                continue
            g = frame.to_m(g)
            if g.geom_type in ("LineString", "MultiLineString"):
                kind = str(f.get("waterway", "stream"))
                g = g.buffer(WATERWAY_HALF_WIDTH_M.get(kind, 2.5))
            water[cells_touching(g)] = True
            n_water_features += 1

    # --- points of interest ---
    poi = {k: np.zeros(rows * cols, dtype=bool) for k in POI_LAYERS}
    poi_counts = {}
    for layer, tags in POI_LAYERS.items():
        print(f"Fetching {layer} (OSM)...")
        pgdf = _query(osm.features_from_polygon, bbox_deg, tags, what=layer)
        n = 0
        if pgdf is not None:
            for g in pgdf.geometry:
                if g is None or g.is_empty:
                    continue
                hit = cells_touching(frame.to_m(g.centroid))
                if len(hit):
                    poi[layer][hit[0]] = True
                    n += 1
        poi_counts[layer] = n

    # --- Akuse water sampling points ---
    samp = np.zeros(rows * cols, dtype=bool)
    if WATER_SAMPLING_CSV.exists():
        df = pd.read_csv(WATER_SAMPLING_CSV, encoding="latin-1")
        df = df[df["site"].astype(str).str.contains("Akuse", case=False, na=False)]
        for _, r in df.iterrows():
            lon, lat = _dms_to_decimal(r["Longitude"]), _dms_to_decimal(r["Latitude"])
            if lon is None or lat is None:
                continue
            hit = cells_touching(frame.to_m(Point(lon, lat)))
            if len(hit):
                samp[hit[0]] = True
    else:
        print(f"  warning: {WATER_SAMPLING_CSV} not found - no sampling points")

    # --- buildings ---
    print("Fetching building footprints (OSM)...")
    building_area = np.zeros(rows * cols, dtype=np.float64)
    n_buildings = 0
    bgdf = _query(osm.features_from_polygon, bbox_deg, {"building": True}, what="buildings")
    if bgdf is not None:
        geoms = [frame.to_m(g) for g in bgdf.geometry
                 if g is not None and not g.is_empty and g.geom_type in ("Polygon", "MultiPolygon")]
        for g in geoms:
            idx = cells_touching(g)
            if len(idx):
                building_area[idx] += shapely.area(shapely.intersection(cells[idx], g))
        n_buildings = len(geoms)

    # --- settlement mask and residences ---
    settlement_mask = shapely.contains_xy(settlement, centres_x, centres_y)
    occupied = poi["place_of_worship"] | poi["school"] | samp
    residences = settlement_mask & ~water & ~occupied

    stack = np.stack([residences, poi["place_of_worship"], poi["school"], samp, water,
                      poi["marketplace"], poi["health_facility"], poi["osm_water_point"]], axis=-1)
    grid = stack.reshape(rows, cols, len(LAYERS)).astype(np.uint8)
    lon_min, lat_min = minx / frame.kx + frame.lon0, miny / frame.ky + frame.lat0
    bounds = np.array([lon_min, lat_min, cell_size_m / frame.kx, cell_size_m / frame.ky])

    out_dir = os.path.join(out_root, grid_id)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "grid.npz"), grid=grid, bounds=bounds,
        property_map={i: name for i, name in enumerate(LAYERS)},
        building_area=building_area.reshape(rows, cols).astype(np.float32),
        settlement=settlement_mask.reshape(rows, cols),
    )
    meta = {**params, "grid_id": grid_id, "rows": rows, "cols": cols, "bounds": bounds.tolist(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "osmnx_version": getattr(osm, "__version__", "unknown"),
            "counts": {"natural_water_features": n_water_features, "water_cells": int(water.sum()),
                       "buildings": n_buildings, "cells_with_buildings": int((building_area > 0).sum()),
                       "settlement_cells": int(settlement_mask.sum()), "residence_cells": int(residences.sum()),
                       **{f"{k}_features": v for k, v in poi_counts.items()},
                       **{f"{k}_cells": int(poi[k].sum()) for k in POI_LAYERS},
                       "sampling_points": int(samp.sum())}}
    with open(os.path.join(out_dir, "grid_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for k, v in meta["counts"].items():
        print(f"  {k}: {v}")
    print(f"Saved {out_dir}/grid.npz  (grid id {grid_id})")
    return grid_id


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--domain", default="abm/environment/domains/akuse_settlement.geojson")
    ap.add_argument("--cell-size-m", type=float, default=25.0)
    ap.add_argument("--surroundings-m", type=float, default=1000.0)
    ap.add_argument("--overpass-url", default=None,
                    help="Overpass API endpoint to use instead of osmnx's default (overpass-api.de)")
    args = ap.parse_args()
    build(args.domain, args.cell_size_m, args.surroundings_m, overpass_url=args.overpass_url)


if __name__ == "__main__":
    main()
