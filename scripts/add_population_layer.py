# scripts/add_population_layer.py
"""
Builds a grid's GHS-POP population layer: grids/<grid_id>/population.npz.

GHS-POP R2023A (Schiavina et al. 2023, JRC; 100 m, World Mollweide
ESRI:54009) gives modelled residential population per 100 m cell for
five-yearly epochs; 2025 and 2030 are projections. For each epoch found, the
tiles are reprojected onto the model grid (WGS84 cells, south-up rows as in
grid.npz) with SUM resampling - which conserves people - and summed across
tiles. The result is stored as population_<epoch> (float32, persons per
cell); grid.npz itself is never modified.

When grid.npz carries a "building_area" array (OSM building footprint area
per cell, written by scripts/make_settlement_grid for grids finer than
GHS-POP), people within each GHS-POP-sized block of cells are redistributed
towards building footprints, so that households land on buildings rather
than anywhere in the 100 m square. OSM mapping is uneven: in Akuse's
well-mapped eastern core a 100 m block has a median 27 m2 of footprint per
GHS-POP resident, but in the west 42% of populated blocks have under 3 m2 -
one or two mapped buildings among many unmapped ones - and moving a whole
block's people onto them would create crowding that is an artefact of
mapping. Each block is therefore a blend, weighted by its mapping
completeness c = min(1, footprint area / (people x FOOTPRINT_M2_PER_PERSON)):
a share c follows footprint area, the rest keeps the area-proportional
split. Block totals are always preserved; blocks with no mapped building
keep the area split.

Usage (tiles may be zipped as downloaded):
    python -m scripts.add_population_layer --grid-id 7d9ce7c720a6 \\
        --tiles "SPRINGS DATA/GHS POP/2020/*.zip" "SPRINGS DATA/GHS POP/2025/*.zip" ...
Requires rasterio.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import zipfile

import numpy as np

EPOCH_RE = re.compile(r"GHS_POP_E(\d{4})_")
GHS_CELL_M = 100.0
#: Footprint area per resident of a fully mapped block (m2): ~ the median in
#: Akuse's well-mapped eastern core (27 m2; quartiles of all populated blocks
#: 8-45). Used only to judge how complete a block's OSM mapping is.
FOOTPRINT_M2_PER_PERSON = 25.0


def _tif_paths(patterns: list[str]) -> list[str]:
    out = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            if path.lower().endswith(".zip"):
                with zipfile.ZipFile(path) as zf:
                    for name in zf.namelist():
                        if name.lower().endswith(".tif"):
                            out.append(f"zip://{os.path.abspath(path)}!{name}")
            elif path.lower().endswith(".tif"):
                out.append(path)
    return out


def population_layers(grid_path: str, tif_paths: list[str]) -> tuple[dict[int, np.ndarray], dict]:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    data = np.load(grid_path, allow_pickle=True)
    rows, cols = data["grid"].shape[:2]
    minx, miny, x_step, y_step = (float(v) for v in data["bounds"])
    maxy = miny + rows * y_step
    dst_transform = from_origin(minx, maxy, x_step, y_step)          # north-up
    layers: dict[int, np.ndarray] = {}
    sources: dict[int, list[str]] = {}
    for path in tif_paths:
        m = EPOCH_RE.search(path)
        if not m:
            continue
        epoch = int(m.group(1))
        with rasterio.open(path) as src:
            dst = np.zeros((rows, cols), dtype=np.float64)
            reproject(source=rasterio.band(src, 1), destination=dst, src_transform=src.transform,
                      src_crs=src.crs, src_nodata=src.nodata, dst_transform=dst_transform,
                      dst_crs="EPSG:4326", dst_nodata=0.0, resampling=Resampling.sum)
        dst[~np.isfinite(dst) | (dst < 0)] = 0.0
        layers[epoch] = layers.get(epoch, np.zeros((rows, cols))) + dst[::-1]   # south-up like grid.npz
        sources.setdefault(epoch, []).append(os.path.basename(path.split("!")[-1]))

    use_buildings = "building_area" in data.files and float(data["building_area"].sum()) > 0
    if use_buildings:
        layers = {e: _redistribute_to_buildings(p, data["building_area"], x_step, y_step, miny, rows)
                  for e, p in layers.items()}

    meta = {"source": "GHS-POP R2023A (JRC), 100 m, ESRI:54009, SUM-resampled onto the grid",
            "tiles": sources, "building_redistribution": use_buildings,
            "footprint_m2_per_person": FOOTPRINT_M2_PER_PERSON if use_buildings else None,
            "totals": {str(e): float(p.sum()) for e, p in layers.items()}}
    return layers, meta


def _redistribute_to_buildings(pop: np.ndarray, area: np.ndarray, x_step: float, y_step: float,
                               miny: float, rows: int) -> np.ndarray:
    """Within each ~100 m block of cells, move a share c of the block's
    people onto cells in proportion to building footprint area, where c is
    the block's mapping completeness (see module docstring); the rest keep
    their area-proportional split. Totals per block are preserved."""
    lat = miny + rows * y_step / 2
    cell_x = x_step * 111_320 * math.cos(math.radians(lat))
    cell_y = y_step * 110_574
    bx = max(1, int(round(GHS_CELL_M / cell_x)))
    by = max(1, int(round(GHS_CELL_M / cell_y)))
    out = pop.copy()
    R, C = pop.shape
    for r0 in range(0, R, by):
        for c0 in range(0, C, bx):
            blk = (slice(r0, r0 + by), slice(c0, c0 + bx))
            a = area[blk].astype(np.float64)
            total = pop[blk].sum()
            if a.sum() > 0 and total > 1e-6:
                c = min(1.0, a.sum() / (total * FOOTPRINT_M2_PER_PERSON))
                out[blk] = c * total * a / a.sum() + (1.0 - c) * pop[blk]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--grid-id", required=True)
    ap.add_argument("--tiles", nargs="+", required=True, help="GHS-POP tile paths or globs (.tif or the downloaded .zip)")
    args = ap.parse_args()
    grid_dir = os.path.join("grids", args.grid_id)
    layers, meta = population_layers(os.path.join(grid_dir, "grid.npz"), _tif_paths(args.tiles))
    if not layers:
        raise SystemExit("No GHS-POP tiles found.")
    out = os.path.join(grid_dir, "population.npz")
    np.savez_compressed(out, **{f"population_{e}": p.astype(np.float32) for e, p in sorted(layers.items())})
    with open(os.path.join(grid_dir, "population_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for e, p in sorted(layers.items()):
        print(f"  {e}: {p.sum():,.0f} people on the grid")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
