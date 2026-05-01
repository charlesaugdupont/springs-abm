# abm/environment/grid_generator.py
import os
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon, box
import json
import hashlib
from pathlib import Path
import pandas as pd
import re

from abm.constants import GridLayer
from abm.utils.rng import set_global_seed, get_np_rng

AKUSE_CENTER_POINT = (6.0993, 0.12821)
AKUSE_BOUNDARY_COORDS = [
    (0.1118, 6.1135),
    (0.1661, 6.1135),
    (0.1661, 6.0677),
    (0.1118, 6.0677),
]
GRID_SIZE = 100
POI_FETCH_RADIUS = 2000
OSM_POI_TAGS = {"amenity": [GridLayer.SCHOOL, GridLayer.WORSHIP]}
PROCEDURAL_POI_COUNTS = {GridLayer.SCHOOL: 5, GridLayer.WORSHIP: 5, GridLayer.WATER: 0}
WATER_SAMPLING_CSV = Path("abm/data/water_sampling_points.csv")

def get_grid_id(boundary_coords, grid_size, osm_tags, procedural_counts) -> str:
    """Creates a unique, deterministic hash from all grid generation parameters."""
    params = {
        "boundary_coords": boundary_coords,
        "grid_size": grid_size,
        "osm_poi_tags": osm_tags,
        "procedural_poi_counts": procedural_counts
    }
    params_string = json.dumps(params, sort_keys=True)
    return hashlib.sha256(params_string.encode('utf-8')).hexdigest()[:12]

def _initialize_grid_and_boundary():
    """Defines the geographical boundary and creates the base valid cells mask."""
    boundary = Polygon(AKUSE_BOUNDARY_COORDS)
    minx, miny, maxx, maxy = boundary.bounds
    x_step = (maxx - minx) / GRID_SIZE
    y_step = (maxy - miny) / GRID_SIZE
    valid_cells_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = box(minx + c * x_step, miny + r * y_step,
                       minx + (c + 1) * x_step, miny + (r + 1) * y_step)
            if cell.intersects(boundary):
                valid_cells_mask[r, c] = True
    return valid_cells_mask, (minx, miny, x_step, y_step)

def _coord_to_cell(lon: float, lat: float, boundary_info) -> tuple[int, int] | None:
    """
    Map a (lon, lat) coordinate to a grid cell (row, col) using the same
    discretisation as _initialize_grid_and_boundary().

    Returns (row, col) or None if the coordinate lies outside the grid.
    """
    minx, miny, x_step, y_step = boundary_info

    if lon < minx or lon > minx + x_step * GRID_SIZE:
        return None
    if lat < miny or lat > miny + y_step * GRID_SIZE:
        return None

    col = int((lon - minx) / x_step)
    row = int((lat - miny) / y_step)

    if row < 0 or row >= GRID_SIZE or col < 0 or col >= GRID_SIZE:
        return None

    return row, col

def _dms_to_decimal(dms_str: str) -> float | None:
    """
    Convert a DMS string like "000° 07' 38.67''" to decimal degrees.
    Uses \\D+ as separator to avoid encoding issues with the degree symbol.
    """
    if dms_str is None or (isinstance(dms_str, float) and pd.isna(dms_str)):
        return None
    s = str(dms_str).strip()
    m = re.search(r"(\d+)\D+(\d+)\D+([0-9.]+)", s)
    if not m:
        return None
    d, m1, s1 = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return d + m1 / 60.0 + s1 / 3600.0


def _load_akuse_water_cells(boundary_info, valid_cells_mask: np.ndarray) -> np.ndarray | None:
    """
    Load real water sampling points for Akuse and map them onto grid cells.

    Returns a binary (0/1) array of shape (GRID_SIZE, GRID_SIZE) where 1
    marks a water source cell (initially CLEAN), or None if the CSV is
    missing or no points fall into the grid.
    """
    if not WATER_SAMPLING_CSV.exists():
        print(f"  - Warning: water sampling CSV not found at {WATER_SAMPLING_CSV}.")
        return None

    try:
        df = pd.read_csv(WATER_SAMPLING_CSV, encoding="latin-1")
    except Exception as e:
        print(f"  - Warning: failed to read {WATER_SAMPLING_CSV}: {e}")
        return None

    if "Longitude" not in df.columns or "Latitude" not in df.columns:
        print("  - Warning: CSV missing Longitude/Latitude columns.")
        return None

    # Restrict to Akuse-labelled sites
    site_mask = df["site"].astype(str).str.contains("Akuse", case=False, na=False)
    df = df[site_mask].copy()
    if df.empty:
        print("  - Warning: no Akuse-labelled water points in CSV.")
        return None

    # Convert to decimal degrees
    df["lon_dd"] = df["Longitude"].apply(_dms_to_decimal)
    df["lat_dd"] = df["Latitude"].apply(_dms_to_decimal)
    df = df.dropna(subset=["lon_dd", "lat_dd"])

    water_layer = np.zeros_like(valid_cells_mask, dtype=np.uint8)
    used_cells: set[tuple[int, int]] = set()

    for _, row in df.iterrows():
        lon = row["lon_dd"]
        lat = row["lat_dd"]
        cell = _coord_to_cell(lon, lat, boundary_info)
        if cell is None:
            continue
        r, c = cell
        # Only use cells that intersect the Akuse polygon
        if not valid_cells_mask[r, c]:
            continue
        if (r, c) in used_cells:
            continue
        used_cells.add((r, c))
        water_layer[r, c] = 1  # 1 corresponds to WaterStatus.CLEAN

    if not used_cells:
        print("  - Warning: no Akuse water sampling points fell inside the grid.")
        return None

    print(f"  - Placed {len(used_cells)} real Akuse water sources from CSV.")
    return water_layer

def _place_osm_pois(tags, boundary_info, valid_cells_mask):
    """Fetches and places POIs from OpenStreetMap onto grid layers."""
    minx, miny, x_step, y_step = boundary_info
    try:
        pois_gdf = ox.features_from_point(AKUSE_CENTER_POINT, tags, dist=POI_FETCH_RADIUS)
    except ox._errors.InsufficientResponseError:
        print("  - Warning: OSM did not return any data. Skipping real POIs.")
        return {}, np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    osm_layers = {}
    for amenity_type in tags.get("amenity", []):
        osm_layers[amenity_type] = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        locations = pois_gdf[pois_gdf["amenity"] == amenity_type]
        for _, poi in locations.iterrows():
            geom = poi['geometry'].centroid if isinstance(poi['geometry'], Polygon) else poi['geometry']
            c = int((geom.x - minx) / x_step)
            r = int((geom.y - miny) / y_step)
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and valid_cells_mask[r, c]:
                osm_layers[amenity_type][r, c] = 1
    all_osm_mask = np.sum(list(osm_layers.values()), axis=0) > 0
    return osm_layers, all_osm_mask

def _place_procedural_pois(counts, valid_cells_mask, occupied_mask, boundary_info):
    """
    Places procedural POIs (schools, worship, water) onto the grid.

    For water, we first try to place real Akuse water sources from the
    sampling CSV; if there are fewer than the requested count, we top up
    with random water cells to approximately match the original density.
    """
    rng = get_np_rng()
    procedural_layers: dict[str, np.ndarray] = {}
    current_occupied = occupied_mask.copy()

    for amenity, num in counts.items():
        layer = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

        if amenity == GridLayer.WATER:
            # 1. Try to place real Akuse water sources
            real_water = _load_akuse_water_cells(boundary_info, valid_cells_mask)
            n_real = int(real_water.sum()) if real_water is not None else 0
            if n_real > 0:
                layer = real_water.copy()
                current_occupied = current_occupied | (layer > 0)

            # 2. Optionally top up with procedural water to reach num
            if n_real < num:
                needed = num - n_real
                available_cells = np.argwhere(valid_cells_mask & ~current_occupied)
                if len(available_cells) >= needed > 0:
                    selected_indices = rng.choice(len(available_cells), size=needed, replace=False)
                    cells = available_cells[selected_indices]
                    layer[cells[:, 0], cells[:, 1]] = 1
                    current_occupied[cells[:, 0], cells[:, 1]] = True

            procedural_layers[amenity] = layer
            continue  # done with water

        # Default behaviour for other POI types (school, worship)
        available_cells = np.argwhere(valid_cells_mask & ~current_occupied)
        if len(available_cells) >= num > 0:
            selected_indices = rng.choice(len(available_cells), size=num, replace=False)
            cells = available_cells[selected_indices]
            layer[cells[:, 0], cells[:, 1]] = 1
            current_occupied[cells[:, 0], cells[:, 1]] = True

        procedural_layers[amenity] = layer

    return procedural_layers

def _place_animal_density(valid_cells):
    """Generates a spatial layer representing poultry/livestock density."""
    rng = get_np_rng()

    density = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # Background noise
    noise = rng.uniform(0.0, 0.1, size=(GRID_SIZE, GRID_SIZE))
    density += noise

    # Clusters
    num_clusters = 15
    for _ in range(num_clusters):
        cx, cy = rng.integers(0, GRID_SIZE, size=2)
        y, x = np.ogrid[-cx:GRID_SIZE-cx, -cy:GRID_SIZE-cy]
        dist_sq = x**2 + y**2
        radius = 8
        mask = dist_sq <= radius**2
        if not np.any(mask):
            continue
        cluster_vals = np.exp(-0.5 * dist_sq[mask] / (radius / 2) ** 2)
        scale = rng.uniform(0.4, 0.9)
        density[mask] += cluster_vals * scale

    return np.clip(density * valid_cells, 0.0, 1.0)

def create_and_save_realistic_grid():
    """Generates and saves a realistic base grid for the model."""
    grid_id = get_grid_id(AKUSE_BOUNDARY_COORDS, GRID_SIZE, OSM_POI_TAGS, PROCEDURAL_POI_COUNTS)
    print(f"1. Generating grid with ID: {grid_id}")

    # Derive a deterministic seed from the grid_id and set the global RNG
    seed_int = int(grid_id[:8], 16)
    set_global_seed(seed_int)

    valid_cells, boundary_info = _initialize_grid_and_boundary()
    osm_layers, osm_occupied = _place_osm_pois(OSM_POI_TAGS, boundary_info, valid_cells)
    proc_layers = _place_procedural_pois(PROCEDURAL_POI_COUNTS, valid_cells, osm_occupied, boundary_info)

    print("2. Combining all layers...")
    final_layers, property_map = [], {}
    all_poi_types = set(osm_layers.keys()) | set(proc_layers.keys())

    # Layer 0: Residences
    final_layers.append(valid_cells.astype(np.uint8))
    property_map[0] = GridLayer.RESIDENCES

    # POIs
    poi_layers_stack = []
    for amenity in sorted(list(all_poi_types)):
        combined = ((osm_layers.get(amenity, 0) + proc_layers.get(amenity, 0)) > 0).astype(np.uint8)
        layer_index = len(final_layers)
        final_layers.append(combined)
        property_map[layer_index] = amenity
        poi_layers_stack.append(combined)

    # Animal density
    animal_density = _place_animal_density(valid_cells)
    final_layers.append(animal_density)
    property_map[len(final_layers) - 1] = GridLayer.ANIMAL_DENSITY

    final_grid = np.stack(final_layers, axis=-1)

    # Ensure residences not on top of POIs
    if poi_layers_stack:
        all_poi_mask = np.sum(np.stack(poi_layers_stack, axis=-1), axis=2) > 0
        final_grid[:, :, 0][all_poi_mask] = 0

    print("3. Saving base grid and metadata...")
    grid_dir = os.path.join("grids", grid_id)
    os.makedirs(grid_dir, exist_ok=True)
    output_path = os.path.join(grid_dir, "grid.npz")
    np.savez_compressed(output_path, grid=final_grid, property_map=property_map, bounds=np.array(boundary_info))
    print(f"\n--- Grid Generation Complete ---\n  Grid ID: {grid_id}\n  Saved to: {output_path}")

if __name__ == '__main__':
    create_and_save_realistic_grid()