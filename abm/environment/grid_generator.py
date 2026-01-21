# abm/environment/grid_generator.py
import os
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon, box
import random
import json
import hashlib

from abm.constants import GridLayer

# These default parameters could be moved to a configuration file
# for greater flexibility in future studies.
AKUSE_CENTER_POINT = (6.100635676379143, 0.12736045405150456)
AKUSE_BOUNDARY_COORDS = [
    (0.11640835827890646, 6.107983818842101), (0.13688109260416578, 6.106918939321614),
    (0.13518748441180858, 6.095601368066649), (0.11628382826476255, 6.090648307705022)
]
GRID_SIZE = 75
POI_FETCH_RADIUS = 1500
OSM_POI_TAGS = {"amenity": [GridLayer.SCHOOL, GridLayer.WORSHIP]}
PROCEDURAL_POI_COUNTS = {GridLayer.SCHOOL: 5, GridLayer.WORSHIP: 5, GridLayer.WATER: 20}

def get_grid_id(boundary_coords, grid_size, osm_tags, procedural_counts) -> str:
    """Creates a unique, deterministic hash from all grid generation parameters."""
    params = {
        "boundary_coords": boundary_coords, "grid_size": grid_size,
        "osm_poi_tags": osm_tags, "procedural_poi_counts": procedural_counts,
        "animal_layer_version": "v1.1" # Versioning the logic itself
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

def _place_osm_pois(tags, boundary_info):
    """Fetches and places POIs from OpenStreetMap onto grid layers."""
    minx, miny, x_step, y_step = boundary_info
    try:
        pois_gdf = ox.features_from_point(AKUSE_CENTER_POINT, tags, dist=POI_FETCH_RADIUS)
    except ox._errors.InsufficientResponse:
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
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                osm_layers[amenity_type][r, c] = 1
    all_osm_mask = np.sum(list(osm_layers.values()), axis=0) > 0
    return osm_layers, all_osm_mask

def _place_procedural_pois(counts, valid_cells_mask, occupied_mask):
    """Places randomly located procedural POIs onto grid layers."""
    procedural_layers = {}
    current_occupied = occupied_mask.copy()
    for amenity, num in counts.items():
        procedural_layers[amenity] = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        available_cells = np.argwhere(valid_cells_mask & ~current_occupied)
        if len(available_cells) >= num:
            selected_indices = random.sample(range(len(available_cells)), num)
            cells = available_cells[selected_indices]
            procedural_layers[amenity][cells[:, 0], cells[:, 1]] = 1
            current_occupied[cells[:, 0], cells[:, 1]] = True
    return procedural_layers

def _place_animal_density(valid_cells):
    """Generates a spatial layer representing poultry/livestock density."""
    density = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    noise = np.random.uniform(0.0, 0.1, size=(GRID_SIZE, GRID_SIZE))
    density += noise
    for _ in range(15): # Number of clusters
        cx, cy = np.random.randint(0, GRID_SIZE, 2)
        y, x = np.ogrid[-cx:GRID_SIZE-cx, -cy:GRID_SIZE-cy]
        dist_sq = x**2 + y**2
        radius = 8
        mask = dist_sq <= radius**2
        cluster_vals = np.exp(-0.5 * dist_sq[mask] / (radius/2)**2)
        density[mask] += cluster_vals * np.random.uniform(0.4, 0.9)
    return np.clip(density * valid_cells, 0.0, 1.0)

def create_and_save_realistic_grid():
    """Generates and saves a realistic base grid for the model."""
    grid_id = get_grid_id(AKUSE_BOUNDARY_COORDS, GRID_SIZE, OSM_POI_TAGS, PROCEDURAL_POI_COUNTS)
    print(f"1. Generating grid with ID: {grid_id}")
    valid_cells, boundary_info = _initialize_grid_and_boundary()
    osm_layers, osm_occupied = _place_osm_pois(OSM_POI_TAGS, boundary_info)
    proc_layers = _place_procedural_pois(PROCEDURAL_POI_COUNTS, valid_cells, osm_occupied)

    print("2. Combining all layers...")
    final_layers, property_map = [], {}
    all_poi_types = set(osm_layers.keys()) | set(proc_layers.keys())
    
    # Layer 0: Residences
    final_layers.append(valid_cells.astype(np.uint8))
    property_map[0] = GridLayer.RESIDENCES
    
    # Subsequent layers: POIs
    poi_layers_stack = []
    for amenity in sorted(list(all_poi_types)):
        combined = ((osm_layers.get(amenity, 0) + proc_layers.get(amenity, 0)) > 0).astype(np.uint8)
        layer_index = len(final_layers)
        final_layers.append(combined)
        property_map[layer_index] = amenity
        poi_layers_stack.append(combined)

    # Final data layer: Animal density
    animal_density = _place_animal_density(valid_cells)
    final_layers.append(animal_density)
    property_map[len(final_layers) - 1] = GridLayer.ANIMAL_DENSITY
    
    final_grid = np.stack(final_layers, axis=-1)

    # Ensure residences are not placed directly on top of POIs
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