# environment/grid_generator.py

import os
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon, box
import random
import json
import hashlib

# ==============================================================================
# --- GRID GENERATION CONFIGURATION ---
# ==============================================================================

# --- GEOGRAPHIC AND GRID PARAMETERS ---
AKUSE_CENTER_POINT = (6.100635676379143, 0.12736045405150456)
AKUSE_BOUNDARY_COORDS = [
    (0.11640835827890646, 6.107983818842101),
    (0.13688109260416578, 6.106918939321614),
    (0.13518748441180858, 6.095601368066649),
    (0.11628382826476255, 6.090648307705022)
]
GRID_SIZE = 75
POI_FETCH_RADIUS = 1500

# --- POINTS OF INTEREST (POI) CONFIGURATION ---

# 1. POIs to fetch from OpenStreetMap (OSM)
OSM_POI_TAGS = {
    "amenity": ["school", "place_of_worship"]
}

# 2. Procedurally generated POIs to add to the map.
#    This enriches the environment and prevents agent pile-ups.
#    The key is the 'property_name' that will be used in the model.
PROCEDURAL_POI_COUNTS = {
    "school": 5,
    "place_of_worship": 5,
    "water": 20
}

# ==============================================================================
# --- HELPER FUNCTIONS ---
# ==============================================================================

def get_grid_id(boundary_coords, grid_size, osm_tags, procedural_counts):
    """Creates a unique, deterministic hash from all grid generation parameters."""
    params = {
        "boundary_coords": boundary_coords,
        "grid_size": grid_size,
        "osm_poi_tags": osm_tags,
        "procedural_poi_counts": procedural_counts,
    }
    params_string = json.dumps(params, sort_keys=True, indent=None)
    return hashlib.sha256(params_string.encode('utf-8')).hexdigest()[:12]

def _initialize_grid_and_boundary():
    """Defines the geographical boundary and creates the base valid cells mask."""
    print("1. Defining grid boundary for Akuse...")
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
    return boundary, valid_cells_mask, (minx, miny, x_step, y_step)

def _place_osm_pois(tags, boundary_info):
    """Fetches and places POIs from OpenStreetMap onto grid layers."""
    print("2. Fetching real-world POIs from OpenStreetMap...")
    minx, miny, x_step, y_step = boundary_info
    
    try:
        pois_gdf = ox.features_from_point(AKUSE_CENTER_POINT, tags, dist=POI_FETCH_RADIUS)
    except ox._errors.InsufficientResponse:
        print("  - Warning: OSM did not return any data. Skipping real POIs.")
        return [], np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    osm_layers = {}
    for tag_key, tag_values in tags.items():
        for amenity_type in tag_values:
            if amenity_type not in osm_layers:
                osm_layers[amenity_type] = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

            locations = pois_gdf[pois_gdf[tag_key] == amenity_type]
            print(f"  - Placing {len(locations)} real '{amenity_type}' locations from OSM.")
            for _, poi in locations.iterrows():
                centroid = poi['geometry'].centroid if isinstance(poi['geometry'], Polygon) else poi['geometry']
                c = int((centroid.x - minx) / x_step)
                r = int((centroid.y - miny) / y_step)
                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                    osm_layers[amenity_type][r, c] = 1
    
    all_osm_locations_mask = np.sum(list(osm_layers.values()), axis=0) > 0
    return osm_layers, all_osm_locations_mask

def _place_procedural_pois(counts, valid_cells_mask, occupied_mask):
    """Places randomly located procedural POIs onto grid layers."""
    print("3. Placing procedurally generated POIs...")
    procedural_layers = {}
    
    # Create a mutable copy of the occupied mask to update within the loop
    current_occupied_mask = occupied_mask.copy()

    for amenity_type, num_to_add in counts.items():
        if num_to_add <= 0:
            continue
            
        print(f"  - Adding {num_to_add} procedural '{amenity_type}' locations.")
        procedural_layers[amenity_type] = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        
        # Find cells that are valid but not yet occupied by ANY POI
        available_cells = np.argwhere(valid_cells_mask & ~current_occupied_mask)
        
        if len(available_cells) >= num_to_add:
            selected_indices = random.sample(range(len(available_cells)), num_to_add)
            new_poi_cells = available_cells[selected_indices]
            
            # Place new POIs on the layer and update the master occupied mask
            procedural_layers[amenity_type][new_poi_cells[:, 0], new_poi_cells[:, 1]] = 1
            current_occupied_mask[new_poi_cells[:, 0], new_poi_cells[:, 1]] = True
        else:
            print(f"    Warning: Not enough available cells ({len(available_cells)}) to place {num_to_add} {amenity_type}s.")
            
    return procedural_layers

# ==============================================================================
# --- MAIN FUNCTION ---
# ==============================================================================

def create_and_save_realistic_grid():
    """
    Fetches real-world geographic data and enriches it with procedurally generated
    locations to create a realistic base grid for the simulation.
    """
    grid_id = get_grid_id(
        boundary_coords=AKUSE_BOUNDARY_COORDS,
        grid_size=GRID_SIZE,
        osm_tags=OSM_POI_TAGS,
        procedural_counts=PROCEDURAL_POI_COUNTS
    )

    # Step 1: Initialize the base grid
    _, valid_cells_mask, boundary_info = _initialize_grid_and_boundary()

    # Step 2: Place POIs from OpenStreetMap
    osm_layers, osm_occupied_mask = _place_osm_pois(OSM_POI_TAGS, boundary_info)
    
    # Step 3: Place procedural POIs, ensuring they don't overlap with OSM POIs
    procedural_layers = _place_procedural_pois(PROCEDURAL_POI_COUNTS, valid_cells_mask, osm_occupied_mask)

    # Step 4: Combine all layers and create the final property map
    print("4. Combining all layers...")
    final_layers = []
    property_map = {}
    
    # Start with the residence layer
    final_layers.append(valid_cells_mask.astype(np.uint8))
    property_map[0] = "residences"
    
    # Combine all POI layers (OSM and procedural)
    all_poi_types = set(osm_layers.keys()) | set(procedural_layers.keys())
    
    for amenity_type in sorted(list(all_poi_types)): # Sort for consistent order
        osm_layer = osm_layers.get(amenity_type, 0)
        proc_layer = procedural_layers.get(amenity_type, 0)
        combined_layer = ((osm_layer + proc_layer) > 0).astype(np.uint8)
        
        layer_index = len(final_layers)
        final_layers.append(combined_layer)
        property_map[layer_index] = amenity_type

    # Step 5: Finalize and save the grid
    final_grid = np.stack(final_layers, axis=-1)
    
    # Ensure residences are not placed directly on top of ANY POI
    all_poi_mask = np.sum(final_grid[:, :, 1:], axis=2) > 0
    final_grid[:, :, 0][all_poi_mask] = 0

    print(f"5. Saving base grid and metadata...")
    grid_dir = os.path.join("grids", grid_id)
    os.makedirs(grid_dir, exist_ok=True)
    output_path = os.path.join(grid_dir, "grid.npz")
    
    np.savez_compressed(
        output_path,
        grid=final_grid,
        bounds=np.array(boundary_info[:2] + (boundary_info[0]+GRID_SIZE*boundary_info[2], boundary_info[1]+GRID_SIZE*boundary_info[3])),
        property_map=property_map
    )
    print(f"\n--- Grid Generation Complete ---")
    print(f"  Grid ID: {grid_id}")
    print(f"  Saved to: {output_path}")
    print("\nUse this ID for the 'simulate' stage:")
    print(f"  uv run main.py simulate --grid-id {grid_id} --policy-set-id <your_policy_id>")

if __name__ == '__main__':
    create_and_save_realistic_grid()