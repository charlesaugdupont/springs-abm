import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
import argparse
import contextily as cx

def inspect_grid(grid_id):
    """
    Loads and visualizes grid layers + OSM context.
    Improved handling of binary vs. continuous data for clearer interpretation.
    """
    grid_path = os.path.join("grids", grid_id, "grid.npz")

    if not os.path.exists(grid_path):
        print(f"Error: Grid file not found at '{grid_path}'")
        return

    data = np.load(grid_path, allow_pickle=True)
    grid_tensor = data['grid']
    
    # --- Bounds Logic ---
    bounds = None
    if 'bounds' in data:
        raw = data['bounds']
        if raw[2] < raw[0]: 
            rows, cols = grid_tensor.shape[0], grid_tensor.shape[1]
            minx, miny = raw[0], raw[1]
            maxx = minx + (cols * raw[2])
            maxy = miny + (rows * raw[3])
            bounds = [minx, miny, maxx, maxy]
        else:
            bounds = raw
    
    property_map = data['property_map'].item()
    num_layers = grid_tensor.shape[2]
    total_plots = num_layers + 1 if bounds is not None else num_layers

    cols_plot = int(np.ceil(np.sqrt(total_plots)))
    rows_plot = int(np.ceil(total_plots / cols_plot))
    
    fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(cols_plot * 5, rows_plot * 5))
    axes = axes.flatten()

    # Define a high-contrast discrete colormap for binary data: Dark Blue (0) and Gold (1)
    binary_cmap = ListedColormap(['#1a1c2c', '#f4b41a'])
    binary_norm = BoundaryNorm([0, 0.5, 1], binary_cmap.N)

    # --- 1. Plot Grid Layers ---
    for i in range(num_layers):
        ax = axes[i]
        ax.set_box_aspect(1)
        
        layer_name = property_map.get(i, f"Layer {i}")
        layer_data = grid_tensor[:, :, i]
        
        # Determine if data is binary or continuous
        is_binary = np.array_equal(layer_data, layer_data.astype(bool))
        
        if is_binary:
            im = ax.imshow(layer_data, cmap=binary_cmap, norm=binary_norm, origin='lower', interpolation='nearest')
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0.25, 0.75])
            cbar.ax.set_yticklabels(['Absent', 'Present'], rotation=90)
        else:
            # Continuous data (e.g. Animal Density)
            im = ax.imshow(layer_data, cmap='plasma', origin='lower')
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Density')
        
        title = layer_name.replace('_', ' ').title().replace("Residences", "Possible Households")
        ax.set_title(title, pad=10, fontsize=16)
        ax.set_xlabel(r"$x$", fontsize=12)
        ax.set_ylabel(r"$y$", rotation=0, labelpad=15, fontsize=12)

    # --- 2. Plot Natural OSM View ---
    if bounds is not None:
        ax_map = axes[num_layers]
        ax_map.set_box_aspect(1)
        
        minx, miny, maxx, maxy = bounds
        ax_map.set_xlim(minx, maxx)
        ax_map.set_ylim(miny, maxy)
        
        ax_map.set_title("Natural View", pad=10, fontsize=16)
        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")
        
        try:
            cx.add_basemap(ax_map, crs="EPSG:4326", source=cx.providers.OpenStreetMap.Mapnik)
        except Exception:
            ax_map.text(0.5, 0.5, "Map Tile Error", ha='center')

        # Dummy colorbar for alignment
        dummy = ScalarMappable(cmap='cividis')
        dummy.set_array([])
        cbar = fig.colorbar(dummy, ax=ax_map, fraction=0.046, pad=0.04)
        cbar.ax.set_visible(False) 

    for i in range(total_plots, len(axes)):
        axes[i].set_visible(False)
        
    plt.tight_layout()
    
    out_file = os.path.join("grids", grid_id, "grid_layers.pdf")
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Plot saved to '{out_file}'")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-id", type=str)
    args = parser.parse_args()
    
    grid_id = args.grid_id or (sorted([d for d in os.listdir("grids") if os.path.isdir(os.path.join("grids", d))])[-1] if os.path.exists("grids") else None)
    if not grid_id:
        raise Exception("No grid ID provided or found.")
    
    inspect_grid(grid_id)