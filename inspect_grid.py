import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def inspect_grid(grid_id):
    """Loads and visualizes the layers of a generated grid."""
    grid_path = os.path.join("grids", grid_id, "grid.npz")

    if not os.path.exists(grid_path):
        print(f"Error: Grid file not found at '{grid_path}'")
        return

    print(f"Loading grid from '{grid_path}'...")
    data = np.load(grid_path, allow_pickle=True)
    grid_tensor = data['grid']
    
    # The property map is stored in a 0-d array, so we extract the dictionary with .item()
    property_map = data['property_map'].item()
    
    # The map is already in the correct {index: name} format
    idx_to_name = property_map
    num_layers = grid_tensor.shape[2]
    
    print("\n--- Grid Summary ---")
    print(f"Shape (Height, Width, Layers): {grid_tensor.shape}")
    print("Layers Found:")
    for i in range(num_layers):
        print(f"  - Index {i}: {idx_to_name.get(i, 'Unknown')}")
    print("---------------------\n")

    # --- Visualization ---
    cols = int(np.ceil(np.sqrt(num_layers)))
    rows = int(np.ceil(num_layers / cols))
    fig, axes = plt.subplots(rows, cols, sharey=True, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i in range(num_layers):
        ax = axes[i]
        layer_name = idx_to_name.get(i, f"Layer {i}")
        layer_data = grid_tensor[:, :, i]

        # Use a different color map for the continuous animal density layer
        cmap = 'plasma' if 'density' in layer_name else 'cividis'
        
        im = ax.imshow(layer_data, cmap=cmap, origin='lower')
        ax.set_title(layer_name.replace('_', ' ').title().replace("Residences","Potential Households").replace("Water","Water Collection Points"))
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(num_layers, len(axes)):
        axes[i].set_visible(False)
        
    fig.suptitle(f"Grid Layers", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join("grids", grid_id, "layers.pdf"), bbox_inches="tight")
    print("Displaying grid plot.")
    plt.show()

if __name__ == "__main__":
    # --- Setup to read command-line arguments ---
    parser = argparse.ArgumentParser(description="Load and visualize a generated grid.")
    parser.add_argument("--grid-id", type=str, help="The unique ID of the grid to inspect.")
    args = parser.parse_args()
    grid_id = args.grid_id
    if not grid_id:
        raise Exception("--grid-id argument was not provided.")
    
    inspect_grid(args.grid_id)