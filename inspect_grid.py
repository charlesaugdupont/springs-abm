import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
import argparse
import contextily as cx


# ---------------------------------------------------------------------------
# Layout definition
# ---------------------------------------------------------------------------
# Row 0: water | place_of_worship | school
# Row 1: animal_density | natural view
#
# Keys must match the values in GridLayer / property_map exactly.
ROW0 = ["water", "place_of_worship", "school"]
ROW1 = ["animal_density", "_natural_view"]   # _natural_view is a special sentinel

TITLES = {
    "water":            "Water Sampling Points",
    "place_of_worship": "Places of Worship",
    "school":           "Schools",
    "animal_density":   "Animal Density",
    "_natural_view":    "Natural View",
}


def _plot_binary(ax, layer_data, title, fig):
    """High-contrast binary layer (absent / present)."""
    cmap = ListedColormap(["#1a1c2c", "#f4b41a"])
    norm = BoundaryNorm([0, 0.5, 1], cmap.N)
    im = ax.imshow(layer_data, cmap=cmap, norm=norm,
                   origin="lower", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(["Absent", "Present"], rotation=90, va="center")
    ax.set_title(title, pad=10, fontsize=14)
    ax.set_xlabel(r"$x$", fontsize=11)
    ax.set_ylabel(r"$y$", rotation=0, labelpad=15, fontsize=11)


def _plot_continuous(ax, layer_data, title, fig):
    """Plasma colormap for continuous data (e.g. animal density)."""
    im = ax.imshow(layer_data, cmap="plasma", origin="lower")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Density", fontsize=10)
    ax.set_title(title, pad=10, fontsize=14)
    ax.set_xlabel(r"$x$", fontsize=11)
    ax.set_ylabel(r"$y$", rotation=0, labelpad=15, fontsize=11)


def _plot_natural_view(ax, bounds, fig):
    """OSM basemap tile, or a placeholder if contextily fails."""
    if bounds is None:
        ax.text(0.5, 0.5, "No bounds available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    minx, miny, maxx, maxy = bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(TITLES["_natural_view"], pad=10, fontsize=14)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

    try:
        cx.add_basemap(ax, crs="EPSG:4326",
                       source=cx.providers.OpenStreetMap.Mapnik)
    except Exception:
        ax.text(0.5, 0.5, "Map tile unavailable",
                ha="center", va="center", transform=ax.transAxes)

    # Invisible colorbar to keep column widths consistent with other panels
    dummy = ScalarMappable(cmap="cividis")
    dummy.set_array([])
    cbar = fig.colorbar(dummy, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_visible(False)


def inspect_grid(grid_id: str):
    grid_path = os.path.join("grids", grid_id, "grid.npz")

    if not os.path.exists(grid_path):
        print(f"Error: Grid file not found at '{grid_path}'")
        return

    data = np.load(grid_path, allow_pickle=True)
    grid_tensor  = data["grid"]
    property_map = data["property_map"].item()          # {int_index: layer_name}
    name_to_idx  = {v: k for k, v in property_map.items()}

    # --- Resolve bounds ---
    bounds = None
    if "bounds" in data:
        raw = data["bounds"]
        if raw[2] < raw[0]:                             # stored as (minx, miny, x_step, y_step)
            rows = grid_tensor.shape[0]
            cols = grid_tensor.shape[1]
            minx, miny = raw[0], raw[1]
            maxx = minx + cols * raw[2]
            maxy = miny + rows * raw[3]
            bounds = [minx, miny, maxx, maxy]
        else:
            bounds = raw.tolist()

    # --- Build figure: 2 rows × 3 cols ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # ---- Row 0: water | place_of_worship | school ----
    for col_idx, layer_name in enumerate(ROW0):
        ax = axes[0, col_idx]
        ax.set_box_aspect(1)
        title = TITLES.get(layer_name, layer_name.replace("_", " ").title())

        if layer_name not in name_to_idx:
            ax.text(0.5, 0.5, f"Layer '{layer_name}'\nnot found",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, pad=10, fontsize=14)
            ax.set_axis_off()
            continue

        layer_data = grid_tensor[:, :, name_to_idx[layer_name]]
        _plot_binary(ax, layer_data, title, fig)

    # ---- Row 1: animal_density | natural_view | (empty) ----
    for col_idx, layer_name in enumerate(ROW1):
        ax = axes[1, col_idx]
        ax.set_box_aspect(1)

        if layer_name == "_natural_view":
            _plot_natural_view(ax, bounds, fig)
            continue

        title = TITLES.get(layer_name, layer_name.replace("_", " ").title())

        if layer_name not in name_to_idx:
            ax.text(0.5, 0.5, f"Layer '{layer_name}'\nnot found",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, pad=10, fontsize=14)
            ax.set_axis_off()
            continue

        layer_data = grid_tensor[:, :, name_to_idx[layer_name]]
        is_binary  = np.array_equal(layer_data, layer_data.astype(bool))
        if is_binary:
            _plot_binary(ax, layer_data, title, fig)
        else:
            _plot_continuous(ax, layer_data, title, fig)

    # Hide the unused third cell in row 1
    axes[1, 2].set_visible(False)

    plt.tight_layout()

    out_file = os.path.join("grids", grid_id, "grid_layers.pdf")
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Plot saved to '{out_file}'")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-id", type=str)
    args = parser.parse_args()

    grid_id = args.grid_id or (
        sorted([
            d for d in os.listdir("grids")
            if os.path.isdir(os.path.join("grids", d))
        ])[-1]
        if os.path.exists("grids") else None
    )

    if not grid_id:
        raise SystemExit("No grid ID provided or found in grids/.")

    inspect_grid(grid_id)
