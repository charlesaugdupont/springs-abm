import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import contextily as cx

from config import SVEIRCONFIG
from abm.model.initialize_model import SVEIRModel
from abm.utils.rng import set_global_seed
from abm.constants import GridLayer


TITLES = {
    "water":            "Water Sampling Points",
    "place_of_worship": "Places of Worship",
    "school":           "Schools",
    "zoonotic_density": "Zoonotic Density\n(poultry + ruminant, weighted)",
}

SCATTER_COLOR = {
    "water":            "#2196F3",
    "place_of_worship": "#E53935",
    "school":           "#43A047",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_bounds(raw, grid_shape):
    """
    Bounds are always stored as (minx, miny, x_step, y_step) by grid_generator.py.
    Reconstruct (minx, miny, maxx, maxy) from that.
    """
    minx, miny, x_step, y_step = raw
    rows, cols = grid_shape[:2]
    maxx = minx + cols * x_step
    maxy = miny + rows * y_step
    return [minx, miny, maxx, maxy]


def _cell_to_geo(r, c, bounds, grid_shape):
    """Return the geographic centre (lon, lat) of grid cell (r, c)."""
    minx, miny, maxx, maxy = bounds
    rows, cols = grid_shape[:2]
    x_step = (maxx - minx) / cols
    y_step = (maxy - miny) / rows
    lon = minx + (c + 0.5) * x_step
    lat = miny + (r + 0.5) * y_step
    return lon, lat


def _add_osm_basemap(ax, bounds):
    if bounds is None:
        ax.text(0.5, 0.5, "No bounds available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False
    minx, miny, maxx, maxy = bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    try:
        cx.add_basemap(ax, crs="EPSG:4326",
                       source=cx.providers.OpenStreetMap.Mapnik)
        return True
    except Exception:
        ax.text(0.5, 0.5, "Map tile unavailable",
                ha="center", va="center", transform=ax.transAxes)
        return False


def _scatter_on_map(ax, layer_data, bounds, color, title, fig):
    ax.set_box_aspect(1)
    _add_osm_basemap(ax, bounds)

    if bounds is not None and layer_data is not None:
        minx, miny, maxx, maxy = bounds
        rows, cols = layer_data.shape
        x_step = (maxx - minx) / cols
        y_step = (maxy - miny) / rows
        x_coords = minx + (np.arange(cols) + 0.5) * x_step
        y_coords = miny + (np.arange(rows) + 0.5) * y_step

        yy, xx = np.where(layer_data > 0)
        geo_x = x_coords[xx]
        geo_y = y_coords[yy]

        ax.scatter(geo_x, geo_y, c=color, s=40, marker="o",
                   edgecolors="white", linewidths=0.3,
                   label=title, zorder=5, alpha=0.85)

    ax.set_title(title, pad=10, fontsize=24)
    ax.set_xlabel("Longitude", fontsize=20)
    ax.set_ylabel("Latitude", fontsize=20)

    dummy = ScalarMappable(cmap="cividis")
    dummy.set_array([])
    cbar = fig.colorbar(dummy, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_visible(False)


def _plot_continuous(ax, layer_data, title, fig, exclusion_mask=None):
    ax.set_box_aspect(1)
    im = ax.imshow(layer_data, cmap="plasma", origin="lower")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, pad=10, fontsize=24)

    if exclusion_mask is not None and np.any(exclusion_mask):
        # Outline cells excluded from residence placement (e.g. the river),
        # so it's visually obvious those cells can't host agents/animals.
        ax.contour(exclusion_mask.astype(float), levels=[0.5],
                   colors="cyan", linewidths=2.0)
        ax.plot([], [], color="cyan", linewidth=2.0,
               label="Excluded (natural water body)")
        ax.legend(loc="upper right", fontsize=10, framealpha=0.85)


# ---------------------------------------------------------------------------
# Zoonotic density layers (built dynamically, not read from grid.npz)
# ---------------------------------------------------------------------------

def build_zoonotic_density_layers(grid_id: str, seed: int | None = None, agents: int = 4000) -> dict:
    """
    Poultry/ruminant density are no longer baked into grid.npz - they depend
    on a specific run's stochastic household placement and ownership draw
    (abm/environment/animal_density.py), so they must be rebuilt by actually
    running agent + environment initialization.

    This runs SVEIRModel.initialize_model() (agent placement, household
    ownership draw, dynamic layer construction) but never calls .run(), so
    it's cheap relative to a full simulation. A temp directory is used for
    the model's config/log output and is not needed afterward.

    Returns a dict with 'poultry', 'ruminant', and 'combined' (the
    poultry_weight/ruminant_weight-combined layer actually used in the
    Campylobacter dose calculation), each a (rows, cols) numpy array.
    """
    cfg = SVEIRCONFIG.model_copy(deep=True)
    cfg.spatial_creation_args.grid_id = grid_id
    cfg.number_agents = agents
    if seed is not None:
        cfg.seed = seed
    set_global_seed(cfg.seed)

    model = SVEIRModel(
        model_identifier="_inspect_grid_tmp",
        root_path=os.path.join("outputs", "_inspect_grid_tmp"),
    )
    model.set_model_parameters(**cfg.model_dump())
    model.initialize_model(verbose=False)

    poultry = model.grid_environment.get_dynamic_layer("poultry_density")
    ruminant = model.grid_environment.get_dynamic_layer("ruminant_density")

    if poultry is None or ruminant is None:
        raise RuntimeError(
            "Poultry/ruminant density layers were not built. Is 'campy' "
            "included in cfg.pathogens?"
        )

    poultry_np = poultry.cpu().numpy()
    ruminant_np = ruminant.cpu().numpy()

    campy_cfg = next(p for p in cfg.pathogens if p.name == "campy")
    combined = np.clip(
        poultry_np * campy_cfg.poultry_weight + ruminant_np * campy_cfg.ruminant_weight,
        0.0, 1.0,
    )

    return {"poultry": poultry_np, "ruminant": ruminant_np, "combined": combined}


# ---------------------------------------------------------------------------
# POI utilities (unchanged)
# ---------------------------------------------------------------------------

def list_pois(grid_path, layer_name):
    """Print all POI locations for a given layer with their grid and geo coords."""
    data = np.load(grid_path, allow_pickle=True)
    grid_tensor  = data["grid"]
    property_map = data["property_map"].item()
    name_to_idx  = {v: k for k, v in property_map.items()}

    if layer_name not in name_to_idx:
        print(f"Layer '{layer_name}' not found. Available: {list(name_to_idx.keys())}")
        return

    bounds = _decode_bounds(data["bounds"], grid_tensor.shape) if "bounds" in data else None
    layer_data = grid_tensor[:, :, name_to_idx[layer_name]]
    rows, cols = np.where(layer_data > 0)

    print(f"\n{layer_name} — {len(rows)} point(s):")
    print(f"  {'#':<4}  {'row':>4}  {'col':>4}  {'lon':>10}  {'lat':>10}")
    print("  " + "-" * 40)
    for i, (r, c) in enumerate(zip(rows, cols)):
        if bounds is not None:
            lon, lat = _cell_to_geo(r, c, bounds, grid_tensor.shape)
            print(f"  {i:<4}  {r:>4}  {c:>4}  {lon:>10.5f}  {lat:>10.5f}")
        else:
            print(f"  {i:<4}  {r:>4}  {c:>4}  {'N/A':>10}  {'N/A':>10}")


def remove_poi(grid_path, layer_name, row, col):
    """Zero out a single POI cell in the given layer and re-save the .npz."""
    data = np.load(grid_path, allow_pickle=True)
    grid_tensor  = data["grid"].copy()
    property_map = data["property_map"].item()
    name_to_idx  = {v: k for k, v in property_map.items()}

    if layer_name not in name_to_idx:
        print(f"Layer '{layer_name}' not found. Available: {list(name_to_idx.keys())}")
        return

    idx = name_to_idx[layer_name]
    current = grid_tensor[row, col, idx]
    if current == 0:
        print(f"Cell ({row}, {col}) in layer '{layer_name}' is already 0 — nothing to remove.")
        return

    grid_tensor[row, col, idx] = 0
    print(f"Removed {layer_name} at cell ({row}, {col}).")

    save_kwargs = {"grid": grid_tensor, "property_map": data["property_map"]}
    if "bounds" in data:
        save_kwargs["bounds"] = data["bounds"]
    np.savez_compressed(grid_path, **save_kwargs)
    print(f"Saved updated grid to '{grid_path}'.")


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def inspect_grid(grid_id: str, seed: int | None = None, agents: int = 4000,
                  species_breakdown: bool = False):
    grid_path = os.path.join("grids", grid_id, "grid.npz")

    if not os.path.exists(grid_path):
        print(f"Error: Grid file not found at '{grid_path}'")
        return

    data = np.load(grid_path, allow_pickle=True)
    grid_tensor  = data["grid"]
    property_map = data["property_map"].item()
    name_to_idx  = {v: k for k, v in property_map.items()}

    bounds = _decode_bounds(data["bounds"], grid_tensor.shape) if "bounds" in data else None

    print("Building poultry/ruminant zoonotic density layers "
          f"(seed={seed if seed is not None else SVEIRCONFIG.seed}, agents={agents})...")
    density_layers = build_zoonotic_density_layers(grid_id, seed=seed, agents=agents)

    natural_water_idx = name_to_idx.get(GridLayer.NATURAL_WATER)
    natural_water_mask = (
        grid_tensor[:, :, natural_water_idx] if natural_water_idx is not None else None
    )
    if natural_water_mask is None:
        print("  - Note: no 'natural_water' layer found in this grid.npz. "
              "Regenerate with the updated grid_generator.py to see the "
              "river-exclusion overlay (older grids won't have this layer).")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    panel_map = [
        (0, 0, "water"),
        (0, 1, "place_of_worship"),
        (1, 0, "school"),
        (1, 1, "zoonotic_density"),
    ]

    for row, col, layer_name in panel_map:
        ax = axes[row, col]
        title = TITLES.get(layer_name, layer_name.replace("_", " ").title())

        if layer_name == "zoonotic_density":
            _plot_continuous(ax, density_layers["combined"], title, fig,
                             exclusion_mask=natural_water_mask)
            continue

        if layer_name not in name_to_idx:
            ax.text(0.5, 0.5, f"Layer '{layer_name}'\nnot found",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, pad=10, fontsize=22)
            ax.set_axis_off()
            continue

        layer_data = grid_tensor[:, :, name_to_idx[layer_name]]
        _scatter_on_map(ax, layer_data, bounds, SCATTER_COLOR[layer_name], title, fig)

    plt.tight_layout(pad=2.0)

    out_file = os.path.join("grids", grid_id, "grid_layers.jpg")
    plt.savefig(out_file, bbox_inches="tight", dpi=300, format="jpeg")
    print(f"Plot saved to '{out_file}'")
    plt.show()

    if species_breakdown:
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
        _plot_continuous(axes2[0], density_layers["poultry"], "Poultry Density (raw)", fig2,
                         exclusion_mask=natural_water_mask)
        _plot_continuous(axes2[1], density_layers["ruminant"], "Ruminant Density (raw)", fig2,
                         exclusion_mask=natural_water_mask)
        plt.tight_layout(pad=2.0)
        out_file2 = os.path.join("grids", grid_id, "zoonotic_species_breakdown.jpg")
        plt.savefig(out_file2, bbox_inches="tight", dpi=300, format="jpeg")
        print(f"Species breakdown plot saved to '{out_file2}'")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-id", type=str,
                        help="Grid ID to inspect. Defaults to the latest grid.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the household placement/ownership draw "
                             "used to build zoonotic density layers (default: config seed).")
    parser.add_argument("--agents", type=int, default=4000,
                        help="Number of agents to place when building density layers (default: 4000).")
    parser.add_argument("--species-breakdown", action="store_true",
                        help="Also save a separate figure with raw (unweighted) poultry and ruminant panels.")
    parser.add_argument("--list-pois", type=str, metavar="LAYER",
                        help="List all POI locations for a layer (e.g. 'school').")
    parser.add_argument("--remove-poi", type=str, metavar="LAYER",
                        help="Layer from which to remove a POI (e.g. 'school').")
    parser.add_argument("--row", type=int, help="Row index of the POI to remove.")
    parser.add_argument("--col", type=int, help="Col index of the POI to remove.")
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

    grid_path = os.path.join("grids", grid_id, "grid.npz")

    if args.list_pois:
        list_pois(grid_path, args.list_pois)
    elif args.remove_poi:
        if args.row is None or args.col is None:
            raise SystemExit("--remove-poi requires --row and --col.")
        remove_poi(grid_path, args.remove_poi, args.row, args.col)
    else:
        inspect_grid(grid_id, seed=args.seed, agents=args.agents,
                     species_breakdown=args.species_breakdown)