"""
Quick simulation run with two-panel output:
  1. Spatial infection heatmap overlaid on OSM basemap
  2. Under-5 epidemic curves (Rotavirus + Campylobacter)

Usage:
    python run_viz.py --grid-id <YOUR_GRID_ID>

Optional:
    --agents  N   number of agents (default: 3000)
    --steps   N   simulation steps / days (default: 150)
    --seed    N   random seed (default: 23)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

from config import SVEIRCONFIG
from abm.model.initialize_model import SVEIRModel
from abm.constants import AgentPropertyKeys
from abm.utils.rng import set_global_seed
from abm.environment.grid_generator import AKUSE_BOUNDARY_COORDS, GRID_SIZE

try:
    import contextily as cx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    print("contextily not found – install with: pip install contextily")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grid_bounds():
    lons = [p[0] for p in AKUSE_BOUNDARY_COORDS]
    lats = [p[1] for p in AKUSE_BOUNDARY_COORDS]
    return min(lons), min(lats), max(lons), max(lats)


def _build_infection_grid(model: SVEIRModel) -> np.ndarray:
    """Accumulate total infections per grid cell across all pathogens."""
    g = model.graph
    x = g.ndata[AgentPropertyKeys.X].cpu().numpy().astype(int)
    y = g.ndata[AgentPropertyKeys.Y].cpu().numpy().astype(int)
    x = np.clip(x, 0, GRID_SIZE - 1)
    y = np.clip(y, 0, GRID_SIZE - 1)

    heat = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    for p in model.pathogens:
        counts = g.ndata[AgentPropertyKeys.num_infections(p.name)].cpu().numpy()
        np.add.at(heat, (y, x), counts)
    return heat


def _make_transparent_cmap(base_name: str = "hot") -> mcolors.LinearSegmentedColormap:
    """
    Returns a colormap where vmin → alpha=0 (fully transparent) and
    vmax → alpha=1 (fully opaque), so the basemap shows through in
    low-infection areas.  A power curve on alpha keeps mid-range cells
    semi-transparent rather than jumping straight to opaque.
    """
    base   = mcm.get_cmap(base_name, 256)
    colors = base(np.linspace(0, 1, 256))
    alphas = np.linspace(0, 1, 256) ** 0.5   # power < 1 → gentler ramp
    colors[:, 3] = alphas
    return mcolors.LinearSegmentedColormap.from_list(f"{base_name}_alpha", colors)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    cfg = SVEIRCONFIG.model_copy(deep=True)
    cfg.number_agents = args.agents
    cfg.step_target   = args.steps
    cfg.seed          = args.seed
    cfg.spatial_creation_args.grid_id = args.grid_id

    set_global_seed(cfg.seed)

    model = SVEIRModel(model_identifier="viz_run", root_path="outputs/viz")
    model.set_model_parameters(**cfg.model_dump())
    model.initialize_model(verbose=True)
    model.run(verbose=False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

# Tile providers tried in order — first one that succeeds is used.
# Esri.WorldImagery  → satellite photo (greens, blues, browns of real terrain)
# Stadia.StamenTerrain → stylised terrain with natural colours
# OpenStreetMap.Mapnik → familiar OSM colours (green parks, blue water)
_TILE_PROVIDERS = [
    ("Esri.WorldImagery",      lambda: cx.providers.Esri.WorldImagery),
    ("Stadia.StamenTerrain",   lambda: cx.providers.Stadia.StamenTerrain),
    ("OpenStreetMap.Mapnik",   lambda: cx.providers.OpenStreetMap.Mapnik),
]


def _add_basemap(ax, minx, miny, maxx, maxy, zoom: int = 14):
    """Try each tile provider in order; fall back to a plain green background."""
    if not HAS_CONTEXTILY:
        ax.set_facecolor("#2d4a2d")
        return

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    for name, provider_fn in _TILE_PROVIDERS:
        try:
            cx.add_basemap(ax, crs="EPSG:4326", source=provider_fn(), zoom=zoom)
            print(f"  Basemap: {name}")
            return
        except Exception:
            continue

    # All providers failed — use a plain terrain-coloured background
    ax.set_facecolor("#4a7c4a")
    print("  Basemap: fallback (no tiles available)")


def plot(model: SVEIRModel, steps: int,
         out_path: str = "outputs/viz/simulation_overview.png"):

    from scipy.ndimage import gaussian_filter

    minx, miny, maxx, maxy = _grid_bounds()

    heat       = _build_infection_grid(model)
    rota_prev  = np.array(model.u5_prevalence_history.get("rota",  []))
    campy_prev = np.array(model.u5_prevalence_history.get("campy", []))
    days       = np.arange(len(rota_prev))

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.patch.set_facecolor("#1a1a2e")
    axes[1].set_facecolor("#1a1a2e")

    # ── Panel 1: Spatial infection heatmap ───────────────────────────────────
    ax1 = axes[0]
    _add_basemap(ax1, minx, miny, maxx, maxy, zoom=14)

    # Smooth + power-normalise the infection grid
    heat_smooth = gaussian_filter(heat.astype(float), sigma=2.0)
    vmax = heat_smooth.max()
    heat_norm = (heat_smooth / vmax) ** 0.4 if vmax > 0 else heat_smooth

    # "hot" cmap: black → red → yellow → white, with transparent background.
    # Works well over both satellite and terrain tiles.
    cmap_alpha = _make_transparent_cmap("hot")

    ax1.imshow(
        heat_norm,
        extent=[minx, maxx, miny, maxy],
        origin="lower",
        cmap=cmap_alpha,
        vmin=0, vmax=1,
        aspect="auto",
        zorder=2,
        interpolation="bilinear",
    )

    # Colorbar mapped back to raw infection counts
    sm = plt.cm.ScalarMappable(
        cmap="hot",
        norm=mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=vmax if vmax > 0 else 1),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, fraction=0.035, pad=0.02)
    cbar.set_label("Total infections (all pathogens)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax1.set_title("Spatial Infection Risk", color="white", fontsize=24, pad=10)
    ax1.set_xlabel("Longitude (°E)", color="white", fontsize=16)
    ax1.set_ylabel("Latitude (°N)",  color="white", fontsize=16)
    ax1.tick_params(axis="both", labelsize=14)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#555")

    # ── Panel 2: Under-5 epidemic curves ─────────────────────────────────────
    ax2 = axes[1]

    ROTA_COL  = "#42a5f5"
    CAMPY_COL = "#ef5350"

    if len(rota_prev) > 0:
        ax2.plot(days, rota_prev * 100,
                 color=ROTA_COL, linewidth=2.2, label="Rotavirus", zorder=3)
        ax2.fill_between(days, 0, rota_prev * 100, color=ROTA_COL, alpha=0.15)

    if len(campy_prev) > 0:
        ax2.plot(days, campy_prev * 100,
                 color=CAMPY_COL, linewidth=2.2, label="Campylobacter", zorder=3)
        ax2.fill_between(days, 0, campy_prev * 100, color=CAMPY_COL, alpha=0.15)

    if len(rota_prev) > 0 and len(campy_prev) > 0:
        min_len  = min(len(rota_prev), len(campy_prev))
        combined = rota_prev[:min_len] + campy_prev[:min_len]
        ax2.plot(np.arange(min_len), combined * 100,
                 color="white", linewidth=1.2, linestyle="--",
                 alpha=0.5, label="Combined", zorder=2)

    ax2.set_title("Under 5 Epidemic Curves", color="white", fontsize=24, pad=10)
    ax2.set_xlabel("Simulation Day", color="white", fontsize=16)
    ax2.set_ylabel("% of Under 5s Infectious", color="white", fontsize=16)
    ax2.tick_params(colors="white")
    ax2.tick_params(axis="both", labelsize=14)
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(0, max(len(rota_prev), len(campy_prev), 1) - 1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax2.legend(fontsize=10, facecolor="#2d2d44",
               edgecolor="white", labelcolor="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#555577")
    ax2.grid(True, color="#333", lw=0.5, ls="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nFigure saved → {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick simulation + visualisation")
    parser.add_argument("-g", "--grid-id", required=True,
                        help="Grid ID from 'python main.py create-grid'")
    parser.add_argument("-n", "--agents", type=int, default=3000)
    parser.add_argument("-s", "--steps",  type=int, default=150)
    parser.add_argument("--seed",         type=int, default=1)
    parser.add_argument("-o", "--output", type=str, default="outputs/viz/simulation_overview.png")
    args = parser.parse_args()

    model = run(args)
    plot(model, args.steps, args.output)
