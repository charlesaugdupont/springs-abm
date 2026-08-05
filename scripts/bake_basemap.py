"""
Dev-only, offline tool: fetches a static basemap tile image for the fixed
Akuse boundary and saves it for the webapp to use as a background image
under the spatial infection heatmap on the results page.

Run once locally whenever the basemap needs to be (re)generated. The output
PNG (+ a small JSON sidecar with its exact geographic extent) is committed
to the repo so the deployed webapp never needs contextily/osmnx/GDAL at
runtime - see webapp/simulation_runner.py and Finding #2 in the plan.

Usage (run as a module, like the experiments/ scripts, so the repo root
lands on sys.path and `abm`/`config` imports resolve):
    python -m scripts.bake_basemap
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abm.environment.grid_constants import AKUSE_BOUNDARY_COORDS

try:
    import contextily as cx
except ImportError as e:
    raise SystemExit(
        "contextily is required to bake the basemap (a dev-only dependency, "
        "not needed by the deployed webapp - see requirements.txt, not "
        "requirements-web.txt). pip install contextily."
    ) from e

OUT_DIR = os.path.join("webapp", "static", "img")
OUT_IMG = os.path.join(OUT_DIR, "akuse_basemap.png")
OUT_META = os.path.join(OUT_DIR, "akuse_basemap.json")
PIXELS = 1000  # square output image, in pixels

# Same fallback chain as run_viz.py's _add_basemap(), for consistency.
_TILE_PROVIDERS = [
    ("Esri.WorldImagery", lambda: cx.providers.Esri.WorldImagery),
    ("Stadia.StamenTerrain", lambda: cx.providers.Stadia.StamenTerrain),
    ("OpenStreetMap.Mapnik", lambda: cx.providers.OpenStreetMap.Mapnik),
]


def _grid_bounds():
    lons = [p[0] for p in AKUSE_BOUNDARY_COORDS]
    lats = [p[1] for p in AKUSE_BOUNDARY_COORDS]
    return min(lons), min(lats), max(lons), max(lats)


def main():
    minx, miny, maxx, maxy = _grid_bounds()

    fig = plt.figure(figsize=(PIXELS / 100, PIXELS / 100), dpi=100)
    # Axes fill the figure exactly (no margins/padding) so the saved PNG's
    # pixel grid maps linearly to [minx,maxx] x [miny,maxy] - required for
    # correct alignment with the heatmap trace overlaid on it client-side.
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.axis("off")

    used = None
    for name, provider_fn in _TILE_PROVIDERS:
        try:
            cx.add_basemap(ax, crs="EPSG:4326", source=provider_fn(), zoom=14)
            used = name
            break
        except Exception:
            continue
    if used is None:
        raise SystemExit("All tile providers failed - check network connectivity and try again.")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_IMG, dpi=100)
    plt.close(fig)

    with open(OUT_META, "w") as f:
        json.dump(
            {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy, "provider": used, "pixels": PIXELS},
            f, indent=2,
        )

    print(f"Basemap saved -> {OUT_IMG}  (provider: {used})")
    print(f"Extent metadata -> {OUT_META}")


if __name__ == "__main__":
    main()
