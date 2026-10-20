# abm/environment/water_network.py
"""
Catchment water model: drinking-water sources, faecal runoff and exposure.

Replaces the original point-source water model (WaterConfig.model="legacy"),
in which the only water sources were 11 sampling points, a source became
contaminated only when an infectious person stood at it, and only the people
who visited a source that day - or whose household fetched from it - were
exposed. Three problems with that, all raised in review:

  * Contamination of drinking-water sources in a setting like Akuse comes
    from faecal material reaching the water (runoff from open defecation and
    unsafe child-stool disposal, seepage from pits into shallow wells), not
    from people visiting the source.
  * Exposure should scale with how contaminated the water is, which needs a
    continuous concentration for every pathogen (rotavirus had a binary
    clean/contaminated flag).
  * Most households in the Akuse area do not fetch from the sampled points at
    all: 64.5% drink piped water and 13.7% sachet/packaged water, while the
    15.7% who drink surface water use the extensive canal network and the
    Volta, not only the seven sampled spots.

Structure
---------
Sources. Surface water comes from the grid's natural-water layer (OSM rivers,
canals, streams, water bodies). It is classed into RIVER (wide features that
survive a morphological opening) and CANAL (thin features), and tiled into
REACHES (one per block of reach_block_cells x reach_block_cells cells and
class). Groundwater sources are the wells and boreholes in the SPRINGS
water-sampling file. Two pseudo-sources complete the set: the PIPED network
(no local loading) and PACKAGED water (never contaminated).

Households. Each household is given a primary drinking-water source type
(WaterConfig shares, from DHS/MICS clusters near Akuse), with users of
surface water and groundwater drawn preferentially from households near
those sources, and a sanitation type (same survey). A surface user drinks
from the reach containing its nearest water cell; a groundwater user from its
nearest well/borehole. Collectors of surface and groundwater users go to that
source on WATER days; piped and packaged users fetch at home.

Daily dynamics, per waterborne pathogen p and source s
    C[s] <- C[s] * exp(-k_p) + L[s] (+ shock pulse)
  where k_p is the pathogen's environmental die-off rate and the loading L
  comes from every infectious agent i in household h:
    surface reach s:  r_i * exp(-d_h / runoff_length) * (share of h's runoff
                      going to reach s, by proximity among nearby reaches)
    well/borehole s:  g_i * exp(-d_hs / groundwater_length) [* borehole factor]
  r_i, g_i are the sanitation-dependent release weights (a child's stools are
  additionally disposed of unsafely with probability child_stool_unsafe_frac).
  Units are infectious-person-days of shedding reaching the source; each
  pathogen's water_dose_scale converts them to a dose.

Exposure. Every member of a household drinks from its source every day:
  P(infection) = pathogen.water_infection_prob(dilution_s * C[s])
times the agent's relative susceptibility on the "water" route (which
includes the household environmental-exposure multiplier).

What is NOT represented (documented limitations): flow routing along the canal
network (each reach is well mixed and keeps its own memory); seasonality of
runoff (loading is the same every day; only shocks pulse it); point-of-use
treatment and storage recontamination; animal faecal loading (a lever is left
in place via WaterNetwork.animal_loading, zero by default).
"""
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
import torch

from abm.constants import AgentPropertyKeys, Compartment, GridLayer

# Source kinds (per real source).
CANAL, RIVER, WELL, BOREHOLE = 0, 1, 2, 3
# Household primary-source types.
HH_PIPED, HH_SURFACE, HH_GROUND, HH_PACKAGED = 0, 1, 2, 3
# Household sanitation types (order matches WaterConfig.sanitation_shares()).
SAN_IMPROVED_ONSITE, SAN_SEWER, SAN_UNIMPROVED, SAN_OPEN = 0, 1, 2, 3

# Package-relative (not working-directory-relative), so the network is the
# same whichever directory the model is run from. abm/data/ is gitignored:
# see load_groundwater_points for what happens when the file is absent.
WATER_SAMPLING_CSV = Path(__file__).resolve().parents[1] / "data" / "water_sampling_points.csv"
_METERS_PER_DEG_LAT = 110_574.0
_METERS_PER_DEG_LON_EQ = 111_320.0


def _dms_to_decimal(dms) -> float | None:
    if dms is None or (isinstance(dms, float) and math.isnan(dms)):
        return None
    m = re.search(r"(\d+)\D+(\d+)\D+([0-9.]+)", str(dms))
    if not m:
        return None
    return float(m.group(1)) + float(m.group(2)) / 60.0 + float(m.group(3)) / 3600.0


def load_groundwater_points(bounds, grid_shape) -> list[tuple[int, int, int]]:
    """(row, col, kind) of the Akuse wells and boreholes in the sampling CSV.

    Read from the same file the grid generator places water points from, but
    keeping the source TYPE, which the grid's binary water layer does not.
    Standpipes (piped supply) and surface-water points are not groundwater
    and are skipped: surface water comes from the natural-water layer.

    The CSV is project data under the gitignored abm/data/, so a checkout or
    a deployment built from git does not have it. Without it there are no
    wells or boreholes: the ~6% of households assigned groundwater by the
    survey shares get piped water instead, and the random stream shifts, so
    results differ from the calibrated model - hence the warning rather than
    a silent fallback.
    """
    if not WATER_SAMPLING_CSV.exists():
        warnings.warn(
            f"{WATER_SAMPLING_CSV} not found: the catchment water model will have no wells "
            "or boreholes, and groundwater households will drink piped water instead. Results "
            "will differ from the calibrated model (which had 2 wells and 1 borehole).",
            RuntimeWarning, stacklevel=2,
        )
        return []
    import pandas as pd

    df = pd.read_csv(WATER_SAMPLING_CSV, encoding="latin-1")
    df = df[df["site"].astype(str).str.contains("Akuse", case=False, na=False)]
    minx, miny, x_step, y_step = (float(v) for v in bounds)
    out, seen = [], set()
    for _, row in df.iterrows():
        kind_raw = str(row.get("Water source type", "")).strip().lower()
        if kind_raw == "well":
            kind = WELL
        elif kind_raw == "borehole":
            kind = BOREHOLE
        else:
            continue
        lon, lat = _dms_to_decimal(row["Longitude"]), _dms_to_decimal(row["Latitude"])
        if lon is None or lat is None:
            continue
        c, r = int((lon - minx) / x_step), int((lat - miny) / y_step)
        if not (0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]) or (r, c) in seen:
            continue
        seen.add((r, c))
        out.append((r, c, kind))
    return out


class WaterNetwork:
    """Sources, household assignments, loading kernels and concentrations.

    Built once per run by WaterNetwork.build() (after agents are placed);
    advanced once per day by EnvironmentSystem (loading, die-off, shocks) and
    read once per day by HouseholdSystem (exposure). See the module docstring.
    """

    def __init__(self):
        self.source_kind: torch.Tensor          # (S,) CANAL/RIVER/WELL/BOREHOLE
        self.source_access: torch.Tensor        # (S, 2) (row, col) collectors go to
        self.dilution: torch.Tensor             # (S + 2,) incl. piped, packaged
        self.hh_type: torch.Tensor              # (H,) HH_*
        self.hh_sanitation: torch.Tensor        # (H,) SAN_*
        self.hh_source: torch.Tensor            # (H,) index into the S + 2 sources
        self.agent_hh: torch.Tensor             # (N,) household index 0..H-1
        self.surface_kernel: torch.Tensor       # (H, S) runoff fraction to each reach
        self.ground_kernel: torch.Tensor        # (H, S) seepage to each well/borehole
        self.agent_surface_release: torch.Tensor  # (N,)
        self.agent_ground_release: torch.Tensor   # (N,)
        self.concentration: dict[str, torch.Tensor] = {}
        self.animal_loading: dict[str, torch.Tensor] = {}
        self.last_loading: dict[str, torch.Tensor] = {}
        self.cell_size_m: tuple[float, float] = (1.0, 1.0)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @property
    def num_sources(self) -> int:
        return int(self.source_kind.numel())

    @property
    def piped_index(self) -> int:
        return self.num_sources

    @property
    def packaged_index(self) -> int:
        return self.num_sources + 1

    @classmethod
    def build(cls, grid, agent_state, water_cfg, device="cpu") -> "WaterNetwork":
        from scipy import ndimage

        net = cls()
        rows, cols = int(grid.grid_shape[0]), int(grid.grid_shape[1])
        bounds = getattr(grid, "bounds", None)
        if bounds is not None:
            minx, miny, x_step, y_step = (float(v) for v in bounds)
            lat_mid = miny + 0.5 * y_step * rows
            dy = y_step * _METERS_PER_DEG_LAT
            dx = x_step * _METERS_PER_DEG_LON_EQ * math.cos(math.radians(lat_mid))
        else:
            dy = dx = 55.0
        net.cell_size_m = (dy, dx)

        # --- surface reaches from the natural-water layer ---
        nat = None
        if GridLayer.NATURAL_WATER in grid.property_to_index:
            nat = grid.get_slice(GridLayer.NATURAL_WATER).cpu().numpy() > 0
        if nat is None:
            nat = np.zeros((rows, cols), dtype=bool)
        w = int(water_cfg.river_min_width_cells)
        river = ndimage.binary_opening(nat, structure=np.ones((w, w), dtype=bool))
        canal = nat & ~river
        b = int(water_cfg.reach_block_cells)
        block_id = (np.arange(rows)[:, None] // b) * ((cols + b - 1) // b) + (np.arange(cols)[None, :] // b)

        reach_cells: list[np.ndarray] = []
        kinds: list[int] = []
        for kind, mask in ((CANAL, canal), (RIVER, river)):
            for blk in np.unique(block_id[mask]):
                cells = np.argwhere(mask & (block_id == blk))
                reach_cells.append(cells)
                kinds.append(kind)
        n_reach = len(reach_cells)

        # --- groundwater points ---
        gw = load_groundwater_points(bounds, (rows, cols)) if bounds is not None else []
        n_gw = len(gw)
        S = n_reach + n_gw
        kinds += [k for _, _, k in gw]

        # --- households ---
        hh_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
        unique_hh, agent_hh = torch.unique(hh_ids, return_inverse=True)
        H = int(unique_hh.numel())
        first = torch.full((H,), -1, dtype=torch.long)
        first.scatter_reduce_(0, agent_hh, torch.arange(len(agent_hh)), reduce="amin", include_self=False)
        home = agent_state.ndata[AgentPropertyKeys.HOME_LOCATION][first].cpu().numpy()   # (H, 2) row, col
        home_m = home * np.array([dy, dx])

        # Distance from every household to every surface cell, then per reach.
        all_cells = np.concatenate(reach_cells) if n_reach else np.zeros((0, 2), dtype=int)
        cell_reach = np.concatenate([np.full(len(c), i) for i, c in enumerate(reach_cells)]) if n_reach else np.zeros(0, dtype=int)
        if n_reach:
            d_cells = np.hypot(home_m[:, None, 0] - all_cells[None, :, 0] * dy,
                               home_m[:, None, 1] - all_cells[None, :, 1] * dx)   # (H, n_cells)
            # Cells are concatenated reach by reach, so each reach is a
            # contiguous column block.
            offsets = np.concatenate([[0], np.cumsum([len(c) for c in reach_cells])[:-1]])
            d_reach = np.minimum.reduceat(d_cells, offsets, axis=1)
            nearest_cell = d_cells.argmin(axis=1)
            d_surface = d_cells.min(axis=1)
        else:
            d_reach = np.zeros((H, 0))
            nearest_cell = np.zeros(H, dtype=int)
            d_surface = np.full(H, np.inf)
        if n_gw:
            gw_cells = np.array([(r, c) for r, c, _ in gw], dtype=float)
            d_gw = np.hypot(home_m[:, None, 0] - gw_cells[None, :, 0] * dy,
                            home_m[:, None, 1] - gw_cells[None, :, 1] * dx)       # (H, n_gw)
        else:
            d_gw = np.zeros((H, 0))
        d_gw_min = d_gw.min(axis=1) if n_gw else np.full(H, np.inf)

        # --- source-type assignment (the only RNG this module uses) ---
        p_piped, p_surface, p_ground, p_packaged = water_cfg.source_shares()
        hh_type = torch.full((H,), HH_PIPED, dtype=torch.long)
        available = torch.ones(H, dtype=torch.bool)
        lam = float(water_cfg.source_choice_length_m)

        def _draw(n, dist, type_code):
            n = min(int(round(n)), int(available.sum()))
            if n <= 0:
                return
            weight = torch.exp(-torch.as_tensor(dist, dtype=torch.float64) / lam)
            weight = torch.where(torch.isfinite(weight), weight, torch.zeros_like(weight))
            weight = weight * available.double()
            if float(weight.sum()) <= 0:
                weight = available.double()
            chosen = torch.multinomial(weight, n, replacement=False)
            hh_type[chosen] = type_code
            available[chosen] = False

        if n_reach:
            _draw(p_surface * H, d_surface, HH_SURFACE)
        if n_gw:
            _draw(p_ground * H, d_gw_min, HH_GROUND)
        _draw(p_packaged * H, np.zeros(H), HH_PACKAGED)
        net.hh_type = hh_type

        san_p = torch.tensor(water_cfg.sanitation_shares(), dtype=torch.float64)
        net.hh_sanitation = torch.multinomial(san_p, H, replacement=True)

        # --- each household's specific source ---
        hh_source = torch.full((H,), S, dtype=torch.long)                 # piped
        hh_source[hh_type == HH_PACKAGED] = S + 1
        if n_reach:
            surf = (hh_type == HH_SURFACE).numpy()
            hh_source[torch.from_numpy(surf)] = torch.from_numpy(cell_reach[nearest_cell[surf]]).long()
        if n_gw:
            grd = (hh_type == HH_GROUND).numpy()
            hh_source[torch.from_numpy(grd)] = torch.from_numpy(n_reach + d_gw[grd].argmin(axis=1)).long()
        net.hh_source = hh_source

        # Access cell of each real source: for a reach, the cell nearest to
        # each of its users (computed per household below); for groundwater,
        # the point itself.
        access = np.zeros((S, 2), dtype=float)
        for i, cells in enumerate(reach_cells):
            access[i] = cells[len(cells) // 2]
        for j, (r, c, _) in enumerate(gw):
            access[n_reach + j] = (r, c)
        net.source_access = torch.from_numpy(access).float()
        net.source_kind = torch.tensor(kinds, dtype=torch.long)
        net.hh_water_location = torch.from_numpy(home.astype(np.float32)).clone()
        if n_reach:
            surf_idx = np.nonzero(hh_type.numpy() == HH_SURFACE)[0]
            net.hh_water_location[surf_idx] = torch.from_numpy(all_cells[nearest_cell[surf_idx]].astype(np.float32))
        if n_gw:
            grd_idx = np.nonzero(hh_type.numpy() == HH_GROUND)[0]
            net.hh_water_location[grd_idx] = torch.from_numpy(
                gw_cells[d_gw[grd_idx].argmin(axis=1)].astype(np.float32)
            )

        # --- loading kernels ---
        # Surface: fraction of a household's release reaching water at all,
        # exp(-d_nearest / L), split among reaches within 3L of the household
        # in proportion to exp(-d_reach / L) (mass-conserving).
        L = float(water_cfg.runoff_length_m)
        if n_reach:
            prox = np.exp(-d_reach / L) * (d_reach <= 3 * L)
            denom = prox.sum(axis=1, keepdims=True)
            share = np.divide(prox, denom, out=np.zeros_like(prox), where=denom > 0)
            surface_kernel = share * np.exp(-d_surface / L)[:, None]
        else:
            surface_kernel = np.zeros((H, 0))
        Lg = float(water_cfg.groundwater_length_m)
        if n_gw:
            gk = np.exp(-d_gw / Lg) * (d_gw <= 4 * Lg)
            gk[:, np.array([k == BOREHOLE for _, _, k in gw])] *= float(water_cfg.borehole_protection)
        else:
            gk = np.zeros((H, 0))
        net.surface_kernel = torch.zeros((H, S), dtype=torch.float32)
        net.surface_kernel[:, :n_reach] = torch.from_numpy(surface_kernel).float()
        net.ground_kernel = torch.zeros((H, S), dtype=torch.float32)
        net.ground_kernel[:, n_reach:] = torch.from_numpy(gk).float()

        # --- per-agent release weights ---
        surface_w = torch.tensor([
            water_cfg.surface_release_improved_onsite, water_cfg.surface_release_sewer,
            water_cfg.surface_release_unimproved, water_cfg.surface_release_open_defecation,
        ], dtype=torch.float32)
        ground_w = torch.tensor([
            water_cfg.ground_release_improved_onsite, water_cfg.ground_release_sewer,
            water_cfg.ground_release_unimproved, water_cfg.ground_release_open_defecation,
        ], dtype=torch.float32)
        san_agent = net.hh_sanitation[agent_hh]
        is_child = agent_state.ndata[AgentPropertyKeys.IS_CHILD].bool().cpu()
        u = float(water_cfg.child_stool_unsafe_frac)
        rs, rg = surface_w[san_agent], ground_w[san_agent]
        net.agent_surface_release = torch.where(
            is_child, u * water_cfg.surface_release_open_defecation + (1 - u) * rs, rs)
        net.agent_ground_release = torch.where(
            is_child, u * water_cfg.ground_release_open_defecation + (1 - u) * rg, rg)
        net.agent_hh = agent_hh.cpu()

        dil = torch.ones(S + 2, dtype=torch.float32)
        dil[:S][net.source_kind == RIVER] = float(water_cfg.river_dilution)
        dil[S + 1] = 0.0
        net.dilution = dil
        net.water_cfg = water_cfg
        net._n_reach = n_reach
        return net

    # ------------------------------------------------------------------
    # Per-pathogen state
    # ------------------------------------------------------------------

    def init_pathogen(self, name: str):
        S = self.num_sources
        self.concentration[name] = torch.zeros(S + 2, dtype=torch.float32)
        self.animal_loading.setdefault(name, torch.zeros(S, dtype=torch.float32))
        self.last_loading[name] = torch.zeros(S, dtype=torch.float32)

    def loading(self, agent_state, name: str) -> torch.Tensor:
        """Today's load to every real source from infectious agents' households."""
        status = agent_state.ndata[AgentPropertyKeys.status(name)].cpu()
        infectious = (status == Compartment.INFECTIOUS).float()
        H = self.surface_kernel.shape[0]
        shed_surface = torch.zeros(H).index_add_(0, self.agent_hh, infectious * self.agent_surface_release)
        shed_ground = torch.zeros(H).index_add_(0, self.agent_hh, infectious * self.agent_ground_release)
        return shed_surface @ self.surface_kernel + shed_ground @ self.ground_kernel + self.animal_loading[name]

    def advance(self, agent_state, name: str, decay_rate: float, shock: bool):
        """One day for one pathogen: die-off, then today's loading, then any shock."""
        c = self.concentration[name]
        S = self.num_sources
        c[:S].mul_(math.exp(-decay_rate))
        c[S].mul_(math.exp(-(decay_rate + float(self.water_cfg.piped_flush_rate))))
        load = self.loading(agent_state, name)
        self.last_loading[name] = load
        c[:S].add_(load)
        if shock:
            self.apply_shock(name, load)

    def apply_shock(self, name: str, load: torch.Tensor):
        """A flood-type pulse: shock_magnitude days of the current mean surface
        loading added to every source, scaled by its type's vulnerability."""
        cfg = self.water_cfg
        n_reach = self._n_reach
        if n_reach == 0:
            return
        pulse = float(cfg.shock_magnitude) * float(load[:n_reach].mean())
        if pulse <= 0:
            return
        c = self.concentration[name]
        c[:n_reach].add_(pulse)
        c[n_reach:self.num_sources].add_(pulse * float(cfg.shock_vulnerability_ground))
        c[self.piped_index].add_(pulse * float(cfg.shock_vulnerability_piped))

    def agent_exposure_concentration(self, name: str) -> torch.Tensor:
        """Dilution-weighted concentration of each agent's drinking water."""
        c = self.concentration[name] * self.dilution
        return c[self.hh_source[self.agent_hh]]

    def agent_water_location(self) -> torch.Tensor:
        """(N, 2) row/col each agent goes to on a WATER day."""
        return self.hh_water_location[self.agent_hh]

    def summary(self) -> dict:
        kinds = self.source_kind
        return {
            "n_canal_reaches": int((kinds == CANAL).sum()),
            "n_river_reaches": int((kinds == RIVER).sum()),
            "n_wells": int((kinds == WELL).sum()),
            "n_boreholes": int((kinds == BOREHOLE).sum()),
            "household_type_shares": torch.bincount(self.hh_type, minlength=4).float().div(len(self.hh_type)).tolist(),
            "sanitation_shares": torch.bincount(self.hh_sanitation, minlength=4).float().div(len(self.hh_sanitation)).tolist(),
        }
