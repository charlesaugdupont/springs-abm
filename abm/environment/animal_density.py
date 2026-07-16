# abm/environment/animal_density.py
"""
Household-ownership-derived animal density layers for Campylobacter's
zoonotic transmission route.

Unlike the water/school/worship layers baked into grid.npz at grid-creation
time, these depend on a specific simulation run's random household placement
and ownership draw (varies with random seed), so they're built at runtime
in SVEIRModel._build_animal_density_layers() - once per initialize_model()
call - and attached to the GridEnvironment as dynamic layers rather than
saved to disk.

Method
------
Each owning household "stamps" an unnormalized 2D Gaussian bump (peak = 1.0
at its own home cell) onto an otherwise-empty raster. Bumps from all owning
households of a species are summed and clipped to [0, 1]. Concretely:
  - A household that owns poultry experiences ~1.0 density in its own yard
    (its own bump peak isn't diluted by surrounding empty cells, unlike a
    conventional mean-preserving blur).
  - Non-owning neighbours still experience meaningful, distance-decayed
    exposure, matching field evidence that free-ranging animals expose
    non-owning households too, not just their owners.
  - Areas with several nearby owning households accumulate higher density
    than an isolated owner, up to the [0, 1] cap.

poultry_sigma and ruminant_sigma (in grid cells) set the effective roaming
radius per species.

Calibration
-----------
Ownership probabilities (CampylobacterConfig.poultry_ownership_prob and
ruminant_ownership_prob) come from a compiled Ghana DHS/MICS-style survey
(notebooks/ghana_data.ipynb), IDW-interpolated over rural clusters nationwide
since no survey cluster falls inside Akuse - effectively a rural-southern-
Ghana average. Species weights and roam sigmas are literature-informed
starting points, not fitted values - treat as sensitivity-analysis
parameters, and recalibrate human_animal_interaction_rate alongside them.
"""
from typing import Tuple

import torch
import torch.nn.functional as F


def _unnormalized_gaussian_1d(sigma: float, device) -> torch.Tensor:
    """1D Gaussian kernel with peak value 1.0 at its center (not sum-normalized)."""
    if sigma <= 0:
        return torch.tensor([1.0], device=device)
    radius = max(1, int(round(3 * sigma)))
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    return torch.exp(-0.5 * (offsets / sigma) ** 2)


def _stamp_and_sum(indicator: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Convolves a sparse 0/1 indicator raster with an unnormalized, separable
    2D Gaussian kernel, then clips to [0, 1].

    Because the kernel is unnormalized (peak = 1, not sum = 1), this is
    equivalent to stamping a Gaussian "bump" of height 1 at each indicator
    cell and summing overlapping bumps -- it does NOT dilute a household's
    own-cell value the way a conventional mean-preserving blur would.
    """
    if sigma <= 0:
        return torch.clamp(indicator, 0.0, 1.0)

    device = indicator.device
    kernel_1d = _unnormalized_gaussian_1d(sigma, device)
    k = kernel_1d.numel()
    pad = k // 2

    x = indicator.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    x = F.conv2d(x, kernel_1d.view(1, 1, 1, k), padding=(0, pad))  # horizontal pass
    x = F.conv2d(x, kernel_1d.view(1, 1, k, 1), padding=(pad, 0))  # vertical pass

    return torch.clamp(x.squeeze(0).squeeze(0), 0.0, 1.0)


def _place_on_grid(
    household_y: torch.Tensor,
    household_x: torch.Tensor,
    household_indicator: torch.Tensor,
    grid_shape: Tuple[int, int],
    device,
) -> torch.Tensor:
    """Scatters one value per household onto its home cell of an empty raster."""
    rows, cols = grid_shape
    raster = torch.zeros(rows * cols, device=device, dtype=torch.float32)

    y_idx = household_y.long().clamp(0, rows - 1)
    x_idx = household_x.long().clamp(0, cols - 1)
    flat_idx = (y_idx * cols + x_idx).to(device)

    raster.index_add_(0, flat_idx, household_indicator.to(device).float())
    return raster.view(rows, cols)


def build_animal_density_layers(
    household_y: torch.Tensor,
    household_x: torch.Tensor,
    household_owns_poultry: torch.Tensor,
    household_owns_ruminant: torch.Tensor,
    grid_shape: Tuple[int, int],
    poultry_sigma: float,
    ruminant_sigma: float,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds the poultry and ruminant density layers from per-household
    ownership indicators and home coordinates. Returns (poultry_density,
    ruminant_density), each a (rows, cols) tensor clipped to [0, 1].
    """
    poultry_points = _place_on_grid(household_y, household_x, household_owns_poultry, grid_shape, device)
    ruminant_points = _place_on_grid(household_y, household_x, household_owns_ruminant, grid_shape, device)

    poultry_density = _stamp_and_sum(poultry_points, poultry_sigma)
    ruminant_density = _stamp_and_sum(ruminant_points, ruminant_sigma)

    return poultry_density, ruminant_density