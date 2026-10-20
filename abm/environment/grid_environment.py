"""These functions pertain to spatial grid creation."""
import torch

class GridEnvironment:
    """This class represents a grid environment with properties."""
    def __init__(self, grid_tensor: torch.Tensor, property_index: dict):
        self.grid_tensor = grid_tensor
        self.property_to_index = property_index
        self.grid_shape = grid_tensor.shape
        # layers derived per simulation run (e.g. household-ownership-driven
        # animal density) rather than baked into static grid.npz file.
        self._dynamic_layers: dict[str, torch.Tensor] = {}
        # Set by EnvironmentFactory: (minx, miny, x_step, y_step) in degrees.
        self.bounds = None
        # Set by SVEIRModel when WaterConfig.model == "catchment".
        self.water_network = None

    def get_slice(self, property_name: str) -> torch.Tensor:
        """Returns a 2D slice of the grid for a given property."""
        if property_name not in self.property_to_index:
            raise KeyError(f"Property '{property_name}' not found.")
        index = self.property_to_index[property_name]
        return self.grid_tensor[:, :, index]

    def __getitem__(self, property_name: str) -> torch.Tensor:
        return self.get_slice(property_name)
    
    def set_dynamic_layer(self, name: str, layer: torch.Tensor):
        """Attaches a per-run derived layer (not part of the static grid)."""
        self._dynamic_layers[name] = layer

    def get_dynamic_layer(self, name: str) -> torch.Tensor | None:
        """Returns a previously-attached dynamic layer, or None if absent."""
        return self._dynamic_layers.get(name)

    # ------------------------------------------------------------------
    # Per-pathogen water contamination state
    # ------------------------------------------------------------------
    # Water contamination used to live directly in grid_tensor's 'water' layer,
    # mutated in place and owned de facto by rotavirus (which referred to it by
    # hardcoded name). That cannot represent two pathogens contaminating the
    # same sources on different timescales: rotavirus clears on a ~5-day
    # timescale, Cryptosporidium oocysts persist for weeks and accumulate.
    #
    # So each waterborne pathogen now owns a dynamic layer of its own, seeded
    # from the static water-point mask. The static layer stays pristine, which
    # also removes a latent fragility: AgentFactory._find_nearest_locations
    # assigns WATER_LOCATION by looking for cells equal to 1 in the static
    # layer, and would silently have skipped any cell a previous run had left
    # marked CONTAMINATED (==2).

    @staticmethod
    def water_state_key(pathogen_name: str) -> str:
        return f"water_state_{pathogen_name}"

    def set_water_state(self, pathogen_name: str, layer: torch.Tensor):
        self._dynamic_layers[self.water_state_key(pathogen_name)] = layer

    def get_water_state(self, pathogen_name: str) -> torch.Tensor | None:
        return self._dynamic_layers.get(self.water_state_key(pathogen_name))

    def water_point_mask(self) -> torch.Tensor | None:
        """Boolean mask of cells that contain a water source at all.

        Read from the STATIC grid layer, so it is unaffected by any pathogen's
        contamination state. Returns None when the grid has no water layer.
        """
        from abm.constants import GridLayer
        water_idx = self.property_to_index.get(GridLayer.WATER)
        if water_idx is None:
            return None
        return self.grid_tensor[:, :, water_idx] > 0