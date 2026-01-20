# config.py

"""Configuration parameters for the SVEIR model.
The configuration parameters are stored in a pydantic object. The model is
initialized with default values. The default values can be overwritten by
providing a yaml file or a dictionary.
"""

from pathlib import Path

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

class InitialGraphArgs(BaseModel):
    """Base class for initial graph arguments."""
    seed: int = 1
    new_node_edges: int = 1
    model_config = ConfigDict(validate_default=True)

class GridCreationParams(BaseModel):
    """Base class for grid creation arguments"""
    method: str = "realistic_import"
    grid_id: str | None = None
    x: int | None = 75
    y: int | None = 75
    properties: dict | None = None
    model_config = ConfigDict(validate_default=True)

class SteeringParamsSVEIR(BaseModel):
    """Steering parameters used within each step of the SVEIR model."""
    npath: str = "./agent_data.zarr"
    epath: str = "./edge_data"
    ndata: list[str | list[str | list[str]]] | None = ["all_except", ["a_table"]]
    edata: list[str] | None = ["all"]
    mode: str = "w"
    infection_prob_mean: float = 0.002
    infection_prob_std: float = 0.0002
    recovery_rate: float = 0.33
    vaccination_rate: float = 0.01
    vaccine_efficacy: float = 0.9
    exposure_period: int = 5
    initial_infected_proportion: float = 0.03
    human_to_water_infection_prob: float = 0.001
    water_to_human_infection_prob: float = 0.001
    infection_reduction_factor_per_health_unit: float = 0.005
    beta: float = 0.95
    theta: float = 0.88
    P_H_increase: float = 0.75
    P_H_decrease: float = 0.50
    efficacy_multiplier: float = 1.0
    cost_subsidy_factor: float = 1.0
    infection_health_shock: int = 20
    wealth_update_A: float = 0.50
    water_recovery_prob: float = 0.2
    shock_frequency: int = 30
    shock_infection_prob: float = 0.10
    truncation_weight: float = 1.0e-10
    proximity_decay_rate: float = 0.5
    data_collection_period: int = 0
    data_collection_list: list[int] | None = None
    max_state_value: int = 100
    prior_infection_immunity_factor: float = 1.5

class SVEIRConfig(BaseModel):
    """Main configuration class for the SVEIR model."""
    model_identifier: str = Field("sveir_model", alias='_model_identifier')
    description: str = "Configuration for the SVEIR agent-based model."
    device: str = "cpu"
    seed: int = 42
    number_agents: PositiveInt = 300
    spatial: bool = True
    spatial_creation_args: GridCreationParams = GridCreationParams()
    initial_graph_type: str = "barabasi-albert"
    initial_graph_args: InitialGraphArgs = InitialGraphArgs()
    step_target: PositiveInt = 150

    # Demographic Parameters
    average_household_size: int = 4
    # probability that non-head-of-household member is a child
    child_probability: float = 0.2
    
    # Parameters for Policy Pre-computation
    policy_library_path: str = "./policy_library.npz"
    num_agent_personas: int = 32
    alpha_range: list[float] = [0.1, 0.9]
    gamma_range: list[float] = [0.2, 0.8]
    omega_range: list[float] = [1.0, 4.0]
    eta_range:   list[float] = [0.5, 1.0]

    steering_parameters: SteeringParamsSVEIR = SteeringParamsSVEIR()
    checkpoint_period: int = 0  # Set to 0 to disable default checkpointing
    milestones: list[PositiveInt] | None = None
    
    model_config = ConfigDict(
        validate_default=True,
        protected_namespaces=(),
        populate_by_name=True,
        validate_assignment=True,
        extra="forbid",
    )

    @classmethod
    def from_dict(cls, cfg):
        if not isinstance(cfg, dict):
            raise TypeError("Input must be a dictionary.")
        return cls(**cfg)
    
    def to_yaml(self, config_file):
        if Path(config_file).exists():
            print(f"Overwriting config file {config_file}.")

        cfg = self.model_dump(by_alias=True, warnings=False)

        def _convert_tensors(nested_dict):
            for key, value in nested_dict.items():
                if isinstance(value, torch.Tensor):
                    nested_dict[key] = value.tolist()
                elif isinstance(value, dict):
                    nested_dict[key] = _convert_tensors(value)
            return nested_dict

        cfg = _convert_tensors(cfg)
        with open(config_file, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)

    @classmethod
    def from_yaml(cls, config_file):
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Config file {config_file} not found.")
        with open(config_file) as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise SyntaxError(f"Error parsing config file {config_file}.") from exc
        return cls(**cfg)

# The only config object we instantiate and use in the project
SVEIRCONFIG = SVEIRConfig()