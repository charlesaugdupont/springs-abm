# calibration.py
import argparse
import numpy as np
from tabulate import tabulate

from abm.model.initialize_model import SVEIRModel
from config import SVEIRCONFIG

def run_sweep(grid_id, sweep_type, steps=150, agents=500):
    """Runs a parameter sweep for specific calibration targets."""

    results = []
    
    # Define the parameter ranges based on the sweep type
    if sweep_type == "rota_r0":
        # Sweeping Infectiousness to find realistic R0 / Attack Rate
        param_name = "infection_prob_mean"
        values = [0.01, 0.015, 0.02, 0.025]
        print(f"\n--- CALIBRATION: Rotavirus Infectiousness ({len(values)} runs) ---")
        
    elif sweep_type == "campy_env":
        # Sweeping Interaction Rate to find endemic baseline
        param_name = "human_animal_interaction_rate"
        values = [0.25, 0.5, 1, 2, 4]
        print(f"\n--- CALIBRATION: Campylobacter Environmental Risk ({len(values)} runs) ---")
    
    else:
        print("Unknown sweep type.")
        return

    for val in values:
        # 1. Setup Configuration
        config = SVEIRCONFIG.model_copy(deep=True)
        config.number_agents = agents
        config.step_target = steps
        config.spatial_creation_args.grid_id = grid_id
        config.device = 'cpu' # CPU is usually faster for small agent counts (overhead)

        # 2. Isolate the Variable
        if sweep_type == "rota_r0":
            # Disable Campy
            config.pathogens[1].human_animal_interaction_rate = 0.0 
            
            # --- Disable Water Transmission for pure H2H calibration ---
            config.steering_parameters.water_to_human_infection_prob = 0.0
            config.steering_parameters.human_to_water_infection_prob = 0.0
            config.steering_parameters.shock_daily_prob = 0.0
            
            # Set the variable we are testing
            config.pathogens[0].infection_prob_mean = val
            
        elif sweep_type == "campy_env":
            # Disable Rota (H2H + Water)
            config.pathogens[0].infection_prob_mean = 0.0
            config.steering_parameters.water_to_human_infection_prob = 0.0
            config.steering_parameters.human_to_water_infection_prob = 0.0
            config.steering_parameters.shock_daily_prob = 0.0
            
            # Set the variable we are testing
            config.pathogens[1].human_animal_interaction_rate = val

        # 3. Run Simulation
        print(f" > Running {param_name} = {val} ...", end="", flush=True)
        model = SVEIRModel(model_identifier=f"calib_{sweep_type}_{val}", root_path="tests/outputs")
        model.set_model_parameters(**config.model_dump())
        model.initialize_model(verbose=False)
        model.run()
        
        # 4. Extract Metrics
        ts = model.get_time_series_data()
        incidence = np.array(ts['incidence'])
        prevalence = np.array(ts['prevalence'])
        
        total_infections = np.sum(incidence)
        attack_rate = total_infections / agents
        peak_day = np.argmax(prevalence)
        peak_count = np.max(prevalence)
        
        # Simple R0 estimation (Growth rate in first 10 days)
        # N_t = N_0 * e^(rt) -> r approx slope of log(incidence)
        # This is rough but useful for relative comparison
        if np.sum(incidence[:10]) > 0:
            early_growth = incidence[:15]
            # Filter zeros for log
            valid_idx = early_growth > 0
            if np.sum(valid_idx) > 3:
                r = np.polyfit(np.arange(15)[valid_idx], np.log(early_growth[valid_idx]), 1)[0]
            else:
                r = 0
        else:
            r = 0

        results.append({
            param_name: val,
            "Attack Rate %": f"{attack_rate*100:.1f}%",
            "Peak Day": peak_day,
            "Peak Count": peak_count,
            "Est. Growth (r)": f"{r:.3f}"
        })
        print(" Done.")

    # 5. Report
    print("\n" + tabulate(results, headers="keys", tablefmt="github"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("grid_id", type=str, help="The Grid ID to use")
    parser.add_argument("mode", choices=["rota_r0", "campy_env"], help="Calibration mode")
    args = parser.parse_args()
    
    run_sweep(args.grid_id, args.mode)