# tests/verification_suite.py
import torch
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abm.model.initialize_model import SVEIRModel
from config import SVEIRCONFIG
from abm.constants import AgentPropertyKeys, Compartment


def run_conservation_check():
    """TEST 1: Unit Verification - Conservation of Agents"""
    print("\n--- TEST 1: Conservation of Agents ---")
    config = SVEIRCONFIG.model_copy(deep=True)
    config.number_agents = 500
    config.step_target = 10
    config.device = 'cpu' # Safer for tests
    config.spatial = False
    
    # Disable natural deaths or births (if any existed) to ensure strict conservation
    
    model = SVEIRModel(model_identifier="test_conservation", root_path="tests/outputs")
    model.set_model_parameters(**config.model_dump())
    model.initialize_model(verbose=False)
    
    initial_count = model.graph.num_nodes()
    print(f"Initial Agents: {initial_count}")
    
    model.run()
    
    # Check Rota Stats
    status = model.graph.ndata[AgentPropertyKeys.status('rota')]
    s = torch.sum(status == Compartment.SUSCEPTIBLE).item()
    e = torch.sum(status == Compartment.EXPOSED).item()
    i = torch.sum(status == Compartment.INFECTIOUS).item()
    r = torch.sum(status == Compartment.RECOVERED).item()
    v = torch.sum(status == Compartment.VACCINATED).item()
    
    total = s + e + i + r + v
    print(f"Final Counts -> S:{s} E:{e} I:{i} R:{r} V:{v}")
    print(f"Sum: {total}")
    
    if total == initial_count:
        print("✅ PASS: Agent count conserved.")
    else:
        print(f"❌ FAIL: Agent count mismatch ({initial_count} vs {total})")


def run_care_seeking_calibration():
    """TEST 2: Component Calibration - Care Seeking Logic"""
    print("\n--- TEST 2: Care Seeking Calibration ---")
    
    # We will manually inject a state into the CareSeekingSystem to see if it responds correctly
    from abm.systems.care_seeking import CareSeekingSystem
    
    config = SVEIRCONFIG.model_copy(deep=True)
    config.number_agents = 10 # Small number
    config.device = 'cpu'
    config.spatial = False
    
    # Initialize basic state
    model = SVEIRModel(model_identifier="test_care", root_path="tests/outputs")
    model.set_model_parameters(**config.model_dump())
    model.initialize_model(verbose=False)
    
    # MANUALLY SETUP AGENTS
    # Agent 0: Parent, Rich, Child is severely sick -> Should Seek Care
    # Agent 1: Parent, Poor, Child is severely sick -> Can't Seek Care
    
    # Reset all relevant arrays
    model.graph.ndata[AgentPropertyKeys.IS_PARENT] = torch.zeros(10, dtype=torch.bool)
    model.graph.ndata[AgentPropertyKeys.IS_CHILD] = torch.zeros(10, dtype=torch.bool)
    model.graph.ndata[AgentPropertyKeys.HOUSEHOLD_ID] = torch.arange(10) # Everyone in own house for simplicity
    
    # Setup Pair 1 (Rich)
    model.graph.ndata[AgentPropertyKeys.IS_PARENT][0] = True
    model.graph.ndata[AgentPropertyKeys.IS_CHILD][1] = True
    model.graph.ndata[AgentPropertyKeys.HOUSEHOLD_ID][1] = 0 # Child 1 belongs to Parent 0
    model.graph.ndata[AgentPropertyKeys.WEALTH][0] = 1.0 # Max Wealth
    model.graph.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][1] = 0.9 # Severe
    model.graph.ndata[AgentPropertyKeys.ILLNESS_DURATION][1] = 10
    
    # Setup Pair 2 (Poor)
    model.graph.ndata[AgentPropertyKeys.IS_PARENT][2] = True
    model.graph.ndata[AgentPropertyKeys.IS_CHILD][3] = True
    model.graph.ndata[AgentPropertyKeys.HOUSEHOLD_ID][3] = 2 
    model.graph.ndata[AgentPropertyKeys.WEALTH][2] = 0.01 # Poor
    model.graph.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][3] = 0.9 # Severe
    model.graph.ndata[AgentPropertyKeys.ILLNESS_DURATION][3] = 10

    # Initialize System
    care_system = CareSeekingSystem(config)
    
    print(f"Pre-Update Wealth [Rich Parent]: {model.graph.ndata[AgentPropertyKeys.WEALTH][0]:.2f}")
    
    # Run System Update
    care_system.update(model.graph)
    
    # Check outcomes
    # 1. Rich parent should have spent money
    new_wealth_rich = model.graph.ndata[AgentPropertyKeys.WEALTH][0].item()
    
    if new_wealth_rich < 1.0:
        print(f"✅ PASS: Rich parent spent money (Current: {new_wealth_rich:.2f}).")
    else:
        print(f"❌ FAIL: Rich parent did not spend money.")

    # 2. Poor parent should not have spent money (can't afford)
    new_wealth_poor = model.graph.ndata[AgentPropertyKeys.WEALTH][2].item()
    
    if math.isclose(new_wealth_poor, 0.01, rel_tol=1e-5):
        print(f"✅ PASS: Poor parent did not spend money.")
    else:
        print(f"❌ FAIL: Poor parent spent money despite being broke (Wealth: {new_wealth_poor}).")


def run_sanity_check_epidemic():
    """TEST 3: Sanity Check - Basic Epidemic Curve"""
    print("\n--- TEST 3: Epidemic Sanity Check ---")
    config = SVEIRCONFIG.model_copy(deep=True)
    config.number_agents = 1000
    config.step_target = 50
    config.spatial = False
    config.device = 'cpu'
    
    # Force high infectivity to guarantee a curve
    for p in config.pathogens:
        if p.name == 'rota':
            p.infection_prob_mean = 0.1 # High transmission
            p.initial_infected_proportion = 0.05

    model = SVEIRModel(model_identifier="test_epi", root_path="tests/outputs")
    model.set_model_parameters(**config.model_dump())
    model.initialize_model(verbose=False)
    model.run()
    
    curve = model.get_time_series_data()['prevalence']
    peak_infections = max(curve)
    
    print(f"Peak Infections: {peak_infections}")
    print(f"Curve: {curve[::5]} ... (sampled)")
    
    if peak_infections > config.number_agents * 0.1:
        print("✅ PASS: Epidemic took off (Peak > 10% population).")
    else:
        print("❌ FAIL: No epidemic generated (Check transmission params).")

if __name__ == "__main__":
    run_conservation_check()
    run_care_seeking_calibration()
    run_sanity_check_epidemic()