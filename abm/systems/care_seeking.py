# abm/systems/care_seeking.py
import torch
from .system import System
from abm.state import AgentGraph
from abm.constants import AgentPropertyKeys
from abm.agent.health_cpt_utils import utility, cpt_value_function, probability_weighting

# --- Care Seeking Model Parameters (could be moved to config) ---
MODERATE_SEVERITY_THRESHOLD = 0.3
SEVERE_SEVERITY_THRESHOLD = 0.7
COST_OF_CARE = 0.15  # Cost as a proportion of max wealth
TREATMENT_SUCCESS_PROB = 0.80 # Probability care-seeking is effective
DURATION_REDUCTION_ON_SUCCESS = 5 # Days illness is shortened
NATURAL_WORSENING_PROB = 0.25 # Prob illness worsens if untreated
PARENT_STRESS_HEALTH_IMPACT = 0.05 # Health drop for parent if child worsens

class CareSeekingSystem(System):
    """
    Handles parent agent decision-making regarding seeking care for sick children.
    """
    def update(self, agent_graph: AgentGraph, **kwargs):
        child_severity = agent_graph.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY]
        sick_children_mask = (child_severity > MODERATE_SEVERITY_THRESHOLD) & \
                             (agent_graph.ndata[AgentPropertyKeys.IS_CHILD])

        if not torch.any(sick_children_mask):
            return

        # --- 1. Identify Parents of Sick Children ---
        sick_child_indices = sick_children_mask.nonzero(as_tuple=True)[0]
        child_hh_ids = agent_graph.ndata[AgentPropertyKeys.HOUSEHOLD_ID][sick_child_indices]
        parent_mask = agent_graph.ndata[AgentPropertyKeys.IS_PARENT]

        # Use a map for efficient lookup: hh_id -> list of sick children indices
        hh_to_sick_children = {}
        for i, hh_id in enumerate(child_hh_ids):
            hh_id_item = hh_id.item()
            if hh_id_item not in hh_to_sick_children:
                hh_to_sick_children[hh_id_item] = []
            hh_to_sick_children[hh_id_item].append(sick_child_indices[i])

        # --- 2. Parents Make Decisions ---
        # Find all parents who have at least one sick child
        all_hh_ids = agent_graph.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
        parent_indices = parent_mask.nonzero(as_tuple=True)[0]
        parent_hh_ids = all_hh_ids[parent_indices]

        for i, parent_idx in enumerate(parent_indices):
            parent_hh_id = parent_hh_ids[i].item()
            if parent_hh_id in hh_to_sick_children:
                # For simplicity, parent makes one decision for the sickest child in the HH
                children_in_hh = hh_to_sick_children[parent_hh_id]
                severities = child_severity[children_in_hh]
                sickest_child_local_idx = torch.argmax(severities)
                sickest_child_idx = children_in_hh[sickest_child_local_idx]
                
                self._parent_makes_decision(agent_graph, parent_idx, sickest_child_idx)


    def _parent_makes_decision(self, agent_graph: AgentGraph, parent_idx: int, child_idx: int):
        parent_wealth = agent_graph.ndata[AgentPropertyKeys.WEALTH][parent_idx]
        child_severity = agent_graph.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][child_idx]

        # Automatic decision for severe cases if affordable
        if child_severity >= SEVERE_SEVERITY_THRESHOLD:
            if parent_wealth >= COST_OF_CARE:
                self._apply_treatment_outcome(agent_graph, parent_idx, child_idx)
            return # If can't afford, do nothing

        # CPT calculation for moderate cases
        if parent_wealth < COST_OF_CARE:
            return # Cannot afford, so no decision to make

        # --- Define Prospects for CPT ---
        parent_params = {
            'gamma': agent_graph.ndata[AgentPropertyKeys.GAMMA][parent_idx].item(),
            'theta': self.config.steering_parameters.theta,
            'lambda': agent_graph.ndata[AgentPropertyKeys.LAMBDA][parent_idx].item(),
            'eta': self.config.steering_parameters.eta,
        }
        ref_utility = utility(parent_wealth, agent_graph.ndata[AgentPropertyKeys.HEALTH][parent_idx],
                              agent_graph.ndata[AgentPropertyKeys.ALPHA][parent_idx])

        # Prospect 1: Seek Care
        w_after_cost = parent_wealth - COST_OF_CARE
        # Outcome 1a: Treatment works (utility is just based on financial loss)
        util_care_success = utility(w_after_cost, agent_graph.ndata[AgentPropertyKeys.HEALTH][parent_idx], agent_graph.ndata[AgentPropertyKeys.ALPHA][parent_idx])
        cpt_val_care_success = cpt_value_function(util_care_success - ref_utility, parent_params)
        
        pi_care_success = probability_weighting(TREATMENT_SUCCESS_PROB, parent_params['gamma'])
        pi_care_fail = probability_weighting(1 - TREATMENT_SUCCESS_PROB, parent_params['gamma'])
        
        # In this simple model, utility is the same if care fails (just loss of money). A more complex model could add more stress.
        expected_value_seek_care = (pi_care_success * cpt_val_care_success) + (pi_care_fail * cpt_val_care_success)

        # Prospect 2: Wait
        # Outcome 2a: Child worsens (parent suffers stress health drop)
        h_after_stress = agent_graph.ndata[AgentPropertyKeys.HEALTH][parent_idx] - PARENT_STRESS_HEALTH_IMPACT
        util_wait_worsen = utility(parent_wealth, h_after_stress, agent_graph.ndata[AgentPropertyKeys.ALPHA][parent_idx])
        cpt_val_wait_worsen = cpt_value_function(util_wait_worsen - ref_utility, parent_params)
        # Outcome 2b: Child does not worsen (utility is unchanged)
        cpt_val_wait_stable = 0.0

        pi_wait_worsen = probability_weighting(NATURAL_WORSENING_PROB, parent_params['gamma'])
        pi_wait_stable = probability_weighting(1 - NATURAL_WORSENING_PROB, parent_params['gamma'])
        expected_value_wait = (pi_wait_worsen * cpt_val_wait_worsen) + (pi_wait_stable * cpt_val_wait_stable)

        # --- Make Decision ---
        if expected_value_seek_care > expected_value_wait:
            self._apply_treatment_outcome(agent_graph, parent_idx, child_idx)
        # Else, do nothing (wait)

    def _apply_treatment_outcome(self, agent_graph: AgentGraph, parent_idx: int, child_idx: int):
        # Apply cost to parent
        agent_graph.ndata[AgentPropertyKeys.WEALTH][parent_idx] -= COST_OF_CARE
        
        # Apply health outcome to child
        if torch.rand(1).item() < TREATMENT_SUCCESS_PROB:
            current_duration = agent_graph.ndata[AgentPropertyKeys.ILLNESS_DURATION][child_idx]
            new_duration = max(0, current_duration - DURATION_REDUCTION_ON_SUCCESS)
            agent_graph.ndata[AgentPropertyKeys.ILLNESS_DURATION][child_idx] = new_duration