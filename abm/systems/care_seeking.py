# abm/systems/care_seeking.py
import torch
from .system import System
from abm.state import AgentState
from abm.constants import AgentPropertyKeys
from abm.agent.health_cpt_utils import utility, cpt_value_function, probability_weighting

class CareSeekingSystem(System):
    """
    Handles parent agent decision-making regarding seeking care for sick children.
    """
    def update(self, agent_state: AgentState, **kwargs):
        params = self.config.steering_parameters
        child_severity = agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY]
        sick_children_mask = (child_severity > params.moderate_severity_threshold) & \
                             (agent_state.ndata[AgentPropertyKeys.IS_CHILD])

        if not torch.any(sick_children_mask):
            return

        # --- 1. Identify Parents of Sick Children ---
        sick_child_indices = sick_children_mask.nonzero(as_tuple=True)[0]
        child_hh_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID][sick_child_indices]
        parent_mask = agent_state.ndata[AgentPropertyKeys.IS_PARENT]

        # Use a map for efficient lookup: hh_id -> list of sick children indices
        hh_to_sick_children = {}
        for i, hh_id in enumerate(child_hh_ids):
            hh_id_item = hh_id.item()
            if hh_id_item not in hh_to_sick_children:
                hh_to_sick_children[hh_id_item] = []
            hh_to_sick_children[hh_id_item].append(sick_child_indices[i])

        # --- 2. Parents Make Decisions ---
        # Find all parents who have at least one sick child
        all_hh_ids = agent_state.ndata[AgentPropertyKeys.HOUSEHOLD_ID]
        parent_indices = parent_mask.nonzero(as_tuple=True)[0]
        parent_hh_ids = all_hh_ids[parent_indices]

        for i, parent_idx in enumerate(parent_indices):
            parent_hh_id = parent_hh_ids[i].item()
            if parent_hh_id in hh_to_sick_children:
                # For simplicity, parent makes one decision for the sickest child in the HH
                children_in_hh_indices = hh_to_sick_children[parent_hh_id]

                # Convert the list of indices into a single, proper tensor for indexing
                indices_tensor = torch.tensor(
                    [idx.item() for idx in children_in_hh_indices],
                    device=agent_state.device,
                    dtype=torch.long
                )
                severities = child_severity[indices_tensor]

                sickest_child_local_idx = torch.argmax(severities)
                sickest_child_idx = children_in_hh_indices[sickest_child_local_idx]
                
                self._parent_makes_decision(agent_state, parent_idx, sickest_child_idx)


    def _parent_makes_decision(self, agent_state: AgentState, parent_idx: int, child_idx: int):
        params = self.config.steering_parameters
        parent_wealth = agent_state.ndata[AgentPropertyKeys.WEALTH][parent_idx]
        child_severity = agent_state.ndata[AgentPropertyKeys.SYMPTOM_SEVERITY][child_idx]

        # Automatic decision for severe cases if affordable
        if child_severity >= params.severe_severity_threshold:
            if parent_wealth >= params.cost_of_care:
                self._apply_treatment_outcome(agent_state, parent_idx, child_idx)
            return # If can't afford, do nothing

        # CPT calculation for moderate cases
        if parent_wealth < params.cost_of_care:
            return # Cannot afford, so no decision to make

        # --- Define Prospects for CPT ---
        parent_params = {
            'gamma': agent_state.ndata[AgentPropertyKeys.GAMMA][parent_idx].item(),
            'theta': self.config.steering_parameters.theta,
            'lambda': agent_state.ndata[AgentPropertyKeys.LAMBDA][parent_idx].item(),
            'eta': self.config.steering_parameters.eta,
        }
        ref_utility = utility(parent_wealth, agent_state.ndata[AgentPropertyKeys.HEALTH][parent_idx],
                              agent_state.ndata[AgentPropertyKeys.ALPHA][parent_idx])

        # Prospect 1: Seek Care
        w_after_cost = parent_wealth - params.cost_of_care
        # Outcome 1a: Treatment works (utility is just based on financial loss)
        util_care_success = utility(w_after_cost, agent_state.ndata[AgentPropertyKeys.HEALTH][parent_idx], agent_state.ndata[AgentPropertyKeys.ALPHA][parent_idx])
        cpt_val_care_success = cpt_value_function(util_care_success - ref_utility, parent_params)
        
        pi_care_success = probability_weighting(params.treatment_success_prob, parent_params['gamma'])
        pi_care_fail = probability_weighting(1 - params.treatment_success_prob, parent_params['gamma'])
        
        # In this simple model, utility is the same if care fails (just loss of money). A more complex model could add more stress.
        expected_value_seek_care = (pi_care_success * cpt_val_care_success) + (pi_care_fail * cpt_val_care_success)

        # Prospect 2: Wait
        # Outcome 2a: Child worsens (parent suffers stress health drop)
        h_after_stress = agent_state.ndata[AgentPropertyKeys.HEALTH][parent_idx] - params.parent_stress_health_impact
        util_wait_worsen = utility(parent_wealth, h_after_stress, agent_state.ndata[AgentPropertyKeys.ALPHA][parent_idx])
        cpt_val_wait_worsen = cpt_value_function(util_wait_worsen - ref_utility, parent_params)
        # Outcome 2b: Child does not worsen (utility is unchanged)
        cpt_val_wait_stable = 0.0

        pi_wait_worsen = probability_weighting(params.natural_worsening_prob, parent_params['gamma'])
        pi_wait_stable = probability_weighting(1 - params.natural_worsening_prob, parent_params['gamma'])
        expected_value_wait = (pi_wait_worsen * cpt_val_wait_worsen) + (pi_wait_stable * cpt_val_wait_stable)

        # --- Make Decision ---
        if expected_value_seek_care > expected_value_wait:
            self._apply_treatment_outcome(agent_state, parent_idx, child_idx)
        # Else, do nothing (wait)

    def _apply_treatment_outcome(self, agent_state: AgentState, parent_idx: int, child_idx: int):
        params = self.config.steering_parameters
        # Apply cost to parent
        agent_state.ndata[AgentPropertyKeys.WEALTH][parent_idx] -= params.cost_of_care

        # Increment care seeking counter
        agent_state.ndata[AgentPropertyKeys.CARE_SEEKING_COUNT][parent_idx] += 1
        
        # Apply health outcome to child
        if torch.rand(1).item() < params.treatment_success_prob:
            current_duration = agent_state.ndata[AgentPropertyKeys.ILLNESS_DURATION][child_idx]
            new_duration = max(0, current_duration - params.duration_reduction_on_success)
            agent_state.ndata[AgentPropertyKeys.ILLNESS_DURATION][child_idx] = new_duration