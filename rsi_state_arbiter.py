from __future__ import annotations

"""RSI State Arbiter - Module Consistency Verification

Inter-module consistency validation for RSI testing framework.
"""

from enum import Enum, auto
from typing import Tuple, List, Dict


class ArbiterState(Enum):
    """System states determined by RSI State Arbiter."""
    HEALTHY_GROWTH = auto()
    EFFICIENT_EXPLORATION = auto()
    INEFFICIENT_EXPLORATION = auto()
    TRUE_CONVERGENCE = auto()
    CRITICAL_FAILURE = auto()
    MODULE_CONFLICT = auto()
    INITIALIZING = auto()


class RSIStateArbiter:
    """Arbitrates between three modules to determine system health."""
    
    def __init__(self, k_steps_for_warning: int = 10, hv_epsilon: float = 1e-9):
        self.k_steps_for_warning = k_steps_for_warning
        self.hv_epsilon = hv_epsilon
        self.state_history: List[Tuple[int, ArbiterState, str]] = []
        self.inefficient_counter: int = 0
        self.previous_hv: float = -1.0
        self.current_step: int = 0
    
    def arbitrate(self, convergence_status: Dict, new_hv: float, 
                  meta_learning_report: Dict = None) -> ArbiterState:
        """Determine system state from module inputs."""
        self.current_step += 1
        hv_state = self._get_hv_state(new_hv)
        
        # Extract state string from convergence_status dict
        conv_state = convergence_status.get('state', 'INITIALIZING')
        
        # Handle optional meta_learning_report
        meta_robust = False
        if meta_learning_report:
            meta_robust = meta_learning_report.get("robustness_analysis", {}).get(
                "is_robust (CI_lower > 0)", False)
        
        # Rule-based state determination
        if hv_state == 'DECREASING':
            state = ArbiterState.CRITICAL_FAILURE
            reason = "HV decreased - optimizer bug detected"
        elif conv_state == 'CONVERGED' and hv_state == 'INCREASING':
            state = ArbiterState.MODULE_CONFLICT
            reason = "Contradiction: convergence but HV still increasing"
        elif conv_state == 'CONVERGED':
            state = ArbiterState.TRUE_CONVERGENCE
            reason = "System converged - exploration stopped"
        elif conv_state == 'EXPLORING' and hv_state == 'INCREASING':
            state = ArbiterState.EFFICIENT_EXPLORATION
            reason = "Excellent: Performance stagnant but HV growing"
        elif conv_state == 'IMPROVING' and hv_state == 'INCREASING':
            state = ArbiterState.HEALTHY_GROWTH
            reason = "Ideal: Both performance and HV improving"
        elif conv_state in ['EXPLORING', 'IMPROVING'] and hv_state == 'STAGNANT':
            state = ArbiterState.INEFFICIENT_EXPLORATION
            reason = "Warning: Exploring but frontier not expanding"
        else:
            state = ArbiterState.INITIALIZING
            reason = f"Data collection (Conv: {conv_state}, HV: {hv_state})"
        
        self._update_inefficient_counter(state)
        self.state_history.append((self.current_step, state, reason))
        self.previous_hv = new_hv
        return state
    
    def _get_hv_state(self, new_hv: float) -> str:
        if self.previous_hv < 0:
            return 'INITIALIZING'
        delta = new_hv - self.previous_hv
        if delta < -self.hv_epsilon:
            return 'DECREASING'
        elif delta > self.hv_epsilon:
            return 'INCREASING'
        return 'STAGNANT'
    
    def _update_inefficient_counter(self, state: ArbiterState):
        if state == ArbiterState.INEFFICIENT_EXPLORATION:
            self.inefficient_counter += 1
        else:
            self.inefficient_counter = 0
    
    def check_for_persistent_warning(self) -> Tuple[bool, str]:
        if self.inefficient_counter >= self.k_steps_for_warning:
            msg = (f"[Alert] INEFFICIENT_EXPLORATION persisted for "
                   f"{self.inefficient_counter} steps (>= {self.k_steps_for_warning})")
            self.inefficient_counter = 0
            return True, msg
        return False, ""
    
    def get_state_history(self) -> List[Tuple[int, ArbiterState, str]]:
        return self.state_history
    
    def get_state_summary(self) -> Dict:
        """Get state summary with counts and transition history."""
        state_counts = {state.name: 0 for state in ArbiterState}
        for _, state, _ in self.state_history:
            state_counts[state.name] += 1
        
        total = self.current_step
        percentages = {}
        if total > 0:
            percentages = {s: f"{(c/total)*100:.1f}%" for s, c in state_counts.items()}
        
        return {
            'state_counts': state_counts,
            'state_percentages': percentages,
            'transition_history': [
                {'step': step, 'state': state.name, 'reason': reason}
                for step, state, reason in self.state_history[-10:]  # Last 10 transitions
            ],
            'total_steps': total
        }
