from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from optimized_convergence_detector import OptimizedEnhancedConvergenceDetector
from rsi_state_arbiter import RSIStateArbiter, ArbiterState


class TestPhase(Enum):
    """Testing phases with different evaluation policies."""
    COLD_START = "cold_start"  # Initial learning phase
    ACTIVE_EXPLORATION = "active_exploration"  # Heavy exploration
    CONVERGENCE_WATCH = "convergence_watch"  # Monitoring convergence
    VERIFICATION = "verification"  # Final validation


class IntegratedRSITest:
    """Integrated RSI testing framework with phased evaluation."""
    
    def __init__(self, cold_start_steps: int = 100, exploration_steps: int = 500,
                 convergence_steps: int = 300, verification_steps: int = 200):
        """
        Args:
            cold_start_steps: Steps for initial cold start phase
            exploration_steps: Steps for active exploration phase
            convergence_steps: Steps for convergence monitoring
            verification_steps: Steps for final verification
        """
        self.cold_start_steps = cold_start_steps
        self.exploration_steps = exploration_steps
        self.convergence_steps = convergence_steps
        self.verification_steps = verification_steps
        
        # Initialize components
        self.convergence_detector = OptimizedEnhancedConvergenceDetector(
            window_size=50, volatility_threshold=0.05, exploration_threshold=0.1
        )
        self.arbiter = RSIStateArbiter(k_steps_for_warning=5)
        
        # State tracking
        self.current_step = 0
        self.current_phase = TestPhase.COLD_START
        self.phase_history = []
        self.arbiter_states = []
        
    def update(self, new_hv: float, exploration_activity: float,
               meta_learning_report: Optional[Dict] = None) -> Dict:
        """
        Update framework with new data and get recommendations.
        
        Args:
            new_hv: New hypervolume value
            exploration_activity: Current exploration metric
            meta_learning_report: Optional meta-learning evaluation report
            
        Returns:
            Dict with current state, recommendations, and metrics
        """
        self.current_step += 1
        
        # Update phase
        self._update_phase()
        
        # Get convergence status
        convergence_status = self.convergence_detector.update(new_hv, exploration_activity)
        
        # Arbitrate state
        arbiter_state = self.arbiter.arbitrate(
            convergence_status=convergence_status,
            new_hv=new_hv,
            meta_learning_report=meta_learning_report
        )
        self.arbiter_states.append(arbiter_state)
        
        # Generate recommendations based on phase and state
        recommendations = self._generate_recommendations(
            convergence_status, arbiter_state, meta_learning_report
        )
        
        return {
            'step': self.current_step,
            'phase': self.current_phase.value,
            'arbiter_state': arbiter_state.value,
            'convergence_status': convergence_status,
            'recommendations': recommendations,
            'should_continue': self._should_continue(arbiter_state)
        }
    
    def _update_phase(self):
        """Update current testing phase based on step count."""
        if self.current_step <= self.cold_start_steps:
            new_phase = TestPhase.COLD_START
        elif self.current_step <= self.cold_start_steps + self.exploration_steps:
            new_phase = TestPhase.ACTIVE_EXPLORATION
        elif self.current_step <= (self.cold_start_steps + self.exploration_steps + 
                                    self.convergence_steps):
            new_phase = TestPhase.CONVERGENCE_WATCH
        else:
            new_phase = TestPhase.VERIFICATION
        
        if new_phase != self.current_phase:
            self.phase_history.append({
                'step': self.current_step,
                'old_phase': self.current_phase.value,
                'new_phase': new_phase.value
            })
            self.current_phase = new_phase
    
    def _generate_recommendations(self, convergence_status: Dict,
                                  arbiter_state: ArbiterState,
                                  meta_learning_report: Optional[Dict]) -> List[str]:
        """Generate actionable recommendations based on current state."""
        recommendations = []
        
        # Phase-specific recommendations
        if self.current_phase == TestPhase.COLD_START:
            recommendations.append("Cold start: Focus on diverse exploration")
            recommendations.append("Evaluation frequency: Every 20 steps")
        
        elif self.current_phase == TestPhase.ACTIVE_EXPLORATION:
            recommendations.append("Active exploration: Monitor frontier growth")
            recommendations.append("Evaluation frequency: Every 10 steps")
        
        elif self.current_phase == TestPhase.CONVERGENCE_WATCH:
            recommendations.append("Convergence watch: Check for stabilization")
            recommendations.append("Evaluation frequency: Every 5 steps")
        
        else:  # VERIFICATION
            recommendations.append("Verification: Validate final results")
            recommendations.append("Evaluation frequency: Every step")
        
        # State-specific warnings
        if arbiter_state == ArbiterState.INEFFICIENT_EXPLORATION:
            recommendations.append("WARNING: Inefficient exploration detected")
            recommendations.append("Action: Consider parameter adjustment or early stop")
        
        elif arbiter_state == ArbiterState.CRITICAL_FAILURE:
            recommendations.append("CRITICAL: Hypervolume decreasing")
            recommendations.append("Action: Stop test immediately and review logs")
        
        elif arbiter_state == ArbiterState.MODULE_CONFLICT:
            recommendations.append("WARNING: Module conflict detected")
            recommendations.append("Action: Review convergence thresholds")
        
        elif arbiter_state == ArbiterState.TRUE_CONVERGENCE:
            recommendations.append("SUCCESS: True convergence achieved")
            if self.current_phase != TestPhase.VERIFICATION:
                recommendations.append("Action: Consider transitioning to verification phase")
        
        # Meta-learning specific
        if meta_learning_report and meta_learning_report.get('is_significant'):
            recommendations.append(f"Meta-learning: Improvement trend detected")
        
        return recommendations
    
    def _should_continue(self, arbiter_state: ArbiterState) -> bool:
        """Determine if testing should continue."""
        # Stop on critical failure
        if arbiter_state == ArbiterState.CRITICAL_FAILURE:
            return False
        
        # Stop if past all phases
        total_steps = (self.cold_start_steps + self.exploration_steps + 
                       self.convergence_steps + self.verification_steps)
        if self.current_step >= total_steps:
            return False
        
        # Early stop if converged and verified
        if (arbiter_state == ArbiterState.TRUE_CONVERGENCE and 
            self.current_phase == TestPhase.VERIFICATION and
            self.current_step >= self.cold_start_steps + self.exploration_steps + 50):
            return False
        
        return True
    
    def get_summary(self) -> Dict:
        """Get comprehensive test summary."""
        arbiter_summary = self.arbiter.get_state_summary()
        detector_stats = self.convergence_detector.get_statistics()
        
        return {
            'total_steps': self.current_step,
            'current_phase': self.current_phase.value,
            'phase_history': self.phase_history,
            'arbiter_summary': arbiter_summary,
            'detector_statistics': detector_stats,
            'final_arbiter_state': self.arbiter_states[-1].value if self.arbiter_states else None
        }
    
    def reset(self):
        """Reset framework state."""
        self.current_step = 0
        self.current_phase = TestPhase.COLD_START
        self.phase_history = []
        self.arbiter_states = []
        self.convergence_detector.reset()
        # Note: RSIStateArbiter doesn't need reset as it's stateless per call
