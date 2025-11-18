"""
Integration tests for the Enhanced RSI Test Framework.

Ensures that all modules work together as expected.
"""

import pytest
import numpy as np
from meta_learning_evaluator import AdvancedMetaLearningEvaluator
from optimized_convergence_detector import OptimizedEnhancedConvergenceDetector
from rsi_state_arbiter import RSIStateArbiter, ArbiterState
from fast_pareto_optimizer import FastParetoOptimizer
from integrated_rsi_test import IntegratedRSITest, TestPhase

def test_meta_learning_evaluator_lifecycle():
    """Tests the full lifecycle of the meta-learning evaluator."""
    evaluator = AdvancedMetaLearningEvaluator(min_data_points=15, bootstrap_samples=100)

    # Check insufficient data case
    for i in range(10):
        evaluator.update(i, 100 + i * 2)
    report = evaluator.get_report()
    assert 'insufficient' in report.get('final_interpretation', '').lower()

    # Check sufficient data case
    for i in range(10, 20):
        evaluator.update(i, 100 + i * 2)
    report = evaluator.get_report()
    assert 'adaptive_smoothing' in report and 'trend_model_comparison' in report
    assert report['is_significant'] is not None

def test_convergence_detector_states():
    """Tests the state transitions of the convergence detector."""
    detector = OptimizedEnhancedConvergenceDetector(
        window_size=10,
        volatility_threshold=0.05,
        exploration_threshold=0.1
    )

    # Test IMPROVING state
    for i in range(15):
        status = detector.update(100 + i * 5, 0.05)
    assert status['state'] == 'IMPROVING'

    # Test CONVERGED state
    for i in range(20):
        status = detector.update(200.0, 0.01)
    assert status['state'] == 'CONVERGED'

    # Test EXPLORING state with volatility
    base_hv = 210.0
    for i in range(15):
        # Use a sine wave to create non-trivial volatility
        hv = base_hv + np.sin(i) * base_hv * 0.1
        status = detector.update(hv, 0.2)
    assert status['state'] == 'EXPLORING'

def test_pareto_optimizer_functionality():
    """Tests the core functionality of the Pareto optimizer."""
    # Note: reference_point is now handled inside the class.
    # We pass it here to ensure hypervolume is calculated correctly.
    optimizer = FastParetoOptimizer(
        objective_directions={'performance': 'maximize', 'cost': 'minimize'},
        reference_point={'performance': 0.0, 'cost': 200.0}
    )

    solutions = [
        {'performance': 0.9, 'cost': 100},
        {'performance': 0.8, 'cost': 50},
        {'performance': 0.7, 'cost': 30},
        {'performance': 0.6, 'cost': 80},  # Dominated
    ]

    for sol in solutions:
        optimizer.add_solution(sol)

    frontier = optimizer.get_frontier()
    assert len(frontier) == 3

    report = optimizer.get_report()
    assert 'metrics' in report and report['metrics']['frontier_size'] == 3
    # Check that hypervolume is positive after adding solutions
    assert report['metrics']['hypervolume'] > 0

def test_state_arbiter_logic():
    """Tests the decision logic of the state arbiter."""
    arbiter = RSIStateArbiter(k_steps_for_warning=3)

    arbiter.arbitrate(convergence_status={'state': 'IMPROVING'}, new_hv=100.0)

    state = arbiter.arbitrate(convergence_status={'state': 'IMPROVING'}, new_hv=110.0)
    assert state == ArbiterState.HEALTHY_GROWTH

    state = arbiter.arbitrate(convergence_status={'state': 'EXPLORING'}, new_hv=120.0)
    assert state == ArbiterState.EFFICIENT_EXPLORATION

    for _ in range(3):
        arbiter.arbitrate(convergence_status={'state': 'EXPLORING'}, new_hv=120.0)

    warning, msg = arbiter.check_for_persistent_warning()
    assert warning is True
    assert "persisted for 3 steps" in msg

def test_full_integration_lifecycle():
    """
    Tests the integration of all modules through the IntegratedRSITest framework.
    """
    framework = IntegratedRSITest(
        cold_start_steps=10,
        exploration_steps=15,
        convergence_steps=10,
        verification_steps=5
    )

    for i in range(10):
        result = framework.update(new_hv=100.0 + i, exploration_activity=0.3)
        assert result['phase'] == TestPhase.COLD_START.value

    for i in range(15):
        result = framework.update(new_hv=110.0 + i, exploration_activity=0.2)
        assert result['phase'] == TestPhase.ACTIVE_EXPLORATION.value

    for i in range(10):
        result = framework.update(new_hv=125.0, exploration_activity=0.05)
        assert result['phase'] == TestPhase.CONVERGENCE_WATCH.value

    for i in range(5):
        result = framework.update(new_hv=125.0 + i*0.1, exploration_activity=0.01)
        assert result['phase'] == TestPhase.VERIFICATION.value

    summary = framework.get_summary()
    assert summary['total_steps'] == 40
    assert len(summary['phase_history']) > 0
    assert summary['final_arbiter_state'] is not None
