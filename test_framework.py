#!/usr/bin/env python3
"""
Comprehensive test script for Enhanced RSI Test Framework
Tests all modules and their integration
"""

import sys
import traceback
import numpy as np
from typing import Dict, List

# Test results tracking
test_results = []

def log_test(name: str, passed: bool, message: str = ""):
    """Log test result"""
    status = "✓ PASS" if passed else "✗ FAIL"
    test_results.append({'name': name, 'passed': passed, 'message': message})
    print(f"{status}: {name}")
    if message:
        print(f"  → {message}")

def test_meta_learning_evaluator():
    """Test MetaLearningEvaluator module"""
    print("\n=== Testing MetaLearningEvaluator ===")
    try:
        from meta_learning_evaluator import AdvancedMetaLearningEvaluator
        
        evaluator = AdvancedMetaLearningEvaluator(min_data_points=15, bootstrap_samples=1000)
        log_test("MetaLearningEvaluator: Import and initialization", True)
        
        # Test with insufficient data
        for i in range(10):
            evaluator.update(i, 100 + i * 2)
        
        report = evaluator.get_report()
        if 'insufficient' in report.get('final_interpretation', '').lower():
            log_test("MetaLearningEvaluator: Insufficient data handling", True)
        else:
            log_test("MetaLearningEvaluator: Insufficient data handling", False, 
                    f"Expected insufficient data message, got: {report.get('final_interpretation')}")
        
        # Test with sufficient data
        for i in range(10, 20):
            evaluator.update(i, 100 + i * 2)
        
        report = evaluator.get_report()
        if 'adaptive_smoothing' in report and 'trend_model_comparison' in report:
            log_test("MetaLearningEvaluator: Full report generation", True)
        else:
            log_test("MetaLearningEvaluator: Full report generation", False,
                    f"Missing keys in report: {report.keys()}")
        
        return True
        
    except Exception as e:
        log_test("MetaLearningEvaluator: Module test", False, f"Exception: {str(e)}")
        traceback.print_exc()
        return False

def test_optimized_convergence_detector():
    """Test OptimizedEnhancedConvergenceDetector module"""
    print("\n=== Testing OptimizedEnhancedConvergenceDetector ===")
    try:
        from optimized_convergence_detector import OptimizedEnhancedConvergenceDetector
        
        detector = OptimizedEnhancedConvergenceDetector(
            window_size=10, 
            volatility_threshold=0.05,
            exploration_threshold=0.1
        )
        log_test("OptimizedConvergenceDetector: Import and initialization", True)
        
        # Test improving phase (low exploration activity should give IMPROVING)
        for i in range(15):
            status = detector.update(100 + i * 5, 0.05)  # Low exploration
        
        if status['state'] == 'IMPROVING':
            log_test("OptimizedConvergenceDetector: Improving state detection", True)
        else:
            log_test("OptimizedConvergenceDetector: Improving state detection", False,
                    f"Expected IMPROVING, got {status['state']}")
        
        # Test convergence detection
        for i in range(20):
            status = detector.update(200.0, 0.01)  # Flat performance, low exploration
        
        if status['state'] in ['CONVERGED', 'EXPLORING']:
            log_test("OptimizedConvergenceDetector: Convergence detection", True)
        else:
            log_test("OptimizedConvergenceDetector: Convergence detection", False,
                    f"Unexpected state: {status['state']}")
        
        # Test statistics
        stats = detector.get_statistics()
        if 'total_updates' in stats and 'current_state' in stats:
            log_test("OptimizedConvergenceDetector: Statistics generation", True)
        else:
            log_test("OptimizedConvergenceDetector: Statistics generation", False,
                    f"Missing keys in stats: {stats.keys()}")
        
        return True
        
    except Exception as e:
        log_test("OptimizedConvergenceDetector: Module test", False, f"Exception: {str(e)}")
        traceback.print_exc()
        return False

def test_rsi_state_arbiter():
    """Test RSIStateArbiter module"""
    print("\n=== Testing RSIStateArbiter ===")
    try:
        from rsi_state_arbiter import RSIStateArbiter, ArbiterState
        
        arbiter = RSIStateArbiter(k_steps_for_warning=3)
        log_test("RSIStateArbiter: Import and initialization", True)
        
        # Test healthy growth (need previous_hv set first)
        # First call to initialize
        arbiter.arbitrate(
            convergence_status={'state': 'IMPROVING', 'converged': False, 'exploring': False},
            new_hv=100.0,
            meta_learning_report={'is_significant': True}
        )
        # Second call with increasing HV
        state = arbiter.arbitrate(
            convergence_status={'state': 'IMPROVING', 'converged': False, 'exploring': False},
            new_hv=110.0,
            meta_learning_report={'is_significant': True}
        )
        
        if state == ArbiterState.HEALTHY_GROWTH:
            log_test("RSIStateArbiter: Healthy growth detection", True)
        else:
            log_test("RSIStateArbiter: Healthy growth detection", False,
                    f"Expected HEALTHY_GROWTH, got {state}")
        
        # Test efficient exploration (exploring with increasing HV)
        state = arbiter.arbitrate(
            convergence_status={'state': 'EXPLORING', 'converged': False, 'exploring': True},
            new_hv=120.0,  # Increasing from 110.0
            meta_learning_report=None
        )
        
        if state == ArbiterState.EFFICIENT_EXPLORATION:
            log_test("RSIStateArbiter: Efficient exploration detection", True)
        else:
            log_test("RSIStateArbiter: Efficient exploration detection", False,
                    f"Expected EFFICIENT_EXPLORATION, got {state}")
        
        # Test inefficient exploration (k consecutive steps)
        for i in range(5):
            state = arbiter.arbitrate(
                convergence_status={'state': 'EXPLORING', 'converged': False, 'exploring': True},
                new_hv=110.0,  # No growth
                meta_learning_report=None
            )
        
        if state == ArbiterState.INEFFICIENT_EXPLORATION:
            log_test("RSIStateArbiter: Inefficient exploration detection", True)
        else:
            log_test("RSIStateArbiter: Inefficient exploration detection", False,
                    f"Expected INEFFICIENT_EXPLORATION, got {state}")
        
        # Test state summary
        summary = arbiter.get_state_summary()
        if 'state_counts' in summary and 'transition_history' in summary:
            log_test("RSIStateArbiter: State summary generation", True)
        else:
            log_test("RSIStateArbiter: State summary generation", False,
                    f"Missing keys in summary: {summary.keys()}")
        
        return True
        
    except Exception as e:
        log_test("RSIStateArbiter: Module test", False, f"Exception: {str(e)}")
        traceback.print_exc()
        return False

def test_fast_pareto_optimizer():
    """Test FastParetoOptimizer module"""
    print("\n=== Testing FastParetoOptimizer ===")
    try:
        from fast_pareto_optimizer import FastParetoOptimizer
        
        optimizer = FastParetoOptimizer(
            objective_directions={'performance': 'maximize', 'cost': 'minimize'}
        )
        log_test("FastParetoOptimizer: Import and initialization", True)
        
        # Add solutions
        solutions = [
            {'performance': 0.9, 'cost': 100},
            {'performance': 0.8, 'cost': 50},
            {'performance': 0.7, 'cost': 30},
            {'performance': 0.6, 'cost': 80},  # Dominated
        ]
        
        for sol in solutions:
            optimizer.add_solution(sol)
        
        frontier = optimizer.get_frontier()
        if len(frontier) == 3:  # Should exclude dominated solution
            log_test("FastParetoOptimizer: Pareto frontier filtering", True)
        else:
            log_test("FastParetoOptimizer: Pareto frontier filtering", False,
                    f"Expected 3 frontier solutions, got {len(frontier)}")
        
        # Test hypervolume calculation
        hv = optimizer.calculate_hypervolume({'performance': 0.0, 'cost': 200})
        if hv > 0:
            log_test("FastParetoOptimizer: Hypervolume calculation", True)
        else:
            log_test("FastParetoOptimizer: Hypervolume calculation", False,
                    f"Expected positive hypervolume, got {hv}")
        
        # Test report generation
        report = optimizer.get_report()
        if 'metrics' in report and 'frontier_size' in report['metrics']:
            log_test("FastParetoOptimizer: Report generation", True)
        else:
            log_test("FastParetoOptimizer: Report generation", False,
                    f"Missing keys in report: {report.keys()}")
        
        return True
        
    except Exception as e:
        log_test("FastParetoOptimizer: Module test", False, f"Exception: {str(e)}")
        traceback.print_exc()
        return False

def test_integrated_rsi_test():
    """Test IntegratedRSITest module"""
    print("\n=== Testing IntegratedRSITest ===")
    try:
        from integrated_rsi_test import IntegratedRSITest, TestPhase
        
        framework = IntegratedRSITest(
            cold_start_steps=20,
            exploration_steps=30,
            convergence_steps=20,
            verification_steps=10
        )
        log_test("IntegratedRSITest: Import and initialization", True)
        
        # Test cold start phase
        result = framework.update(new_hv=100.0, exploration_activity=0.3)
        if result['phase'] == TestPhase.COLD_START.value:
            log_test("IntegratedRSITest: Cold start phase", True)
        else:
            log_test("IntegratedRSITest: Cold start phase", False,
                    f"Expected COLD_START, got {result['phase']}")
        
        # Simulate progression through phases
        for i in range(1, 25):
            result = framework.update(new_hv=100.0 + i, exploration_activity=0.3)
        
        if result['phase'] == TestPhase.ACTIVE_EXPLORATION.value:
            log_test("IntegratedRSITest: Phase transition", True)
        else:
            log_test("IntegratedRSITest: Phase transition", False,
                    f"Expected ACTIVE_EXPLORATION, got {result['phase']}")
        
        # Test summary generation
        summary = framework.get_summary()
        if 'total_steps' in summary and 'phase_history' in summary:
            log_test("IntegratedRSITest: Summary generation", True)
        else:
            log_test("IntegratedRSITest: Summary generation", False,
                    f"Missing keys in summary: {summary.keys()}")
        
        return True
        
    except Exception as e:
        log_test("IntegratedRSITest: Module test", False, f"Exception: {str(e)}")
        traceback.print_exc()
        return False

def print_summary():
    """Print test summary"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in test_results if r['passed'])
    total = len(test_results)
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if total - passed > 0:
        print("\nFailed Tests:")
        for r in test_results:
            if not r['passed']:
                print(f"  - {r['name']}")
                if r['message']:
                    print(f"    {r['message']}")
    
    print("\n" + "="*60)
    
    return passed == total

def main():
    """Main test execution"""
    print("Enhanced RSI Test Framework - Comprehensive Test Suite")
    print("="*60)
    
    # Run all tests
    test_meta_learning_evaluator()
    test_optimized_convergence_detector()
    test_rsi_state_arbiter()
    test_fast_pareto_optimizer()
    test_integrated_rsi_test()
    
    # Print summary
    all_passed = print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
