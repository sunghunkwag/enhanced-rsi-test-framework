# Enhanced RSI Test Framework

Statistically robust RSI (Recursive Self-Improvement) testing framework with meta-learning evaluation, convergence detection, and Pareto optimization.

## Overview

This framework provides production-ready tools for rigorously testing recursive self-improvement (RSI) systems. It addresses three critical challenges in RSI evaluation:

1. **Meta-Learning Validation**: Statistically validate that systems actually exhibit meta-learning 
2. **Convergence Detection**: Distinguish productive exploration plateaus from true convergence
3. **Multi-Objective Optimization**: Balance performance, efficiency, and complexity trade-offs

## Key Features

### AdvancedMetaLearningEvaluator
- **Incremental Updates**: Add data points progressively with `update()` method
- **Adaptive Alpha Selection**: Automatically optimizes EMA smoothing parameter via one-step-ahead prediction error minimization
- **Nonlinear Trend Modeling**: AIC-based comparison of linear vs quadratic trends to detect acceleration patterns
- **Block Bootstrap**: Time-series preserving statistical resampling for robust confidence intervals
- **Statistical Significance**: Hypothesis testing for trend coefficients with confidence intervals

### OptimizedEnhancedConvergenceDetector
- **O(1) Complexity**: Constant-time updates using deque and running sums
- **State Classification**: Returns IMPROVING, EXPLORING, or CONVERGED states
- **Volatility Analysis**: Coefficient of variation to detect exploration vs exploitation phases
- **Memory Efficient**: Automatic sliding window management with configurable size

### FastParetoOptimizer
- **Named Objectives**: Support for maximize/minimize directions per objective
- **Frontier Tracking**: Maintains non-dominated solutions in multi-objective space
- **Hypervolume Calculation**: Exact computation for 2D, Monte Carlo approximation for higher dimensions
- **Quality Metrics**: Spacing (uniformity) and spread (coverage) indicators
- **O(log n) Operations**: Efficient insertion and dominance checking using sorted containers

### RSIStateArbiter
- **Inter-Module Validation**: Cross-validates outputs from multiple modules
- **7 System States**: HEALTHY_GROWTH, EFFICIENT_EXPLORATION, INEFFICIENT_EXPLORATION, TRUE_CONVERGENCE, CRITICAL_FAILURE, MODULE_CONFLICT, INITIALIZING
- **Inconsistency Detection**: Identifies when modules disagree
- **State Transition Tracking**: Maintains history for debugging and analysis

### IntegratedRSITest
- **Phased Testing**: Four-phase lifecycle (cold start, exploration, convergence watch, verification)
- **Adaptive Policies**: Different evaluation frequencies per phase
- **Orchestration**: Coordinates all modules for comprehensive RSI testing

## Installation

```bash
# Clone repository
git clone https://github.com/sunghunkwag/enhanced-rsi-test-framework.git
cd enhanced-rsi-test-framework

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Meta-Learning Evaluation

```python
from meta_learning_evaluator import AdvancedMetaLearningEvaluator

# Initialize evaluator
evaluator = AdvancedMetaLearningEvaluator(
    min_data_points=15,
    bootstrap_samples=5000
)

# Update with iteration data incrementally
for iteration in range(100):
    performance = evaluate_model(iteration)
    evaluator.update(iteration, performance)

# Get comprehensive report
report = evaluator.get_report()
print(f"Interpretation: {report['final_interpretation']}")
print(f"Significant: {report['is_significant']}")
print(f"Best model: {report['trend_model_comparison']['best_model']}")
```

### Convergence Detection

```python
from optimized_convergence_detector import OptimizedEnhancedConvergenceDetector

# Initialize detector
detector = OptimizedEnhancedConvergenceDetector(
    window_size=50,
    volatility_threshold=0.05,
    exploration_threshold=0.1
)

# Update with hypervolume and exploration activity
for iteration in range(num_iterations):
    hv = calculate_hypervolume()
    exploration = get_exploration_metric()
    
    status = detector.update(hv, exploration)
    
    print(f"State: {status['state']}")
    print(f"Converged: {status['converged']}")
    print(f"Volatility: {status['volatility']:.4f}")
    
    if status['state'] == 'CONVERGED':
        print("Convergence detected - stopping")
        break
```

### Pareto Optimization

```python
from fast_pareto_optimizer import FastParetoOptimizer

# Initialize optimizer with named objectives
optimizer = FastParetoOptimizer(
    objective_directions={
        'performance': 'maximize',
        'efficiency': 'maximize',
        'complexity': 'minimize'
    }
)

# Add solutions
for model in evaluated_models:
    result = optimizer.add_solution(
        objectives={
            'performance': model.accuracy,
            'efficiency': model.speed,
            'complexity': model.parameter_count
        },
        metadata={'model_id': model.id}
    )
    
    if result['added']:
        print(f"Added to frontier. New size: {result['frontier_size']}")
        print(f"Hypervolume: {result['hypervolume']:.4f}")

# Get frontier analysis
report = optimizer.get_report()
print(f"Frontier size: {report['metrics']['frontier_size']}")
print(f"Spacing: {report['metrics']['spacing']:.4f}")
print(f"Recommendation: {report['recommendation']}")

# Calculate hypervolume with reference point
reference = {'performance': 0.0, 'efficiency': 0.0, 'complexity': 1000000}
hypervolume = optimizer.calculate_hypervolume(reference)
print(f"Hypervolume: {hypervolume:.4f}")
```

### RSI State Arbitration

```python
from rsi_state_arbiter import RSIStateArbiter, ArbiterState

# Initialize arbiter
arbiter = RSIStateArbiter(k_steps_for_warning=5)

# Arbitrate system state
convergence_status = detector.update(new_hv, exploration)
state = arbiter.arbitrate(
    convergence_status=convergence_status,
    new_hv=new_hv,
    meta_learning_report=evaluator.get_report()
)

if state == ArbiterState.HEALTHY_GROWTH:
    print("System is improving well")
elif state == ArbiterState.INEFFICIENT_EXPLORATION:
    print("Warning: Exploring without frontier growth")
elif state == ArbiterState.CRITICAL_FAILURE:
    print("Critical: Hypervolume decreased!")
```

### Integrated Testing

```python
from integrated_rsi_test import IntegratedRSITest

# Initialize integrated framework
framework = IntegratedRSITest(
    cold_start_steps=100,
    exploration_steps=500,
    convergence_steps=300,
    verification_steps=200
)

# Run RSI test
for iteration in range(1000):
    # Your RSI system iteration here
    new_hv = optimizer.get_current_hypervolume()
    exploration = calculate_exploration_metric()
    
    result = framework.update(
        new_hv=new_hv,
        exploration_activity=exploration,
        meta_learning_report=evaluator.get_report()
    )
    
    print(f"Phase: {result['phase']}")
    print(f"State: {result['arbiter_state']}")
    print(f"Recommendations: {result['recommendations']}")
    
    if not result['should_continue']:
        print("Framework recommends stopping")
        break

# Get final summary
summary = framework.get_summary()
print(f"Total steps: {summary['total_steps']}")
print(f"Final phase: {summary['current_phase']}")
```

## Architecture

```
enhanced-rsi-test-framework/
├── meta_learning_evaluator.py           # Meta-learning validation
├── optimized_convergence_detector.py    # O(1) convergence detection
├── fast_pareto_optimizer.py             # Pareto frontier optimization
├── rsi_state_arbiter.py                 # Inter-module consistency
├── integrated_rsi_test.py               # Integrated testing framework
├── test_framework.py                    # Comprehensive test suite
├── requirements.txt                     # Dependencies
├── CHANGELOG.md                         # Version history
├── ANALYSIS_REPORT.md                   # Problem analysis documentation
├── LICENSE                              # MIT License
└── README.md                            # This file
```

## API Reference

### AdvancedMetaLearningEvaluator

**Constructor**:
```python
__init__(
    min_data_points: int = 15,
    bootstrap_samples: int = 5000,
    confidence_level: float = 0.95,
    alphas_to_try: List[float] = None
)
```

**Methods**:
- `update(iteration: int, performance: float)`: Add new iteration data point
- `get_report() -> Dict`: Generate comprehensive analysis

**Report Structure**:
```python
{
    'final_interpretation': str,
    'is_significant': bool,
    'data_points': int,
    'adaptive_smoothing': {
        'optimal_alpha': float,
        'smoothed_history': List[float],
        'smoothed_speeds': List[float]
    },
    'trend_model_comparison': {
        'best_model': str,  # 'linear' or 'quadratic'
        'linear_aic': float,
        'quadratic_aic': float,
        'best_params': List[float]
    },
    'robustness_analysis': {
        'is_robust': bool,
        'confidence_interval': Tuple[float, float],
        'n_bootstraps': int
    }
}
```

### OptimizedEnhancedConvergenceDetector

**Constructor**:
```python
__init__(
    window_size: int = 50,
    volatility_threshold: float = 0.05,
    exploration_threshold: float = 0.1
)
```

**Methods**:
- `update(new_hv: float, exploration_activity: float) -> Dict`: Update with new data
- `get_statistics() -> Dict`: Get current detector statistics
- `reset()`: Reset detector state

**Return Structure**:
```python
{
    'state': str,  # 'IMPROVING', 'EXPLORING', or 'CONVERGED'
    'converged': bool,
    'volatility': float,
    'exploring': bool,
    'steps_since_exploration': int,
    'current_hv': float,
    'mean_hv': float
}
```

### FastParetoOptimizer

**Constructor**:
```python
__init__(
    objective_directions: Dict[str, str] = None,
    num_objectives: int = None,
    reference_point: Optional[List[float]] = None
)
```

Directions: `'maximize'` or `'minimize'` for each objective

**Methods**:
- `add_solution(objectives: Dict[str, float], metadata: Optional[Dict] = None) -> Dict`
- `get_frontier() -> List[Dict[str, float]]`: Get current Pareto frontier
- `calculate_hypervolume(reference: Dict[str, float]) -> float`
- `get_report() -> Dict`: Comprehensive frontier analysis
- `reset()`: Clear frontier

**Add Solution Return**:
```python
{
    'added': bool,
    'dominated_by_frontier': bool,
    'solutions_removed': int,
    'frontier_size': int,
    'hypervolume': float,
    'hv_delta': float
}
```

### RSIStateArbiter

**Constructor**:
```python
__init__(
    k_steps_for_warning: int = 10,
    hv_epsilon: float = 1e-9
)
```

**Methods**:
- `arbitrate(convergence_status: Dict, new_hv: float, meta_learning_report: Dict = None) -> ArbiterState`
- `get_state_summary() -> Dict`: Get state distribution and history
- `check_for_persistent_warning() -> Tuple[bool, str]`

**States**:
- `HEALTHY_GROWTH`: Performance and hypervolume both improving
- `EFFICIENT_EXPLORATION`: Exploring with hypervolume growth
- `INEFFICIENT_EXPLORATION`: Exploring without frontier expansion
- `TRUE_CONVERGENCE`: System has converged
- `CRITICAL_FAILURE`: Hypervolume decreased
- `MODULE_CONFLICT`: Modules disagree on state
- `INITIALIZING`: Collecting initial data

### IntegratedRSITest

**Constructor**:
```python
__init__(
    cold_start_steps: int = 100,
    exploration_steps: int = 500,
    convergence_steps: int = 300,
    verification_steps: int = 200
)
```

**Methods**:
- `update(new_hv: float, exploration_activity: float, meta_learning_report: Optional[Dict] = None) -> Dict`
- `get_summary() -> Dict`: Get comprehensive test summary
- `reset()`: Reset framework state

## Testing

Run the comprehensive test suite:

```bash
python test_framework.py
```

All 20 tests should pass with 100% success rate.

## Requirements

- Python >= 3.8
- numpy >= 1.24.0
- scipy >= 1.10.0
- sortedcontainers >= 2.4.0

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

**Current Version**: 0.2.0
- Fixed all API inconsistencies
- Added comprehensive test suite
- Enhanced documentation

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Key areas for enhancement:
- Additional convergence detection strategies
- Alternative trend modeling approaches
- Higher-dimensional hypervolume computation
- Visualization utilities
- Performance benchmarks

## Citation

If you use this framework in research, please cite:

```bibtex
@software{enhanced_rsi_framework,
  author = {Kwag, Sunghun},
  title = {Enhanced RSI Test Framework},
  version = {0.2.0},
  year = {2025},
  url = {https://github.com/sunghunkwag/enhanced-rsi-test-framework}
}
```

## Development Philosophy

This framework emerged from iterative development focusing on continuous improvement. Each iteration addressed limitations identified in the previous version, demonstrating practical RSI principles in the framework's own development.

The framework prioritizes:
- **Statistical Rigor**: All claims backed by proper statistical testing
- **Computational Efficiency**: O(1) and O(log n) operations where possible
- **Production Readiness**: Comprehensive testing and error handling
- **API Consistency**: Clear, documented interfaces across all modules
