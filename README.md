# Enhanced RSI Test Framework

A statistically robust testing framework for Recursive Self-Improvement (RSI) systems, featuring meta-learning evaluation, convergence detection, and Pareto optimization.

## Overview

This framework provides a suite of tools for the diagnostic evaluation of RSI systems. Its major components include:

-   **Meta-Learning Evaluation**: Statistically validates learning acceleration using advanced time-series analysis.
-   **Convergence Detection**: Offers a memory-efficient O(1) detector to distinguish between improvement, exploration, and convergence phases.
-   **Multi-Objective Pareto Optimization**: Provides a fast optimizer for analyzing trade-offs between multiple objectives (e.g., performance vs. complexity).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sunghunkwag/enhanced-rsi-test-framework.git
    cd enhanced-rsi-test-framework
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

### Meta-Learning Evaluation
```python
from meta_learning_evaluator import AdvancedMetaLearningEvaluator

# Initialize with desired statistical parameters
evaluator = AdvancedMetaLearningEvaluator(min_data_points=15, bootstrap_samples=5000)

# Update with performance data each iteration
for i in range(20):
    performance = 100 + i * 2.0 + (i % 5) # Example data
    evaluator.update(iteration=i, performance=performance)

# Generate a full report once enough data is collected
report = evaluator.get_report()
print(report['final_interpretation'])
```

### Convergence Detection
```python
from optimized_convergence_detector import OptimizedEnhancedConvergenceDetector

detector = OptimizedEnhancedConvergenceDetector(
    window_size=50,
    volatility_threshold=0.05,
    exploration_threshold=0.1
)

# Update per iteration with new hypervolume and exploration data
status = detector.update(new_hv=150.5, exploration_activity=0.25)
print(f"Current State: {status['state']}")
```

### Pareto Optimization
```python
from fast_pareto_optimizer import FastParetoOptimizer

optimizer = FastParetoOptimizer(
    objective_directions={'performance': 'maximize', 'cost': 'minimize'},
    reference_point={'performance': 0.0, 'cost': 200.0} # Set a baseline for hypervolume
)

# Add solutions to the optimizer
optimizer.add_solution({'performance': 0.9, 'cost': 100})
optimizer.add_solution({'performance': 0.8, 'cost': 50})

# Get a report with metrics and the current Pareto frontier
report = optimizer.get_report()
print(f"Frontier size: {report['metrics']['frontier_size']}")
print(f"Hypervolume: {report['metrics']['hypervolume']:.4f}")
```

## Key Modules

-   `meta_learning_evaluator.py`: Meta-learning validation.
-   `optimized_convergence_detector.py`: O(1) convergence detection.
-   `fast_pareto_optimizer.py`: Pareto frontier optimization.
-   `rsi_state_arbiter.py`: State arbitration logic.
-   `integrated_rsi_test.py`: An integrated, multi-phase RSI test harness.

## Architecture

```
enhanced-rsi-test-framework/
├── fast_pareto_optimizer.py
├── integrated_rsi_test.py
├── meta_learning_evaluator.py
├── optimized_convergence_detector.py
├── rsi_state_arbiter.py
├── test_integration.py
├── requirements.txt
├── CHANGELOG.md
├── LICENSE
└── README.md
```

## API Reference

API methods are documented via Python docstrings within each module. Key methods include:

-   **AdvancedMetaLearningEvaluator**: `update()`, `get_report()`
-   **OptimizedEnhancedConvergenceDetector**: `update()`, `get_statistics()`, `reset()`
-   **FastParetoOptimizer**: `add_solution()`, `get_report()`, `reset()`
-   **RSIStateArbiter**: `arbitrate()`, `get_state_summary()`
-   **IntegratedRSITest**: `update()`, `get_summary()`, `reset()`

## Testing

This project uses `pytest` for automated testing. To run the full test suite:

```bash
pytest
```
All tests are expected to pass.

## Requirements

-   Python >= 3.8
-   numpy >= 1.24.0
-   scipy >= 1.10.0
-   sortedcontainers >= 2.4.0
-   pytest >= 7.0.0 (for testing)

## Version History

See `CHANGELOG.md` for a detailed history of changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request with your enhancements, new features, or bug fixes.

## Development Philosophy

This framework is developed with a focus on iterative improvement, statistical rigor, computational efficiency, and production readiness.
