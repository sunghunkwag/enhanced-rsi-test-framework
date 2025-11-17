# Enhanced RSI Test Framework

Statistically robust RSI (Recursive Self-Improvement) testing framework with meta-learning evaluation, convergence detection, and Pareto optimization.

## Table of Contents
- Overview
- Installation
- Quick Start
- Key Modules
- Architecture
- API Reference
- Testing
- Requirements
- Version History
- License
- Contributing
- Citation
- Development Philosophy

## Overview

This framework provides tools for diagnostic evaluation of recursive self-improvement (RSI) systems. Major components include:

- Meta-learning evaluation
- Convergence state detection for exploration/convergence phases
- Multi-objective Pareto optimization

## Installation

```bash
git clone https://github.com/sunghunkwag/enhanced-rsi-test-framework.git
cd enhanced-rsi-test-framework
pip install -r requirements.txt
```

## Quick Start
### Meta-Learning Evaluation
```python
from meta_learning_evaluator import AdvancedMetaLearningEvaluator
evaluator = AdvancedMetaLearningEvaluator(min_data_points=15, bootstrap_samples=5000)
for iteration in range(100):
    performance = evaluate_model(iteration)
    evaluator.update(iteration, performance)
report = evaluator.get_report()
```

### Convergence Detection
```python
from optimized_convergence_detector import OptimizedEnhancedConvergenceDetector
detector = OptimizedEnhancedConvergenceDetector(window_size=50, volatility_threshold=0.05, exploration_threshold=0.1)
# use detector.update() per iteration
```

### Pareto Optimization
```python
from fast_pareto_optimizer import FastParetoOptimizer
optimizer = FastParetoOptimizer(objective_directions={
    'performance': 'maximize',
    'efficiency': 'maximize',
    'complexity': 'minimize'
})
# use optimizer.add_solution(), optimizer.get_report()
```

## Key Modules
- meta_learning_evaluator.py: Meta-learning validation
- optimized_convergence_detector.py: O(1) convergence detection
- fast_pareto_optimizer.py: Pareto frontier optimization
- rsi_state_arbiter.py: State arbitration
- integrated_rsi_test.py: Integrated multi-phase RSI test

## Architecture
```
enhanced-rsi-test-framework/
├── meta_learning_evaluator.py
├── optimized_convergence_detector.py
├── fast_pareto_optimizer.py
├── rsi_state_arbiter.py
├── integrated_rsi_test.py
├── test_framework.py
├── requirements.txt
├── CHANGELOG.md
├── ANALYSIS_REPORT.md
├── LICENSE
└── README.md
```

## API Reference
API methods follow Python docstrings for each module. Major module constructors and methods:
- AdvancedMetaLearningEvaluator: `update()`, `get_report()`
- OptimizedEnhancedConvergenceDetector: `update()`, `get_statistics()`, `reset()`
- FastParetoOptimizer: `add_solution()`, `get_report()`, `calculate_hypervolume()`
- RSIStateArbiter: `arbitrate()`, `get_state_summary()`, `check_for_persistent_warning()`
- IntegratedRSITest: `update()`, `get_summary()`, `reset()`

## Testing
Run the full test suite:
```bash
python test_framework.py
```
All tests should pass.

## Requirements
- Python >= 3.8
- numpy >= 1.24.0
- scipy >= 1.10.0
- sortedcontainers >= 2.4.0

## Version History
See CHANGELOG.md for details. Current: v0.2.0 (API fixes, comprehensive test suite)

## License
MIT License

## Contributing
Contributions and additional module implementations are encouraged. Please fork and submit PRs for enhancements, new detection strategies, visualizations, and benchmarks.

## Citation
If used in academic work:
```
@software{enhanced_rsi_framework,
  author = {Kwag, Sunghun},
  title = {Enhanced RSI Test Framework},
  version = {0.2.0},
  year = {2025},
  url = {https://github.com/sunghunkwag/enhanced-rsi-test-framework}
}
```

## Development Philosophy
Iterative development and continuous improvement, with statistical rigor, computational efficiency, and full production readiness throughout all modules.
