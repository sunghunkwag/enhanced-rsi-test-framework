# Enhanced RSI Test Framework

Statistically robust RSI (Recursive Self-Improvement) testing framework with meta-learning evaluation, convergence detection, and Pareto optimization.

## Overview

This framework provides production-ready tools for rigorously testing recursive self-improvement (RSI) systems. It addresses three critical challenges in RSI evaluation:

1. **Meta-Learning Validation**: Statistically validate that systems actually exhibit meta-learning 
2. **Convergence Detection**: Distinguish productive exploration plateaus from true convergence
3. **Multi-Objective Optimization**: Balance performance, efficiency, and complexity trade-offs

## Key Features

### AdvancedMetaLearningEvaluator
- **Adaptive Alpha Selection**: Automatically optimizes EMA smoothing parameter via one-step-ahead prediction error minimization
- **Nonlinear Trend Modeling**: AIC-based comparison of linear vs quadratic trends to detect acceleration patterns
- **Block Bootstrap**: Time-series preserving statistical resampling for robust confidence intervals
- **Statistical Significance**: p-value based hypothesis testing for trend coefficients

### EnhancedConvergenceDetector
- **State Classification**: Distinguishes IMPROVING, PRODUCTIVE_PLATEAU, TRUE_CONVERGENCE, and DIVERGING states
- **Architecture Tracking**: Monitors structural changes via cosine similarity of architecture vectors
- **Probation Period**: Grace period mechanism prevents premature termination during temporary plateaus
- **Volatility Analysis**: Coefficient of variation to detect exploration vs exploitation phases

### ParetoOptimizer
- **Frontier Tracking**: Maintains non-dominated solutions in multi-objective space
- **Hypervolume Calculation**: Exact computation for 2D, Monte Carlo approximation for higher dimensions
- **Quality Metrics**: Spacing (uniformity) and spread (coverage) indicators
- **Domination Filtering**: Efficient Pareto dominance checking

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

# Update with iteration speeds
for iteration, speed in enumerate(iteration_speeds):
    evaluator.update(iteration, speed)

# Get comprehensive report
report = evaluator.get_report()
print(f"Interpretation: {report['final_interpretation']}")
print(f"Optimal alpha: {report['adaptive_smoothing']['optimal_alpha']}")
print(f"Best model: {report['trend_model_comparison']['best_model']}")
```

### Convergence Detection

```python
from convergence_detector import EnhancedConvergenceDetector, ConvergenceState
import numpy as np

# Initialize detector
detector = EnhancedConvergenceDetector(
    window_size=10,
    convergence_threshold=0.01,
    structural_threshold=0.05
)

# Update with performance and architecture
for iteration in range(num_iterations):
    performance = evaluate_model()
    arch_vector = np.array(model.get_architecture_vector())
    
    state = detector.update(performance, arch_vector)
    
    if state == ConvergenceState.TRUE_CONVERGENCE:
        print("Convergence detected - stopping")
        break
    elif state == ConvergenceState.PRODUCTIVE_PLATEAU:
        print("Productive plateau - continuing")
```

### Pareto Optimization

```python
from pareto_optimizer import ParetoOptimizer

# Initialize optimizer
optimizer = ParetoOptimizer({
    'performance': 'maximize',
    'efficiency': 'maximize',
    'complexity': 'minimize'
})

# Add solutions
for model in evaluated_models:
    objectives = {
        'performance': model.accuracy,
        'efficiency': model.speed,
        'complexity': model.parameter_count
    }
    on_frontier = optimizer.add_solution(objectives, metadata={'model_id': model.id})
    
# Get frontier analysis
report = optimizer.get_report()
print(f"Frontier size: {report['metrics']['frontier_size']}")
print(f"Spacing: {report['metrics']['spacing']}")
print(f"Recommendation: {report['recommendation']}")

# Calculate hypervolume
reference = {'performance': 0.0, 'efficiency': 0.0, 'complexity': 1000000}
hypervolume = optimizer.calculate_hypervolume(reference)
print(f"Hypervolume: {hypervolume}")
```

## Architecture

```
enhanced-rsi-test-framework/
├── meta_learning_evaluator.py    # Meta-learning validation
├── convergence_detector.py       # Convergence detection
├── pareto_optimizer.py          # Multi-objective optimization
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Development History

This framework emerged from 10+ iterative exchanges with Gemini, focusing on continuous improvement:

- **Iteration 1-3**: Basic implementations with fixed hyperparameters
- **Iteration 4-6**: Added adaptive alpha selection and AIC-based model comparison
- **Iteration 7-9**: Integrated block bootstrap for time-series data
- **Iteration 10+**: Refined convergence probation and Pareto metrics

Each iteration addressed limitations identified in the previous version, demonstrating practical RSI principles in the framework's own development.

## API Reference

### AdvancedMetaLearningEvaluator

**Constructor**:
```python
__init__(min_data_points=15, bootstrap_samples=5000, confidence_level=0.95)
```

**Methods**:
- `update(iteration: int, speed: float)`: Add new iteration data
- `get_report() -> Dict`: Generate comprehensive analysis

**Report Structure**:
```python
{
    'final_interpretation': str,
    'adaptive_smoothing': {
        'optimal_alpha': float,
        'smoothed_history': List[float],
        'smoothed_speeds': List[float]
    },
    'trend_model_comparison': {
        'best_model': str,
        'linear_aic': float,
        'quadratic_aic': float,
        'best_params': List[float]
    },
    'robustness_analysis': {
        'is_robust': bool,
        'p_value': float,
        'confidence_interval': Tuple[float, float]
    }
}
```

### EnhancedConvergenceDetector

**Constructor**:
```python
__init__(window_size=10, convergence_threshold=0.01, 
         structural_threshold=0.05, probation_period=5)
```

**Methods**:
- `update(performance: float, architecture_vector: Optional[np.ndarray]) -> ConvergenceState`
- `get_report() -> Dict`: Get convergence metrics

**States**:
- `IMPROVING`: System actively improving
- `PRODUCTIVE_PLATEAU`: Exploring without performance gains
- `TRUE_CONVERGENCE`: No progress for probation period
- `DIVERGING`: Performance degrading

### ParetoOptimizer

**Constructor**:
```python
__init__(objective_directions: Dict[str, str])
```
Directions: 'maximize' or 'minimize' for each objective

**Methods**:
- `add_solution(objectives: Dict[str, float], metadata: Optional[Dict]) -> bool`
- `get_frontier() -> List[Solution]`: Get current Pareto frontier
- `calculate_hypervolume(reference_point: Dict[str, float]) -> float`
- `get_report() -> Dict`: Comprehensive frontier analysis

## Requirements

- Python >= 3.8
- numpy >= 1.24.0
- scipy >= 1.10.0

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Key areas for enhancement:
- Additional convergence detection strategies
- Alternative trend modeling approaches
- Higher-dimensional hypervolume computation
- Visualization utilities

## Citation

If you use this framework in research, please cite:

```bibtex
@software{enhanced_rsi_framework,
  author = {Kwag, Sunghun},
  title = {Enhanced RSI Test Framework},
  year = {2025},
  url = {https://github.com/sunghunkwag/enhanced-rsi-test-framework}
}
```

## Acknowledgments

Developed through iterative improvement dialogue with Google Gemini, demonstrating recursive self-improvement principles in practice.
