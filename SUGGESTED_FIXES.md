# Enhanced RSI Test Framework - 구체적인 수정 제안

이 문서는 발견된 문제점에 대한 구체적인 코드 수정 제안을 포함합니다.

## 1. OptimizedEnhancedConvergenceDetector 수정

### 문제
`update()` 메서드의 반환값에 `state` 필드가 없어 `IntegratedRSITest`와 호환되지 않음.

### 수정 제안

```python
def update(self, new_hv: float, exploration_activity: float) -> Dict[str, any]:
    """
    Update detector with new hypervolume and exploration activity.
    
    Args:
        new_hv: New hypervolume value
        exploration_activity: Measure of current exploration (e.g., mutation rate)
        
    Returns:
        Dict with convergence status and metrics
    """
    # Remove oldest value from sums if window is full
    if len(self.hv_history) == self.window_size:
        oldest_hv = self.hv_history[0]
        self.hv_sum -= oldest_hv
        self.hv_squared_sum -= oldest_hv ** 2
    
    # Add new value
    self.hv_history.append(new_hv)
    self.squared_hv_history.append(new_hv ** 2)
    self.hv_sum += new_hv
    self.hv_squared_sum += new_hv ** 2
    
    # Update exploration tracking
    is_exploring = exploration_activity > self.exploration_threshold
    if is_exploring:
        self.exploration_volatility = exploration_activity
        self.steps_since_last_exploration = 0
    else:
        self.steps_since_last_exploration += 1
    
    # Calculate convergence status
    converged, volatility = self._check_convergence()
    
    # Determine state string for compatibility
    if converged:
        state = 'CONVERGED'
    elif is_exploring:
        state = 'EXPLORING'
    else:
        state = 'IMPROVING'
    
    return {
        'state': state,  # 추가된 필드
        'converged': converged,
        'volatility': volatility,
        'exploring': is_exploring,
        'steps_since_exploration': self.steps_since_last_exploration,
        'current_hv': new_hv,
        'mean_hv': self.hv_sum / len(self.hv_history) if self.hv_history else 0.0
    }

def get_statistics(self) -> Dict[str, float]:
    """Get current statistics about detector state."""
    n = len(self.hv_history)
    
    if n == 0:
        return {
            'total_updates': 0,  # 추가된 필드
            'current_state': 'INITIALIZING',  # 추가된 필드
            'window_fill': 0.0,
            'mean_hv': 0.0,
            'volatility': float('inf'),
            'exploration_volatility': self.exploration_volatility
        }
    
    mean = self.hv_sum / n
    mean_of_squares = self.hv_squared_sum / n
    variance = max(0.0, mean_of_squares - (mean ** 2))
    volatility = np.sqrt(variance) / mean if mean > 0 else float('inf')
    
    # Determine current state
    converged = volatility < self.volatility_threshold
    is_exploring = self.steps_since_last_exploration == 0
    
    if converged:
        current_state = 'CONVERGED'
    elif is_exploring:
        current_state = 'EXPLORING'
    else:
        current_state = 'IMPROVING'
    
    return {
        'total_updates': n,  # 추가된 필드
        'current_state': current_state,  # 추가된 필드
        'window_fill': n / self.window_size,
        'mean_hv': mean,
        'volatility': volatility,
        'exploration_volatility': self.exploration_volatility,
        'steps_since_exploration': self.steps_since_last_exploration
    }
```

---

## 2. RSIStateArbiter 수정

### 문제 1: 생성자 파라미터 이름 불일치

```python
# 수정 전
def __init__(self, inefficient_k_steps: int = 10, hv_epsilon: float = 1e-9):
    self.k_steps_for_warning = inefficient_k_steps

# 수정 후
def __init__(self, k_steps_for_warning: int = 10, hv_epsilon: float = 1e-9):
    self.k_steps_for_warning = k_steps_for_warning
```

### 문제 2: arbitrate() 메서드 시그니처 불일치

```python
# 수정 전
def arbitrate(self, convergence_status: str, new_hv: float, 
              meta_learning_report: Dict) -> Tuple[ArbiterState, str]:

# 수정 후
def arbitrate(self, convergence_status: Dict, new_hv: float, 
              meta_learning_report: Optional[Dict] = None) -> ArbiterState:
    """
    Determine system state from module inputs.
    
    Args:
        convergence_status: Dict with 'state', 'converged', 'exploring' keys
        new_hv: New hypervolume value
        meta_learning_report: Optional meta-learning evaluation report
        
    Returns:
        ArbiterState enum value
    """
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
    
    return state  # 튜플이 아닌 단일 값 반환

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
            for step, state, reason in self.state_history[-10:]  # 최근 10개
        ],
        'total_steps': total
    }
```

---

## 3. AdvancedMetaLearningEvaluator 수정

### 문제
생성자가 전체 히스토리를 요구하여 점진적 업데이트 불가능.

### 수정 제안

```python
class AdvancedMetaLearningEvaluator:
    """
    RSI meta-learning acceleration validator with statistical robustness.
    
    Features:
    1. Adaptive Alpha Selection: Automatically optimizes EMA smoothing parameter
    2. Nonlinear Trend Modeling: Compares linear vs quadratic trend models
    3. Block Bootstrap: Preserves time-series autocorrelation in confidence intervals
    """
    
    def __init__(self, min_data_points: int = 15, 
                 bootstrap_samples: int = 5000,
                 confidence_level: float = 0.95,
                 alphas_to_try: list = None):
        """
        Args:
            min_data_points: Minimum data points required for analysis
            bootstrap_samples: Number of bootstrap samples for robustness analysis
            confidence_level: Confidence level for statistical tests
            alphas_to_try: Candidate alpha values for optimal EMA selection
        """
        self.min_data_points = min_data_points
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.alphas_to_try = alphas_to_try or [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        
        # Data storage
        self.iterations = []
        self.performance_history = []
        
        # Analysis results (computed on demand)
        self._analysis_cache = None
    
    def update(self, iteration: int, performance: float):
        """
        Add new iteration data point.
        
        Args:
            iteration: Iteration number
            performance: Performance metric value
        """
        self.iterations.append(iteration)
        self.performance_history.append(performance)
        
        # Invalidate cache
        self._analysis_cache = None
    
    def get_report(self) -> Dict:
        """
        Generate comprehensive analysis report.
        
        Returns:
            Dict with analysis results or insufficient data message
        """
        if len(self.performance_history) < self.min_data_points:
            return {
                'final_interpretation': f'Insufficient data: {len(self.performance_history)}/{self.min_data_points} points',
                'data_points': len(self.performance_history),
                'required_points': self.min_data_points
            }
        
        # Use cached results if available
        if self._analysis_cache is not None:
            return self._analysis_cache
        
        # Perform analysis
        original_history = np.array(self.performance_history)
        
        # 1. Adaptive Alpha Selection
        optimal_alpha, smoothed_history = self._find_optimal_alpha(
            original_history, self.alphas_to_try
        )
        
        # Calculate smoothed speeds (1st derivative)
        smoothed_speeds = smoothed_history[1:] - smoothed_history[:-1]
        
        # 2. Nonlinear Trend Model Comparison
        trend_analysis = self._compare_trend_models(smoothed_speeds)
        
        # 3. Block Bootstrap
        n_speeds = len(smoothed_speeds)
        block_length = max(1, int(n_speeds ** (1/3)))
        robustness_report = self._get_robustness_analysis(
            smoothed_speeds,
            n_bootstraps=self.bootstrap_samples,
            confidence_level=self.confidence_level
        )
        
        # Generate interpretation
        is_significant = robustness_report.get('is_robust', False)
        best_model = trend_analysis['best_model']
        
        if is_significant:
            if best_model == 'quadratic':
                interpretation = "Strong meta-learning: Accelerating improvement detected (quadratic trend)"
            else:
                interpretation = "Meta-learning detected: Linear improvement trend confirmed"
        else:
            interpretation = "No significant meta-learning: Improvement not statistically robust"
        
        # Cache and return results
        self._analysis_cache = {
            'final_interpretation': interpretation,
            'is_significant': is_significant,
            'adaptive_smoothing': {
                'optimal_alpha': optimal_alpha,
                'smoothed_history': smoothed_history.tolist(),
                'smoothed_speeds': smoothed_speeds.tolist()
            },
            'trend_model_comparison': trend_analysis,
            'robustness_analysis': robustness_report,
            'data_points': len(self.performance_history)
        }
        
        return self._analysis_cache
    
    # ... (나머지 메서드는 동일)
```

---

## 4. FastParetoOptimizer 수정

### 문제
`objective_directions` 파라미터 지원 부재.

### 수정 제안

```python
class FastParetoOptimizer:
    """Efficient Pareto frontier optimizer using sorted containers."""
    
    def __init__(self, objective_directions: Dict[str, str] = None,
                 num_objectives: int = None,
                 reference_point: Optional[List[float]] = None):
        """
        Args:
            objective_directions: Dict mapping objective names to 'maximize' or 'minimize'
                                 e.g., {'performance': 'maximize', 'cost': 'minimize'}
            num_objectives: Number of objectives (used if objective_directions not provided)
            reference_point: Reference point for hypervolume calculation
        """
        if objective_directions is not None:
            self.objective_names = list(objective_directions.keys())
            self.num_objectives = len(self.objective_names)
            self.directions = [objective_directions[name] for name in self.objective_names]
        elif num_objectives is not None:
            self.objective_names = [f'obj_{i}' for i in range(num_objectives)]
            self.num_objectives = num_objectives
            self.directions = ['maximize'] * num_objectives
        else:
            raise ValueError("Either objective_directions or num_objectives must be provided")
        
        self.reference_point = reference_point or [0.0] * self.num_objectives
        
        # Use SortedList for O(log n) insertion and efficient range queries
        self.frontier = SortedList(key=lambda x: x[0])
        
        # Track hypervolume history
        self.hv_history = []
        
        # Metadata storage
        self.solution_metadata = {}
    
    def add_solution(self, objectives: Dict[str, float], 
                    metadata: Optional[Dict] = None) -> dict:
        """
        Add solution to frontier and update Pareto set.
        
        Args:
            objectives: Dict mapping objective names to values
            metadata: Optional metadata to store with solution
            
        Returns:
            Dict with update information
        """
        # Convert dict to tuple, applying direction transformations
        solution_tuple = tuple(
            objectives[name] if self.directions[i] == 'maximize' 
            else -objectives[name]
            for i, name in enumerate(self.objective_names)
        )
        
        if len(solution_tuple) != self.num_objectives:
            raise ValueError(f"Expected {self.num_objectives} objectives, got {len(solution_tuple)}")
        
        # Check if solution is dominated by existing frontier
        if self._is_dominated(solution_tuple):
            return {
                'added': False,
                'dominated_by_frontier': True,
                'solutions_removed': 0,
                'frontier_size': len(self.frontier)
            }
        
        # Remove dominated solutions from frontier
        removed_count = self._remove_dominated_by(solution_tuple)
        
        # Add new solution to frontier
        self.frontier.add(solution_tuple)
        
        # Store metadata
        if metadata:
            self.solution_metadata[solution_tuple] = metadata
        
        # Update hypervolume
        new_hv = self._calculate_hypervolume()
        self.hv_history.append(new_hv)
        
        return {
            'added': True,
            'dominated_by_frontier': False,
            'solutions_removed': removed_count,
            'frontier_size': len(self.frontier),
            'hypervolume': new_hv,
            'hv_delta': new_hv - self.hv_history[-2] if len(self.hv_history) > 1 else new_hv
        }
    
    def get_frontier(self) -> List[Dict[str, float]]:
        """
        Get current Pareto frontier as list of dicts.
        
        Returns:
            List of dicts with objective names as keys
        """
        frontier_dicts = []
        for solution_tuple in self.frontier:
            solution_dict = {
                self.objective_names[i]: (
                    solution_tuple[i] if self.directions[i] == 'maximize'
                    else -solution_tuple[i]
                )
                for i in range(self.num_objectives)
            }
            frontier_dicts.append(solution_dict)
        return frontier_dicts
    
    def calculate_hypervolume(self, reference: Dict[str, float]) -> float:
        """
        Calculate hypervolume with named reference point.
        
        Args:
            reference: Dict mapping objective names to reference values
            
        Returns:
            Hypervolume value
        """
        # Convert reference dict to tuple
        self.reference_point = [
            reference[name] if self.directions[i] == 'maximize'
            else -reference[name]
            for i, name in enumerate(self.objective_names)
        ]
        
        return self._calculate_hypervolume()
    
    def get_report(self) -> Dict:
        """Get comprehensive frontier analysis report."""
        if not self.frontier:
            return {
                'metrics': {
                    'frontier_size': 0,
                    'hypervolume': 0.0
                },
                'recommendation': 'No solutions in frontier yet'
            }
        
        frontier_dicts = self.get_frontier()
        
        # Calculate spacing (uniformity metric)
        spacing = self._calculate_spacing()
        
        # Calculate spread (coverage metric)
        spread = self._calculate_spread()
        
        return {
            'metrics': {
                'frontier_size': len(self.frontier),
                'hypervolume': self.hv_history[-1] if self.hv_history else 0.0,
                'spacing': spacing,
                'spread': spread
            },
            'frontier': frontier_dicts,
            'recommendation': self._generate_recommendation(spacing, spread)
        }
    
    def _calculate_spacing(self) -> float:
        """Calculate spacing metric (lower is better - more uniform)."""
        if len(self.frontier) < 2:
            return 0.0
        
        distances = []
        for i, sol in enumerate(self.frontier):
            if i > 0:
                dist = np.linalg.norm(
                    np.array(sol) - np.array(self.frontier[i-1])
                )
                distances.append(dist)
        
        if not distances:
            return 0.0
        
        mean_dist = np.mean(distances)
        return np.std(distances) / mean_dist if mean_dist > 0 else 0.0
    
    def _calculate_spread(self) -> float:
        """Calculate spread metric (higher is better - more coverage)."""
        if not self.frontier:
            return 0.0
        
        # Calculate range for each objective
        ranges = []
        for i in range(self.num_objectives):
            values = [sol[i] for sol in self.frontier]
            ranges.append(max(values) - min(values))
        
        return np.mean(ranges)
    
    def _generate_recommendation(self, spacing: float, spread: float) -> str:
        """Generate recommendation based on metrics."""
        if spacing < 0.2 and spread > 1.0:
            return "Excellent frontier: Uniform distribution with good coverage"
        elif spacing < 0.2:
            return "Good uniformity but limited coverage - consider expanding search space"
        elif spread > 1.0:
            return "Good coverage but uneven distribution - consider diversity mechanisms"
        else:
            return "Frontier needs improvement - increase both diversity and exploration"
    
    # ... (나머지 메서드는 동일)
```

---

## 5. 통합 테스트 예제

수정된 API를 사용하는 통합 테스트 예제:

```python
#!/usr/bin/env python3
"""
Example usage of corrected Enhanced RSI Test Framework
"""

import numpy as np
from integrated_rsi_test import IntegratedRSITest
from fast_pareto_optimizer import FastParetoOptimizer
from meta_learning_evaluator import AdvancedMetaLearningEvaluator

def main():
    # Initialize framework
    framework = IntegratedRSITest(
        cold_start_steps=20,
        exploration_steps=50,
        convergence_steps=30,
        verification_steps=10
    )
    
    # Initialize Pareto optimizer
    optimizer = FastParetoOptimizer(
        objective_directions={
            'performance': 'maximize',
            'efficiency': 'maximize',
            'complexity': 'minimize'
        }
    )
    
    # Initialize meta-learning evaluator
    meta_evaluator = AdvancedMetaLearningEvaluator(
        min_data_points=15,
        bootstrap_samples=1000
    )
    
    # Simulate RSI iterations
    for iteration in range(100):
        # Simulate model evaluation
        performance = 0.5 + 0.3 * (1 - np.exp(-iteration / 20))
        efficiency = 0.6 + 0.2 * np.random.random()
        complexity = 1000 - iteration * 5
        
        # Add to Pareto frontier
        result = optimizer.add_solution({
            'performance': performance,
            'efficiency': efficiency,
            'complexity': complexity
        })
        
        # Update meta-learning evaluator
        meta_evaluator.update(iteration, performance)
        
        # Get meta-learning report (only after sufficient data)
        meta_report = meta_evaluator.get_report()
        
        # Update integrated framework
        exploration_activity = 0.3 * np.exp(-iteration / 30)
        framework_result = framework.update(
            new_hv=result.get('hypervolume', 0.0),
            exploration_activity=exploration_activity,
            meta_learning_report=meta_report if meta_report.get('is_significant') else None
        )
        
        # Print status
        if iteration % 10 == 0:
            print(f"\nIteration {iteration}")
            print(f"  Phase: {framework_result['phase']}")
            print(f"  Arbiter State: {framework_result['arbiter_state']}")
            print(f"  Frontier Size: {result.get('frontier_size', 0)}")
            print(f"  Hypervolume: {result.get('hypervolume', 0.0):.4f}")
            
            if not framework_result['should_continue']:
                print("\nFramework recommends stopping")
                break
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    summary = framework.get_summary()
    print(f"\nTotal Steps: {summary['total_steps']}")
    print(f"Final Phase: {summary['current_phase']}")
    print(f"Final Arbiter State: {summary['final_arbiter_state']}")
    
    print("\nArbiter State Distribution:")
    for state, percentage in summary['arbiter_summary'].get('state_percentages', {}).items():
        print(f"  {state}: {percentage}")
    
    pareto_report = optimizer.get_report()
    print(f"\nPareto Frontier:")
    print(f"  Size: {pareto_report['metrics']['frontier_size']}")
    print(f"  Hypervolume: {pareto_report['metrics']['hypervolume']:.4f}")
    print(f"  Spacing: {pareto_report['metrics']['spacing']:.4f}")
    print(f"  Recommendation: {pareto_report['recommendation']}")
    
    meta_final = meta_evaluator.get_report()
    print(f"\nMeta-Learning:")
    print(f"  {meta_final['final_interpretation']}")

if __name__ == "__main__":
    main()
```

---

## 적용 방법

1. 각 모듈 파일을 위 제안대로 수정
2. 통합 테스트 실행하여 검증
3. README.md 업데이트
4. 버전을 0.2.0으로 업데이트하고 CHANGELOG 작성

이러한 수정을 통해 모든 모듈이 일관된 API를 제공하고 문서와 일치하게 됩니다.
