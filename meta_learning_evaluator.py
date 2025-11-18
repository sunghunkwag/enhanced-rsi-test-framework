from __future__ import annotations

"""Advanced Meta-Learning Evaluator for RSI Testing

This module provides statistical validation of learning acceleration in RSI systems
through EMA smoothing, linear regression analysis, and block bootstrap resampling.
"""

import random
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


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
                 alphas_to_try: List[float] = None):
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
                'required_points': self.min_data_points,
                'is_significant': False
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
        best_model_type = trend_analysis['best_model']
        best_model_params = trend_analysis[best_model_type]['params']
        
        # 3. Block Bootstrap
        n_speeds = len(smoothed_speeds)
        block_length = max(1, int(n_speeds ** (1/3)))
        robustness_report = self._get_robustness_analysis(
            smoothed_speeds,
            best_model_type,
            block_length,
            n_bootstraps=self.bootstrap_samples,
            confidence_level=self.confidence_level
        )
        
        # Generate interpretation
        is_significant = robustness_report.get('is_robust', False)
        
        if is_significant:
            if best_model_type == 'quadratic':
                interpretation = "Strong meta-learning: Accelerating improvement detected (quadratic trend)"
            else:
                interpretation = "Meta-learning detected: Linear improvement trend confirmed"
        else:
            if best_model_params[0] > 0:
                interpretation = "Weak meta-learning: Positive trend but not statistically robust"
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
            'trend_model_comparison': {
                'best_model': best_model_type,
                'linear_aic': trend_analysis['linear']['aic'],
                'quadratic_aic': trend_analysis['quadratic']['aic'],
                'best_params': best_model_params.tolist()
            },
            'robustness_analysis': robustness_report,
            'data_points': len(self.performance_history)
        }
        
        return self._analysis_cache
    
    def _find_optimal_alpha(self, history: np.ndarray,
                           alphas: List[float]) -> Tuple[float, np.ndarray]:
        """Find alpha that minimizes one-step-ahead prediction error."""
        best_alpha = 1.0
        min_sse = float('inf')
        best_ema_history = history
        
        for alpha in alphas:
            ema_history = np.zeros_like(history)
            ema_history[0] = history[0]
            
            for i in range(1, len(history)):
                ema_history[i] = (alpha * history[i]) + \
                                ((1 - alpha) * ema_history[i-1])
            
            # One-step-ahead prediction error
            sse = np.sum((history[1:] - ema_history[:-1])**2)
            
            if sse < min_sse:
                min_sse = sse
                best_alpha = alpha
                best_ema_history = ema_history
        
        return best_alpha, best_ema_history
    
    def _calculate_aic(self, n: int, rss: float, k: int) -> float:
        """Calculate Akaike Information Criterion."""
        if rss <= 0 or n <= 0:
            return float('inf')
        return n * np.log(rss / n) + 2 * (k + 1)
    
    def _compare_trend_models(self, y_data: np.ndarray) -> Dict:
        """Compare linear vs quadratic trend models using AIC."""
        x = np.arange(len(y_data))
        n = len(y_data)
        results = {}
        
        # Model 1: Linear (y = mx + c)
        params_lin = np.polyfit(x, y_data, 1)
        y_fit_lin = np.polyval(params_lin, x)
        rss_lin = np.sum((y_data - y_fit_lin)**2)
        aic_lin = self._calculate_aic(n, rss_lin, k=2)
        results['linear'] = {
            'params': params_lin,
            'aic': aic_lin,
            'rss': rss_lin
        }
        
        # Model 2: Quadratic (y = ax^2 + bx + c)
        # The 'a' coefficient represents meta-learning acceleration
        params_quad = np.polyfit(x, y_data, 2)
        y_fit_quad = np.polyval(params_quad, x)
        rss_quad = np.sum((y_data - y_fit_quad)**2)
        aic_quad = self._calculate_aic(n, rss_quad, k=3)
        results['quadratic'] = {
            'params': params_quad,
            'aic': aic_quad,
            'rss': rss_quad
        }
        
        # Select best model (lower AIC)
        results['best_model'] = 'quadratic' if aic_quad < aic_lin else 'linear'
        
        return results
    
    def _perform_block_bootstrap(self, data_pairs: List,
                                 block_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Block bootstrap preserving time-series structure."""
        n = len(data_pairs)
        n_blocks = int(np.ceil(n / block_length))
        
        # Divide into blocks
        blocks = [data_pairs[i:i+block_length]
                 for i in range(0, n, block_length)]
        
        # Resample blocks with replacement
        resampled_blocks = random.choices(blocks, k=n_blocks)
        
        # Flatten and trim to original length
        resampled_data = [item for block in resampled_blocks for item in block][:n]
        resampled_x, resampled_y = zip(*resampled_data)
        
        return np.array(resampled_x), np.array(resampled_y)
    
    def _get_robustness_analysis(self, y_data: np.ndarray,
                                best_model_type: str,
                                block_length: int,
                                n_bootstraps: int,
                                confidence_level: float) -> Dict:
        """Bootstrap confidence interval for key parameter."""
        x_data = np.arange(len(y_data))
        data_pairs = list(zip(x_data, y_data))
        bootstrapped_key_params = []
        
        for _ in range(n_bootstraps):
            resampled_x, resampled_y = self._perform_block_bootstrap(
                data_pairs, block_length
            )
            
            if len(np.unique(resampled_x)) < 3:
                continue
            
            if best_model_type == 'quadratic':
                params = np.polyfit(resampled_x, resampled_y, 2)
                bootstrapped_key_params.append(params[0])  # 'a' coefficient
            else:
                params = np.polyfit(resampled_x, resampled_y, 1)
                bootstrapped_key_params.append(params[0])  # slope 'm'
        
        if not bootstrapped_key_params:
            return {"error": "Bootstrapping failed", "is_robust": False}
        
        # Calculate confidence interval
        alpha_level = (1.0 - confidence_level) / 2.0
        ci_lower = np.percentile(bootstrapped_key_params, alpha_level * 100)
        ci_upper = np.percentile(bootstrapped_key_params, (1.0 - alpha_level) * 100)
        
        is_robust = (ci_lower > 0)  # CI lower bound above zero
        
        return {
            "parameter_of_interest": (
                "Quadratic 'a' coefficient" if best_model_type == 'quadratic'
                else "Linear slope 'm'"
            ),
            "n_bootstraps": n_bootstraps,
            "block_length": block_length,
            "confidence_interval": (ci_lower, ci_upper),
            "is_robust (CI_lower > 0)": is_robust,
            "is_robust": is_robust
        }
