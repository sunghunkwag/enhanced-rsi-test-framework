"""Advanced Meta-Learning Evaluator for RSI Testing

This module provides statistical validation of learning acceleration in RSI systems
through EMA smoothing, linear regression analysis, and block bootstrap resampling.
"""

import random
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


class AdvancedMetaLearningEvaluator:
    """
    RSI meta-learning acceleration validator with statistical robustness.
    
    Features:
    1. Adaptive Alpha Selection: Automatically optimizes EMA smoothing parameter
    2. Nonlinear Trend Modeling: Compares linear vs quadratic trend models
    3. Block Bootstrap: Preserves time-series autocorrelation in confidence intervals
    """
    
    def __init__(self, performance_history: list[float],
                 alphas_to_try: list[float] = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]):
        """
        Args:
            performance_history: Raw performance scores (minimum 5 recommended)
            alphas_to_try: Candidate alpha values for optimal EMA selection
        """
        if len(performance_history) < 5:
            raise ValueError(
                f"Need at least 5 performance points, got {len(performance_history)}"
            )
        
        self.original_history = np.array(performance_history)
        
        # 1. Adaptive Alpha Selection
        self.optimal_alpha, self.smoothed_history = self._find_optimal_alpha(
            self.original_history, alphas_to_try
        )
        
        # Calculate smoothed speeds (1st derivative)
        self.smoothed_speeds = self.smoothed_history[1:] - self.smoothed_history[:-1]
        
        # 2. Nonlinear Trend Model Comparison
        self.trend_analysis = self._compare_trend_models(self.smoothed_speeds)
        self.best_model_type = self.trend_analysis['best_model']
        self.best_model_params = self.trend_analysis[self.best_model_type]['params']
        
        # 3. Block Bootstrap
        n_speeds = len(self.smoothed_speeds)
        self.block_length = max(1, int(n_speeds ** (1/3)))
        self.robustness_report = self._get_robustness_analysis(
            self.smoothed_speeds,
            n_bootstraps=1000,
            confidence_level=0.95
        )
    
    def _find_optimal_alpha(self, history: np.ndarray,
                           alphas: list[float]) -> tuple[float, np.ndarray]:
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
    
    def _compare_trend_models(self, y_data: np.ndarray) -> dict:
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
    
    def _perform_block_bootstrap(self, data_pairs: list,
                                 block_length: int) -> tuple[np.ndarray, np.ndarray]:
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
                                n_bootstraps: int,
                                confidence_level: float) -> dict:
        """Bootstrap confidence interval for key parameter."""
        x_data = np.arange(len(y_data))
        data_pairs = list(zip(x_data, y_data))
        bootstrapped_key_params = []
        
        for _ in range(n_bootstraps):
            resampled_x, resampled_y = self._perform_block_bootstrap(
                data_pairs, self.block_length
            )
            
            if len(np.unique(resampled_x)) < 3:
                continue
            
            if self.best_model_type == 'quadratic':
                params = np.polyfit(resampled_x, resampled_y, 2)
                bootstrapped_key_params.append(params[0])  # 'a' coefficient
            else:
                params = np.polyfit(resampled_x, resampled_y, 1)
                bootstrapped_key_params.append(params[0])  # slope 'm'
        
        if not bootstrapped_key_params:
            return {"error": "Bootstrapping failed"}
        
        # Calculate confidence interval
        alpha_level = (1.0 - confidence_level) / 2.0
        ci_lower = np.percentile(bootstrapped_key_params, alpha_level * 100)
        ci_upper = np.percentile(bootstrapped_key_params, (1.0 - alpha_level) * 100)
        
        is_robust = (ci_lower > 0)  # CI lower bound above zero
        
        return {
            "parameter_of_interest": (
                "Quadratic 'a' coefficient" if self.best_model_type == 'quadratic'
                else "Linear slope 'm'"
            ),
            "n_bootstraps": n_bootstraps,
            "block_length": self.block_length,
            "confidence_interval": (ci_lower, ci_upper),
            "is_robust": is_robust
        }
    
    def get_report(self) -> dict:
        """Generate comprehensive analysis report."""
        stats_result = self.robustness_report
        
        if stats_result.get("is_robust"):
            interpretation = "Meta-learning effect positive (statistically robust acceleration)"
        elif self.best_model_params[0] > 0:
            interpretation = "Meta-learning effect weak (trend positive but not robust)"
        else:
            interpretation = "Meta-learning effect neutral or negative (no acceleration)"
        
        return {
            "final_interpretation": interpretation,
            "adaptive_smoothing": {
                "optimal_alpha": self.optimal_alpha,
                "smoothed_history": self.smoothed_history.round(4).tolist(),
                "smoothed_speeds": self.smoothed_speeds.round(4).tolist()
            },
            "trend_model_comparison": {
                "best_model": self.best_model_type,
                "linear_aic": self.trend_analysis['linear']['aic'],
                "quadratic_aic": self.trend_analysis['quadratic']['aic'],
                "best_params": self.best_model_params.tolist()
            },
            "robustness_analysis": stats_result
        }
