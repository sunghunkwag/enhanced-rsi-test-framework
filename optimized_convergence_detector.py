from collections import deque
from typing import Dict, Tuple
import numpy as np


class OptimizedEnhancedConvergenceDetector:
    """Memory-efficient O(1) convergence detector using deque and running sums."""
    
    def __init__(self, window_size: int = 50, volatility_threshold: float = 0.05,
                 exploration_threshold: float = 0.1):
        """
        Args:
            window_size: Size of sliding window for convergence detection
            volatility_threshold: Max acceptable volatility for convergence
            exploration_threshold: Min exploration activity to be considered exploring
        """
        self.window_size = window_size
        self.volatility_threshold = volatility_threshold
        self.exploration_threshold = exploration_threshold
        
        # Deques for O(1) push/pop operations
        self.hv_history = deque(maxlen=window_size)
        self.squared_hv_history = deque(maxlen=window_size)
        
        # Running sums for O(1) mean/variance calculation
        self.hv_sum = 0.0
        self.hv_squared_sum = 0.0
        
        # Exploration tracking
        self.exploration_volatility = 0.0
        self.steps_since_last_exploration = 0
        
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
        if exploration_activity > self.exploration_threshold:
            self.exploration_volatility = exploration_activity
            self.steps_since_last_exploration = 0
        else:
            self.steps_since_last_exploration += 1
        
        # Calculate convergence status
        converged, volatility = self._check_convergence()
        
        return {
            'converged': converged,
            'volatility': volatility,
            'exploring': exploration_activity > self.exploration_threshold,
            'steps_since_exploration': self.steps_since_last_exploration,
            'current_hv': new_hv,
            'mean_hv': self.hv_sum / len(self.hv_history) if self.hv_history else 0.0
        }
    
    def _check_convergence(self) -> Tuple[bool, float]:
        """
        Check if convergence achieved using O(1) variance calculation.
        
        Returns:
            (converged, volatility) tuple
        """
        n = len(self.hv_history)
        
        if n < self.window_size:
            return False, float('inf')
        
        # O(1) mean calculation
        mean = self.hv_sum / n
        
        # O(1) variance calculation
        # Var(X) = E[X^2] - (E[X])^2
        mean_of_squares = self.hv_squared_sum / n
        variance = mean_of_squares - (mean ** 2)
        
        # Prevent negative variance from floating point errors
        variance = max(0.0, variance)
        
        # Volatility as coefficient of variation
        volatility = np.sqrt(variance) / mean if mean > 0 else float('inf')
        
        converged = volatility < self.volatility_threshold
        
        return converged, volatility
    
    def reset(self):
        """Reset detector state."""
        self.hv_history.clear()
        self.squared_hv_history.clear()
        self.hv_sum = 0.0
        self.hv_squared_sum = 0.0
        self.exploration_volatility = 0.0
        self.steps_since_last_exploration = 0
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics about detector state."""
        n = len(self.hv_history)
        
        if n == 0:
            return {
                'window_fill': 0.0,
                'mean_hv': 0.0,
                'volatility': float('inf'),
                'exploration_volatility': self.exploration_volatility
            }
        
        mean = self.hv_sum / n
        mean_of_squares = self.hv_squared_sum / n
        variance = max(0.0, mean_of_squares - (mean ** 2))
        volatility = np.sqrt(variance) / mean if mean > 0 else float('inf')
        
        return {
            'window_fill': n / self.window_size,
            'mean_hv': mean,
            'volatility': volatility,
            'exploration_volatility': self.exploration_volatility,
            'steps_since_exploration': self.steps_since_last_exploration
        }
