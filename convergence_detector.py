"""Enhanced Convergence Detection for RSI Testing.

This module provides sophisticated convergence detection that distinguishes between
true convergence and productive plateaus in recursive self-improvement testing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum


class ConvergenceState(Enum):
    """Enumeration of possible convergence states."""
    IMPROVING = "improving"
    PRODUCTIVE_PLATEAU = "productive_plateau"
    TRUE_CONVERGENCE = "true_convergence"
    DIVERGING = "diverging"


class EnhancedConvergenceDetector:
    """Detects convergence with distinction between plateaus and true convergence.
    
    This detector tracks both performance metrics and structural changes to identify
    whether an apparent plateau represents continued exploration (productive) or
    actual convergence (unproductive).
    
    Attributes:
        window_size: Number of recent iterations to analyze
        convergence_threshold: Maximum allowed change rate for convergence
        structural_threshold: Minimum structural change for productive plateau
        probation_period: Grace period after apparent convergence
    """
    
    def __init__(
        self,
        window_size: int = 10,
        convergence_threshold: float = 0.01,
        structural_threshold: float = 0.05,
        probation_period: int = 5
    ):
        """Initialize convergence detector.
        
        Args:
            window_size: Number of iterations to analyze for trends
            convergence_threshold: Max percent change for convergence
            structural_threshold: Min structural similarity change for plateau
            probation_period: Iterations to wait before confirming convergence
        """
        self.window_size = window_size
        self.convergence_threshold = convergence_threshold
        self.structural_threshold = structural_threshold
        self.probation_period = probation_period
        
        # Historical data
        self.performance_history: List[float] = []
        self.architecture_vectors: List[np.ndarray] = []
        self.probation_counter = 0
        self.last_state = ConvergenceState.IMPROVING
    
    def update(
        self,
        performance: float,
        architecture_vector: Optional[np.ndarray] = None
    ) -> ConvergenceState:
        """Update detector with new iteration data.
        
        Args:
            performance: Current performance metric (higher is better)
            architecture_vector: Optional vector representing model architecture
        
        Returns:
            Current convergence state
        """
        self.performance_history.append(performance)
        
        if architecture_vector is not None:
            self.architecture_vectors.append(architecture_vector)
        
        # Need minimum history for analysis
        if len(self.performance_history) < self.window_size:
            return ConvergenceState.IMPROVING
        
        # Analyze recent performance trend
        recent_performance = self.performance_history[-self.window_size:]
        performance_trend = self._calculate_trend(recent_performance)
        performance_volatility = self._calculate_volatility(recent_performance)
        
        # Analyze structural changes if available
        structural_change = 0.0
        if len(self.architecture_vectors) >= 2:
            structural_change = self._calculate_structural_change()
        
        # Determine state
        state = self._determine_state(
            performance_trend,
            performance_volatility,
            structural_change
        )
        
        # Handle probation period for convergence
        if state == ConvergenceState.TRUE_CONVERGENCE:
            if self.last_state != ConvergenceState.TRUE_CONVERGENCE:
                self.probation_counter = 0
            self.probation_counter += 1
            
            if self.probation_counter < self.probation_period:
                state = ConvergenceState.PRODUCTIVE_PLATEAU
        else:
            self.probation_counter = 0
        
        self.last_state = state
        return state
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values.
        
        Args:
            values: List of performance values
        
        Returns:
            Slope of linear regression (normalized by mean)
        """
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression
        coefficients = np.polyfit(x, y, 1)
        slope = coefficients[0]
        
        # Normalize by mean to get percentage change
        mean_value = np.mean(y)
        if mean_value != 0:
            normalized_slope = slope / mean_value
        else:
            normalized_slope = 0.0
        
        return normalized_slope
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (coefficient of variation).
        
        Args:
            values: List of performance values
        
        Returns:
            Coefficient of variation (std / mean)
        """
        if len(values) < 2:
            return 0.0
        
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)
        
        if mean != 0:
            return std / mean
        else:
            return 0.0
    
    def _calculate_structural_change(self) -> float:
        """Calculate structural similarity between recent architectures.
        
        Returns:
            Average cosine distance between recent architecture pairs
        """
        if len(self.architecture_vectors) < 2:
            return 0.0
        
        # Compare last few architecture vectors
        n_compare = min(5, len(self.architecture_vectors))
        recent_vectors = self.architecture_vectors[-n_compare:]
        
        # Calculate pairwise cosine distances
        distances = []
        for i in range(len(recent_vectors) - 1):
            vec1 = recent_vectors[i]
            vec2 = recent_vectors[i + 1]
            
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
                distance = 1 - similarity
                distances.append(abs(distance))
        
        return np.mean(distances) if distances else 0.0
    
    def _determine_state(
        self,
        trend: float,
        volatility: float,
        structural_change: float
    ) -> ConvergenceState:
        """Determine convergence state from metrics.
        
        Args:
            trend: Performance trend (normalized slope)
            volatility: Performance volatility (coefficient of variation)
            structural_change: Architecture change metric
        
        Returns:
            Determined convergence state
        """
        # Improving: positive trend or high volatility (exploration)
        if trend > self.convergence_threshold or volatility > 0.1:
            return ConvergenceState.IMPROVING
        
        # Diverging: negative trend beyond threshold
        if trend < -self.convergence_threshold * 2:
            return ConvergenceState.DIVERGING
        
        # Check for productive plateau vs true convergence
        if abs(trend) <= self.convergence_threshold:
            # High structural change indicates continued exploration
            if structural_change > self.structural_threshold:
                return ConvergenceState.PRODUCTIVE_PLATEAU
            else:
                return ConvergenceState.TRUE_CONVERGENCE
        
        return ConvergenceState.IMPROVING
    
    def get_report(self) -> Dict:
        """Generate comprehensive convergence analysis report.
        
        Returns:
            Dictionary containing convergence metrics and state
        """
        if len(self.performance_history) < self.window_size:
            return {
                "state": ConvergenceState.IMPROVING.value,
                "sufficient_data": False,
                "iterations": len(self.performance_history)
            }
        
        recent = self.performance_history[-self.window_size:]
        trend = self._calculate_trend(recent)
        volatility = self._calculate_volatility(recent)
        structural = self._calculate_structural_change()
        
        return {
            "state": self.last_state.value,
            "sufficient_data": True,
            "iterations": len(self.performance_history),
            "performance_trend": trend,
            "performance_volatility": volatility,
            "structural_change": structural,
            "probation_counter": self.probation_counter,
            "recent_mean": np.mean(recent),
            "recent_std": np.std(recent),
            "recommendation": self._get_recommendation()
        }
    
    def _get_recommendation(self) -> str:
        """Get action recommendation based on current state.
        
        Returns:
            Human-readable recommendation string
        """
        if self.last_state == ConvergenceState.IMPROVING:
            return "Continue iteration - system is improving"
        elif self.last_state == ConvergenceState.PRODUCTIVE_PLATEAU:
            return "Continue iteration - plateau appears productive"
        elif self.last_state == ConvergenceState.TRUE_CONVERGENCE:
            return "Consider stopping - true convergence detected"
        else:  # DIVERGING
            return "Review system - performance degrading"
