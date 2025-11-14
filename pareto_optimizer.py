"""Pareto Optimization for Multi-Objective RSI Testing.

This module provides Pareto frontier tracking and multi-objective optimization
for balancing performance, efficiency, and complexity in RSI systems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class Solution:
    """Represents a single solution in objective space.
    
    Attributes:
        id: Unique identifier for this solution
        objectives: Dictionary mapping objective names to values
        metadata: Optional additional information about the solution
    """
    id: int
    objectives: Dict[str, float]
    metadata: Optional[Dict] = None
    
    def dominates(self, other: 'Solution', objective_directions: Dict[str, str]) -> bool:
        """Check if this solution Pareto-dominates another.
        
        Args:
            other: Another solution to compare against
            objective_directions: Dict mapping objective names to 'minimize' or 'maximize'
        
        Returns:
            True if this solution dominates the other
        """
        at_least_one_better = False
        
        for obj_name, direction in objective_directions.items():
            this_val = self.objectives.get(obj_name, 0)
            other_val = other.objectives.get(obj_name, 0)
            
            if direction == 'maximize':
                if this_val < other_val:
                    return False
                if this_val > other_val:
                    at_least_one_better = True
            else:  # minimize
                if this_val > other_val:
                    return False
                if this_val < other_val:
                    at_least_one_better = True
        
        return at_least_one_better


class ParetoOptimizer:
    """Tracks and optimizes Pareto frontier for multi-objective problems.
    
    This optimizer maintains a Pareto frontier of non-dominated solutions and
    provides metrics for assessing multi-objective optimization progress.
    
    Attributes:
        objective_directions: Dict specifying 'maximize' or 'minimize' for each objective
        pareto_frontier: Current set of non-dominated solutions
        all_solutions: Complete history of evaluated solutions
    """
    
    def __init__(self, objective_directions: Dict[str, str]):
        """Initialize Pareto optimizer.
        
        Args:
            objective_directions: Dict mapping objective names to 'maximize' or 'minimize'
        
        Example:
            optimizer = ParetoOptimizer({
                'performance': 'maximize',
                'efficiency': 'maximize',
                'complexity': 'minimize'
            })
        """
        self.objective_directions = objective_directions
        self.pareto_frontier: List[Solution] = []
        self.all_solutions: List[Solution] = []
        self.next_id = 0
    
    def add_solution(self, objectives: Dict[str, float], metadata: Optional[Dict] = None) -> bool:
        """Add a new solution and update Pareto frontier.
        
        Args:
            objectives: Dictionary of objective values
            metadata: Optional metadata about the solution
        
        Returns:
            True if solution was added to Pareto frontier
        """
        solution = Solution(id=self.next_id, objectives=objectives, metadata=metadata)
        self.next_id += 1
        self.all_solutions.append(solution)
        
        # Check if new solution is dominated by any frontier solution
        dominated = False
        for frontier_sol in self.pareto_frontier:
            if frontier_sol.dominates(solution, self.objective_directions):
                dominated = True
                break
        
        if dominated:
            return False
        
        # Remove any frontier solutions dominated by new solution
        self.pareto_frontier = [
            sol for sol in self.pareto_frontier
            if not solution.dominates(sol, self.objective_directions)
        ]
        
        # Add new solution to frontier
        self.pareto_frontier.append(solution)
        return True
    
    def get_frontier(self) -> List[Solution]:
        """Get current Pareto frontier.
        
        Returns:
            List of non-dominated solutions
        """
        return self.pareto_frontier.copy()
    
    def calculate_hypervolume(self, reference_point: Dict[str, float]) -> float:
        """Calculate hypervolume indicator for Pareto frontier.
        
        The hypervolume measures the volume of objective space dominated by
        the Pareto frontier relative to a reference point.
        
        Args:
            reference_point: Reference point in objective space (typically worst values)
        
        Returns:
            Hypervolume value (higher is better)
        """
        if not self.pareto_frontier:
            return 0.0
        
        # For 2D case, use simple calculation
        obj_names = list(self.objective_directions.keys())
        
        if len(obj_names) == 2:
            return self._hypervolume_2d(reference_point, obj_names)
        else:
            # For higher dimensions, use Monte Carlo approximation
            return self._hypervolume_monte_carlo(reference_point, obj_names)
    
    def _hypervolume_2d(self, reference_point: Dict[str, float], obj_names: List[str]) -> float:
        """Calculate exact hypervolume for 2D case.
        
        Args:
            reference_point: Reference point coordinates
            obj_names: List of objective names
        
        Returns:
            Hypervolume value
        """
        obj1, obj2 = obj_names[0], obj_names[1]
        
        # Normalize directions (convert all to maximization)
        points = []
        for sol in self.pareto_frontier:
            val1 = sol.objectives[obj1]
            val2 = sol.objectives[obj2]
            
            if self.objective_directions[obj1] == 'minimize':
                val1 = -val1
            if self.objective_directions[obj2] == 'minimize':
                val2 = -val2
            
            points.append((val1, val2))
        
        ref1 = -reference_point[obj1] if self.objective_directions[obj1] == 'minimize' else reference_point[obj1]
        ref2 = -reference_point[obj2] if self.objective_directions[obj2] == 'minimize' else reference_point[obj2]
        
        # Sort by first objective
        points.sort(reverse=True)
        
        # Calculate area
        area = 0.0
        prev_x = ref1
        for x, y in points:
            if x <= ref1 or y <= ref2:
                continue
            width = x - prev_x
            height = y - ref2
            area += width * height
            prev_x = x
        
        return abs(area)
    
    def _hypervolume_monte_carlo(
        self,
        reference_point: Dict[str, float],
        obj_names: List[str],
        n_samples: int = 10000
    ) -> float:
        """Approximate hypervolume using Monte Carlo sampling.
        
        Args:
            reference_point: Reference point coordinates
            obj_names: List of objective names
            n_samples: Number of Monte Carlo samples
        
        Returns:
            Approximate hypervolume value
        """
        if not self.pareto_frontier:
            return 0.0
        
        # Determine bounds for sampling
        bounds = {}
        for obj_name in obj_names:
            frontier_vals = [sol.objectives[obj_name] for sol in self.pareto_frontier]
            ref_val = reference_point[obj_name]
            
            if self.objective_directions[obj_name] == 'maximize':
                bounds[obj_name] = (ref_val, max(frontier_vals))
            else:
                bounds[obj_name] = (min(frontier_vals), ref_val)
        
        # Sample points and check domination
        dominated_count = 0
        for _ in range(n_samples):
            sample_point = {
                obj_name: np.random.uniform(bounds[obj_name][0], bounds[obj_name][1])
                for obj_name in obj_names
            }
            
            # Check if sample is dominated by any frontier solution
            for sol in self.pareto_frontier:
                dominates_sample = True
                for obj_name in obj_names:
                    sol_val = sol.objectives[obj_name]
                    sample_val = sample_point[obj_name]
                    
                    if self.objective_directions[obj_name] == 'maximize':
                        if sol_val < sample_val:
                            dominates_sample = False
                            break
                    else:
                        if sol_val > sample_val:
                            dominates_sample = False
                            break
                
                if dominates_sample:
                    dominated_count += 1
                    break
        
        # Calculate volume
        total_volume = 1.0
        for obj_name in obj_names:
            total_volume *= (bounds[obj_name][1] - bounds[obj_name][0])
        
        return (dominated_count / n_samples) * total_volume
    
    def get_frontier_metrics(self) -> Dict:
        """Calculate metrics describing the Pareto frontier.
        
        Returns:
            Dictionary containing frontier metrics
        """
        if not self.pareto_frontier:
            return {
                "frontier_size": 0,
                "total_solutions": len(self.all_solutions),
                "frontier_ratio": 0.0
            }
        
        # Calculate spacing (diversity metric)
        spacing = self._calculate_spacing()
        
        # Calculate spread (extent metric)
        spread = self._calculate_spread()
        
        return {
            "frontier_size": len(self.pareto_frontier),
            "total_solutions": len(self.all_solutions),
            "frontier_ratio": len(self.pareto_frontier) / len(self.all_solutions),
            "spacing": spacing,
            "spread": spread,
            "objectives": list(self.objective_directions.keys())
        }
    
    def _calculate_spacing(self) -> float:
        """Calculate spacing metric (uniformity of distribution).
        
        Returns:
            Spacing value (lower is more uniform)
        """
        if len(self.pareto_frontier) < 2:
            return 0.0
        
        # Calculate minimum distance to nearest neighbor for each point
        distances = []
        for i, sol1 in enumerate(self.pareto_frontier):
            min_dist = float('inf')
            for j, sol2 in enumerate(self.pareto_frontier):
                if i == j:
                    continue
                
                # Euclidean distance in normalized objective space
                dist = 0.0
                for obj_name in self.objective_directions.keys():
                    val1 = sol1.objectives[obj_name]
                    val2 = sol2.objectives[obj_name]
                    dist += (val1 - val2) ** 2
                dist = np.sqrt(dist)
                
                min_dist = min(min_dist, dist)
            
            distances.append(min_dist)
        
        # Calculate standard deviation of distances
        mean_dist = np.mean(distances)
        spacing = np.std(distances) if mean_dist > 0 else 0.0
        
        return spacing
    
    def _calculate_spread(self) -> float:
        """Calculate spread metric (extent of frontier).
        
        Returns:
            Spread value (higher means larger coverage)
        """
        if len(self.pareto_frontier) < 2:
            return 0.0
        
        # Calculate range for each objective
        total_spread = 0.0
        for obj_name in self.objective_directions.keys():
            values = [sol.objectives[obj_name] for sol in self.pareto_frontier]
            obj_range = max(values) - min(values)
            total_spread += obj_range
        
        return total_spread / len(self.objective_directions)
    
    def get_report(self) -> Dict:
        """Generate comprehensive Pareto optimization report.
        
        Returns:
            Dictionary containing optimization metrics and frontier info
        """
        metrics = self.get_frontier_metrics()
        
        # Add frontier solutions details
        frontier_details = []
        for sol in self.pareto_frontier:
            frontier_details.append({
                "id": sol.id,
                "objectives": sol.objectives,
                "metadata": sol.metadata
            })
        
        report = {
            "metrics": metrics,
            "frontier": frontier_details,
            "objective_directions": self.objective_directions,
            "recommendation": self._get_recommendation(metrics)
        }
        
        return report
    
    def _get_recommendation(self, metrics: Dict) -> str:
        """Generate recommendation based on frontier metrics.
        
        Args:
            metrics: Frontier metrics dictionary
        
        Returns:
            Human-readable recommendation
        """
        frontier_ratio = metrics["frontier_ratio"]
        frontier_size = metrics["frontier_size"]
        
        if frontier_size == 0:
            return "No solutions evaluated yet"
        
        if frontier_ratio > 0.5:
            return "High frontier ratio - consider more exploitation"
        elif frontier_size < 5:
            return "Small frontier - continue exploration"
        else:
            return "Healthy frontier diversity - continue optimization"
