from sortedcontainers import SortedList
from typing import List, Tuple, Optional
import numpy as np


class FastParetoOptimizer:
    """Efficient Pareto frontier optimizer using sorted containers."""
    
    def __init__(self, num_objectives: int = 2, reference_point: Optional[List[float]] = None):
        """
        Args:
            num_objectives: Number of objectives to optimize
            reference_point: Reference point for hypervolume calculation
        """
        self.num_objectives = num_objectives
        self.reference_point = reference_point or [0.0] * num_objectives
        
        # Use SortedList for O(log n) insertion and efficient range queries
        # Sort by first objective for efficient dominance checking
        self.frontier = SortedList(key=lambda x: x[0])
        
        # Track hypervolume history
        self.hv_history = []
        
    def add_solution(self, solution: Tuple[float, ...]) -> dict:
        """
        Add solution to frontier and update Pareto set.
        
        Args:
            solution: Tuple of objective values
            
        Returns:
            Dict with update information
        """
        if len(solution) != self.num_objectives:
            raise ValueError(f"Expected {self.num_objectives} objectives, got {len(solution)}")
        
        # Check if solution is dominated by existing frontier
        if self._is_dominated(solution):
            return {
                'added': False,
                'dominated_by_frontier': True,
                'solutions_removed': 0,
                'frontier_size': len(self.frontier)
            }
        
        # Remove dominated solutions from frontier
        removed_count = self._remove_dominated_by(solution)
        
        # Add new solution to frontier
        self.frontier.add(solution)
        
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
    
    def _is_dominated(self, solution: Tuple[float, ...]) -> bool:
        """
        Check if solution is dominated by any point in frontier.
        Uses sorted structure for efficient pruning.
        
        Args:
            solution: Solution to check
            
        Returns:
            True if dominated, False otherwise
        """
        # Binary search to find candidates
        # Only need to check solutions with first objective >= solution[0]
        idx = self.frontier.bisect_left((solution[0],))
        
        for i in range(idx, len(self.frontier)):
            frontier_sol = self.frontier[i]
            
            # Check if frontier_sol dominates solution
            if self._dominates(frontier_sol, solution):
                return True
        
        return False
    
    def _remove_dominated_by(self, solution: Tuple[float, ...]) -> int:
        """
        Remove all solutions dominated by the new solution.
        
        Args:
            solution: Dominating solution
            
        Returns:
            Number of solutions removed
        """
        to_remove = []
        
        # Only check solutions with first objective <= solution[0]
        idx = self.frontier.bisect_right((solution[0],))
        
        for i in range(idx):
            frontier_sol = self.frontier[i]
            if self._dominates(solution, frontier_sol):
                to_remove.append(frontier_sol)
        
        # Remove dominated solutions
        for sol in to_remove:
            self.frontier.remove(sol)
        
        return len(to_remove)
    
    def _dominates(self, sol_a: Tuple[float, ...], sol_b: Tuple[float, ...]) -> bool:
        """
        Check if sol_a dominates sol_b (assuming maximization).
        
        Args:
            sol_a: First solution
            sol_b: Second solution
            
        Returns:
            True if sol_a dominates sol_b
        """
        # sol_a dominates sol_b if:
        # - sol_a is >= sol_b in all objectives
        # - sol_a is strictly > sol_b in at least one objective
        
        all_geq = all(a >= b for a, b in zip(sol_a, sol_b))
        any_gt = any(a > b for a, b in zip(sol_a, sol_b))
        
        return all_geq and any_gt
    
    def _calculate_hypervolume(self) -> float:
        """
        Calculate hypervolume of current Pareto frontier.
        Uses 2D algorithm for efficiency.
        
        Returns:
            Hypervolume value
        """
        if not self.frontier:
            return 0.0
        
        if self.num_objectives == 2:
            return self._hypervolume_2d()
        else:
            # For higher dimensions, use Monte Carlo approximation
            return self._hypervolume_monte_carlo()
    
    def _hypervolume_2d(self) -> float:
        """
        Efficient O(n log n) hypervolume calculation for 2D.
        Frontier is already sorted by first objective.
        
        Returns:
            2D hypervolume
        """
        hv = 0.0
        prev_y = self.reference_point[1]
        
        # Iterate through sorted frontier
        for sol in self.frontier:
            x, y = sol[0], sol[1]
            
            # Skip if solution is dominated by reference point
            if x <= self.reference_point[0] or y <= self.reference_point[1]:
                continue
            
            # Add rectangular area
            width = x - self.reference_point[0]
            height = y - prev_y
            hv += width * height
            
            prev_y = y
        
        return hv
    
    def _hypervolume_monte_carlo(self, samples: int = 10000) -> float:
        """
        Monte Carlo approximation for high-dimensional hypervolume.
        
        Args:
            samples: Number of Monte Carlo samples
            
        Returns:
            Approximated hypervolume
        """
        if not self.frontier:
            return 0.0
        
        # Find bounds
        max_vals = [max(sol[i] for sol in self.frontier) for i in range(self.num_objectives)]
        
        # Sample random points in bounding box
        dominated_count = 0
        
        for _ in range(samples):
            # Generate random point
            point = tuple(
                np.random.uniform(self.reference_point[i], max_vals[i])
                for i in range(self.num_objectives)
            )
            
            # Check if point is dominated by any frontier solution
            for sol in self.frontier:
                if all(s >= p for s, p in zip(sol, point)):
                    dominated_count += 1
                    break
        
        # Calculate bounding box volume
        box_volume = np.prod([max_vals[i] - self.reference_point[i] 
                              for i in range(self.num_objectives)])
        
        # Hypervolume approximation
        return (dominated_count / samples) * box_volume
    
    def get_frontier(self) -> List[Tuple[float, ...]]:
        """Get current Pareto frontier."""
        return list(self.frontier)
    
    def get_hypervolume_history(self) -> List[float]:
        """Get hypervolume history."""
        return self.hv_history.copy()
    
    def get_current_hypervolume(self) -> float:
        """Get current hypervolume."""
        return self.hv_history[-1] if self.hv_history else 0.0
    
    def reset(self):
        """Reset optimizer state."""
        self.frontier.clear()
        self.hv_history.clear()
