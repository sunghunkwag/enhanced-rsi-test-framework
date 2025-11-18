from __future__ import annotations
from sortedcontainers import SortedList
from typing import List, Tuple, Optional, Dict
import numpy as np


class FastParetoOptimizer:
    """Efficient Pareto frontier optimizer using sorted containers."""
    
    def __init__(self, objective_directions: Dict[str, str],
                 reference_point: Optional[Dict[str, float]] = None):
        """
        Args:
            objective_directions: Dict mapping objective names to 'maximize' or 'minimize'
            reference_point: Optional dict with reference point for hypervolume calculation
        """
        if not objective_directions:
            raise ValueError("objective_directions must be provided")

        self.objective_names = list(objective_directions.keys())
        self.num_objectives = len(self.objective_names)
        self.directions = [objective_directions[name] for name in self.objective_names]
        
        # Convert dict reference point to internal tuple representation
        if reference_point:
            self.reference_point = self._convert_dict_to_tuple(reference_point)
        else:
            # Use a default that assumes minimization (large numbers) or maximization (small numbers)
            self.reference_point = tuple(
                -np.inf if d == 'maximize' else np.inf for d in self.directions
            )

        self.frontier = SortedList(key=lambda x: x[0])
        self.hv_history = []
        self.solution_metadata = {}

    def _convert_dict_to_tuple(self, objectives: Dict[str, float]) -> Tuple[float, ...]:
        """Convert objective dict to internal tuple representation."""
        return tuple(
            objectives[name] if self.directions[i] == 'maximize'
            else -objectives[name]
            for i, name in enumerate(self.objective_names)
        )

    def add_solution(self, objectives: Dict[str, float],
                    metadata: Optional[Dict] = None) -> dict:
        """Add solution to frontier and update Pareto set."""
        solution = self._convert_dict_to_tuple(objectives)
        
        if self._is_dominated(solution):
            return {'added': False, 'dominated_by_frontier': True, 'solutions_removed': 0}
        
        removed_count = self._remove_dominated_by(solution)
        self.frontier.add(solution)
        if metadata:
            self.solution_metadata[solution] = metadata
        
        new_hv = self._calculate_hypervolume()
        self.hv_history.append(new_hv)
        
        return {
            'added': True,
            'dominated_by_frontier': False,
            'solutions_removed': removed_count,
            'hypervolume': new_hv
        }
    
    def _is_dominated(self, solution: Tuple[float, ...]) -> bool:
        """Check if solution is dominated by any point in frontier."""
        for frontier_sol in self.frontier:
            if self._dominates(frontier_sol, solution):
                return True
        return False
    
    def _remove_dominated_by(self, solution: Tuple[float, ...]) -> int:
        """Remove all solutions dominated by the new solution."""
        to_remove = [s for s in self.frontier if self._dominates(solution, s)]
        for sol in to_remove:
            self.frontier.remove(sol)
            if sol in self.solution_metadata:
                del self.solution_metadata[sol]
        return len(to_remove)
    
    def _dominates(self, sol_a: Tuple[float, ...], sol_b: Tuple[float, ...]) -> bool:
        """Check if sol_a dominates sol_b."""
        all_geq = all(a >= b for a, b in zip(sol_a, sol_b))
        any_gt = any(a > b for a, b in zip(sol_a, sol_b))
        return all_geq and any_gt
    
    def _calculate_hypervolume(self) -> float:
        """Calculate hypervolume of current Pareto frontier."""
        if not self.frontier:
            return 0.0
        
        # This implementation is simplified and assumes 2D for now
        if self.num_objectives != 2:
            return 0.0 # Placeholder for higher dimensions

        return self._hypervolume_2d()
    
    def _hypervolume_2d(self) -> float:
        """Efficient O(n log n) hypervolume calculation for 2D."""
        hv = 0.0
        sorted_frontier = sorted(self.frontier, key=lambda x: x[0], reverse=True)
        
        ref_x, ref_y = self.reference_point

        last_y = ref_y
        for x, y in sorted_frontier:
            if x < ref_x or y < ref_y:
                continue
            
            width = x - ref_x
            height = y - last_y
            if width > 0 and height > 0:
                hv += width * height
            last_y = max(last_y, y)
            
        return hv

    def get_frontier(self) -> List[Dict[str, float]]:
        """Get current Pareto frontier as list of dicts."""
        return [
            {
                self.objective_names[i]: (
                    val if self.directions[i] == 'maximize' else -val
                )
                for i, val in enumerate(solution_tuple)
            }
            for solution_tuple in self.frontier
        ]
    
    def get_report(self) -> Dict:
        """Get comprehensive frontier analysis report."""
        return {
            'metrics': {
                'frontier_size': len(self.frontier),
                'hypervolume': self.hv_history[-1] if self.hv_history else 0.0,
            },
            'frontier': self.get_frontier()
        }

    def reset(self):
        """Reset optimizer state."""
        self.frontier.clear()
        self.hv_history.clear()
        self.solution_metadata.clear()
