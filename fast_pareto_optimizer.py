from sortedcontainers import SortedList
from typing import List, Tuple, Optional, Dict
import numpy as np


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
        # Sort by first objective for efficient dominance checking
        self.frontier = SortedList(key=lambda x: x[0])
        
        # Track hypervolume history
        self.hv_history = []
        
        # Metadata storage
        self.solution_metadata = {}
    
    def add_solution(self, objectives: Dict[str, float] = None, 
                    solution_tuple: Tuple[float, ...] = None,
                    metadata: Optional[Dict] = None) -> dict:
        """
        Add solution to frontier and update Pareto set.
        
        Args:
            objectives: Dict mapping objective names to values
            solution_tuple: Tuple of objective values (alternative to objectives dict)
            metadata: Optional metadata to store with solution
            
        Returns:
            Dict with update information
        """
        # Convert dict to tuple if provided, applying direction transformations
        if objectives is not None:
            solution = tuple(
                objectives[name] if self.directions[i] == 'maximize' 
                else -objectives[name]
                for i, name in enumerate(self.objective_names)
            )
        elif solution_tuple is not None:
            solution = solution_tuple
        else:
            raise ValueError("Either objectives or solution_tuple must be provided")
        
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
        
        # Store metadata
        if metadata:
            self.solution_metadata[solution] = metadata
        
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
            # Remove metadata if exists
            if sol in self.solution_metadata:
                del self.solution_metadata[sol]
        
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
    
    def get_hypervolume_history(self) -> List[float]:
        """Get hypervolume history."""
        return self.hv_history.copy()
    
    def get_current_hypervolume(self) -> float:
        """Get current hypervolume."""
        return self.hv_history[-1] if self.hv_history else 0.0
    
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
    
    def reset(self):
        """Reset optimizer state."""
        self.frontier.clear()
        self.hv_history.clear()
        self.solution_metadata.clear()
