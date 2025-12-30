"""Column Generation orchestrator."""

from typing import List, Dict, Optional, Callable, Type, Any
from dataclasses import dataclass, field
import time
import logging

from models import Flight, Crew, Pairing, LegalRules
from models import Solution, SolutionStatistics, CrewAssignment
from optimization.master_problem import MasterProblem, DualValues
from optimization.subproblem.base import PricingSubproblem
from optimization.subproblem.exact_rcspp import ExactRCSPP

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    """Result of a single column generation iteration."""
    iteration: int
    objective: float
    columns_added: int
    best_reduced_cost: float
    solve_time_ms: float
    duals: Optional[DualValues] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "iteration": self.iteration,
            "objective": self.objective,
            "columns_added": self.columns_added,
            "best_reduced_cost": self.best_reduced_cost,
            "solve_time_ms": self.solve_time_ms
        }


class ColumnGeneration:
    """
    Main column generation orchestrator.

    Coordinates the master problem and pricing subproblems
    to solve the integrated crew scheduling problem.
    """

    def __init__(
        self,
        flights: List[Flight],
        crew: List[Crew],
        rules: LegalRules,
        subproblem_class: Type[PricingSubproblem] = ExactRCSPP,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        max_columns_per_iter: int = 20,
        pricing_time_limit_ms: int = 5000,
        callback: Optional[Callable[[IterationResult], None]] = None
    ):
        self.flights = flights
        self.crew = crew
        self.rules = rules
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_columns_per_iter = max_columns_per_iter
        self.pricing_time_limit_ms = pricing_time_limit_ms
        self.callback = callback
        self.subproblem_class = subproblem_class

        # Initialize master problem
        self.master = MasterProblem(flights, crew, rules)

        # Initialize subproblems (one per crew member)
        self.subproblems: Dict[str, PricingSubproblem] = {}
        for c in crew:
            self.subproblems[c.id] = subproblem_class(flights, c, rules)

        # History
        self.iteration_history: List[IterationResult] = []

    def initialize(self) -> int:
        """
        Generate initial feasible pairings.

        Creates pairings to ensure feasibility using multiple strategies:
        1. Enumerate ALL valid two-flight round trips for each crew
        2. Multi-leg pairings via RCSPP
        3. Use high duals to encourage uncovered flight coverage

        Returns:
            Number of initial pairings added
        """
        logger.info("Generating initial pairings...")
        initial_count = 0

        # Strategy 1: Enumerate ALL valid round-trip pairings for each crew
        for crew in self.crew:
            base_departures = [
                f for f in self.flights
                if f.origin == crew.base
            ]
            base_arrivals = [
                f for f in self.flights
                if f.destination == crew.base
            ]

            # Try ALL combinations of departures and arrivals
            for dep_flight in base_departures:
                for arr_flight in base_arrivals:
                    if dep_flight.id == arr_flight.id:
                        continue
                    if dep_flight.can_connect_to(arr_flight, self.rules.min_connection_time):
                        pairing = Pairing.create([dep_flight, arr_flight], crew)
                        if pairing.is_legal(crew, self.rules):
                            if self.master.add_pairing(pairing):
                                initial_count += 1
                                logger.debug(f"Round-trip pairing for {crew.id}: {pairing}")

        # Strategy 2: Try multi-leg pairings (3 flights) for all crews
        for crew in self.crew:
            initial_count += self._find_all_multi_leg_pairings(crew)

        # Log which flights are covered by initial pairings
        covered_flights = set()
        for pairing in self.master.pairings.values():
            for flight in pairing.flights:
                covered_flights.add(flight.id)

        all_flight_ids = {f.id for f in self.flights}
        uncovered = all_flight_ids - covered_flights

        logger.info(f"Initialized with {initial_count} pairings covering {len(covered_flights)}/{len(all_flight_ids)} flights")
        if uncovered:
            logger.warning(f"Uncovered flights in initial solution: {uncovered}")
            # Try to generate additional pairings covering uncovered flights
            self._generate_additional_pairings(uncovered)

        return initial_count

    def _generate_additional_pairings(self, uncovered_flights: set) -> None:
        """Generate additional pairings to cover uncovered flights."""
        for crew in self.crew:
            subproblem = self.subproblems[crew.id]

            # Use high dual prices for uncovered flights to encourage their inclusion
            flight_duals = {}
            for f in self.flights:
                if f.id in uncovered_flights:
                    flight_duals[f.id] = 1000.0  # High value to encourage coverage
                else:
                    flight_duals[f.id] = 0.0

            result = subproblem.solve(flight_duals, 0.0, time_limit_ms=5000)

            if result.pairing and result.pairing.flights:
                if result.pairing.is_legal(crew, self.rules):
                    if self.master.add_pairing(result.pairing):
                        logger.debug(f"Additional pairing for {crew.id}: {result.pairing}")

    def _find_all_multi_leg_pairings(self, crew: Crew) -> int:
        """Find all valid multi-leg pairings for a crew member."""
        count = 0
        base_departures = [f for f in self.flights if f.origin == crew.base]

        for f1 in base_departures:
            # Find flights connecting from f1
            for f2 in self.flights:
                if f1.id == f2.id:
                    continue
                if not f1.can_connect_to(f2, self.rules.min_connection_time):
                    continue

                # 2-leg pairing if f2 returns to base
                if f2.destination == crew.base:
                    pairing = Pairing.create([f1, f2], crew)
                    if pairing.is_legal(crew, self.rules):
                        if self.master.add_pairing(pairing):
                            logger.debug(f"2-leg pairing for {crew.id}: {pairing}")
                            count += 1

                # Try 3-leg pairing
                for f3 in self.flights:
                    if f3.id in (f1.id, f2.id):
                        continue
                    if not f2.can_connect_to(f3, self.rules.min_connection_time):
                        continue
                    if f3.destination == crew.base:
                        pairing = Pairing.create([f1, f2, f3], crew)
                        if pairing.is_legal(crew, self.rules):
                            if self.master.add_pairing(pairing):
                                logger.debug(f"3-leg pairing for {crew.id}: {pairing}")
                                count += 1

                    # Try 4-leg pairing
                    for f4 in self.flights:
                        if f4.id in (f1.id, f2.id, f3.id):
                            continue
                        if not f3.can_connect_to(f4, self.rules.min_connection_time):
                            continue
                        if f4.destination == crew.base:
                            pairing = Pairing.create([f1, f2, f3, f4], crew)
                            if pairing.is_legal(crew, self.rules):
                                if self.master.add_pairing(pairing):
                                    logger.debug(f"4-leg pairing for {crew.id}: {pairing}")
                                    count += 1

        return count

    def run(self, verbose: bool = True) -> Solution:
        """
        Run the column generation algorithm.

        Args:
            verbose: If True, print progress information

        Returns:
            Solution object with final assignments
        """
        start_time = time.time()

        # Initialize
        self.initialize()

        # Main loop
        converged = False
        for iteration in range(1, self.max_iterations + 1):
            iter_start = time.time()

            # Step 1: Solve master problem (LP relaxation)
            self.master.build_model(relax=True)
            try:
                objective, solution = self.master.solve()
            except RuntimeError as e:
                logger.error(f"Master problem solve failed: {e}")
                break

            duals = self.master.get_duals()

            if verbose:
                logger.info(
                    f"Iter {iteration:3d}: Objective = {objective:10.2f}, "
                    f"Columns = {self.master.num_columns}"
                )

            # Step 2: Solve pricing subproblems
            new_columns: List[Pairing] = []
            best_reduced_cost = float('inf')

            for crew_id, subproblem in self.subproblems.items():
                result = subproblem.solve(
                    duals.flight_duals,
                    duals.get_crew_dual(crew_id),
                    time_limit_ms=self.pricing_time_limit_ms
                )

                if result.pairing and result.reduced_cost < -self.tolerance:
                    new_columns.append(result.pairing)
                    best_reduced_cost = min(best_reduced_cost, result.reduced_cost)

                    logger.debug(
                        f"  {crew_id}: reduced cost = {result.reduced_cost:.4f}"
                    )

            # Step 3: Add columns and check convergence
            columns_to_add = new_columns[:self.max_columns_per_iter]
            columns_added = self.master.add_pairings(columns_to_add)

            iter_time = (time.time() - iter_start) * 1000

            # Record iteration
            iter_result = IterationResult(
                iteration=iteration,
                objective=objective,
                columns_added=columns_added,
                best_reduced_cost=best_reduced_cost if best_reduced_cost < float('inf') else 0.0,
                solve_time_ms=iter_time,
                duals=duals
            )
            self.iteration_history.append(iter_result)

            if self.callback:
                self.callback(iter_result)

            # Check convergence
            if columns_added == 0:
                logger.info(f"Converged at iteration {iteration}")
                converged = True
                break

        if not converged:
            logger.warning(f"Did not converge within {self.max_iterations} iterations")

        # Step 4: Get integer solution
        solution = self._get_integer_solution()

        total_time = time.time() - start_time
        solution.statistics.solve_time_seconds = total_time
        solution.iteration_history = [r.to_dict() for r in self.iteration_history]

        return solution

    def _get_integer_solution(self) -> Solution:
        """
        Obtain integer solution from LP solution.

        Solves MIP with all generated columns.
        """
        # Solve MIP
        self.master.build_model(relax=False)
        try:
            objective, solution = self.master.solve()
        except RuntimeError as e:
            logger.error(f"MIP solve failed: {e}")
            # Return empty solution
            return Solution(
                assignments={},
                statistics=SolutionStatistics(
                    total_cost=float('inf'),
                    total_flight_coverage=0,
                    total_crew_assigned=0,
                    total_duty_hours=0,
                    average_duty_hours=0,
                    iterations=len(self.iteration_history),
                    solve_time_seconds=0,
                    columns_generated=self.master.num_columns,
                    is_integer=False
                )
            )

        # Build solution object
        assignments = {}
        for (pairing_id, crew_id), value in solution.items():
            if value is not None and value > 0.5:
                pairing = self.master.pairings[(pairing_id, crew_id)]
                crew = self.master.crew[crew_id]
                assignments[crew_id] = CrewAssignment(
                    crew=crew,
                    pairing=pairing,
                    cost=pairing.compute_cost(crew)
                )

        total_duty = sum(a.duty_hours for a in assignments.values())

        stats = SolutionStatistics(
            total_cost=objective,
            total_flight_coverage=sum(
                len(a.flights) for a in assignments.values()
            ),
            total_crew_assigned=len(assignments),
            total_duty_hours=total_duty,
            average_duty_hours=total_duty / max(len(assignments), 1),
            iterations=len(self.iteration_history),
            solve_time_seconds=0,  # Updated by caller
            columns_generated=self.master.num_columns,
            is_integer=True
        )

        return Solution(assignments=assignments, statistics=stats)

    @property
    def num_iterations(self) -> int:
        """Number of iterations completed."""
        return len(self.iteration_history)

    @property
    def final_objective(self) -> Optional[float]:
        """Final objective value (if available)."""
        if self.iteration_history:
            return self.iteration_history[-1].objective
        return None
