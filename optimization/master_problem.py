"""Restricted Master Problem (RMP) for column generation."""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pulp

from models import Flight, Crew, Pairing, LegalRules


@dataclass
class DualValues:
    """Container for dual values from LP relaxation."""
    flight_duals: Dict[str, float]   # pi_i for each flight
    crew_duals: Dict[str, float]     # sigma_k for each crew

    def get_flight_dual(self, flight_id: str) -> float:
        """Get dual value for a flight."""
        return self.flight_duals.get(flight_id, 0.0)

    def get_crew_dual(self, crew_id: str) -> float:
        """Get dual value for a crew member."""
        return self.crew_duals.get(crew_id, 0.0)


class MasterProblem:
    """
    Restricted Master Problem (RMP) for column generation.

    Manages:
    - LP/MIP formulation
    - Column (pairing) pool
    - Dual value extraction
    - Solution retrieval
    """

    def __init__(
        self,
        flights: List[Flight],
        crew: List[Crew],
        rules: LegalRules
    ):
        self.flights = {f.id: f for f in flights}
        self.crew = {c.id: c for c in crew}
        self.rules = rules

        # Column pool: (pairing_id, crew_id) -> Pairing
        self.pairings: Dict[Tuple[str, str], Pairing] = {}

        # Model objects (rebuilt on each solve)
        self.model: Optional[pulp.LpProblem] = None
        self.variables: Dict[Tuple[str, str], pulp.LpVariable] = {}

        # Constraint references (for dual extraction)
        self.flight_constraints: Dict[str, pulp.LpConstraint] = {}
        self.crew_constraints: Dict[str, pulp.LpConstraint] = {}

        # Last solution
        self._last_objective: Optional[float] = None
        self._last_solution: Optional[Dict[Tuple[str, str], float]] = None

    def add_pairing(self, pairing: Pairing) -> bool:
        """
        Add a new pairing to the column pool.

        Returns True if pairing was added (not a duplicate).
        """
        key = (pairing.id, pairing.crew_id)
        if key in self.pairings:
            return False

        self.pairings[key] = pairing
        return True

    def add_pairings(self, pairings: List[Pairing]) -> int:
        """Add multiple pairings. Returns count of new pairings added."""
        return sum(1 for p in pairings if self.add_pairing(p))

    def build_model(self, relax: bool = True) -> pulp.LpProblem:
        """
        Build or rebuild the optimization model.

        Args:
            relax: If True, use continuous variables (LP relaxation)
                   If False, use binary variables (MIP)

        Returns:
            The PuLP model object
        """
        self.model = pulp.LpProblem("CrewScheduling3D", pulp.LpMinimize)
        self.variables = {}
        self.flight_constraints = {}
        self.crew_constraints = {}

        # Create variables for each (pairing, crew) combination
        for (pairing_id, crew_id), pairing in self.pairings.items():
            crew = self.crew[crew_id]
            cost = pairing.compute_cost(crew)

            var_name = f"x_{pairing_id}_{crew_id}"
            if relax:
                var = pulp.LpVariable(var_name, 0, 1, cat=pulp.LpContinuous)
            else:
                var = pulp.LpVariable(var_name, cat=pulp.LpBinary)

            self.variables[(pairing_id, crew_id)] = var

        # Objective: minimize total cost
        objective_terms = []
        for (pairing_id, crew_id), var in self.variables.items():
            pairing = self.pairings[(pairing_id, crew_id)]
            crew = self.crew[crew_id]
            cost = pairing.compute_cost(crew)
            objective_terms.append(cost * var)

        self.model += pulp.lpSum(objective_terms), "TotalCost"

        # Constraint 1: Flight coverage (each flight exactly once)
        for flight_id in self.flights:
            covering_vars = []
            for (pairing_id, crew_id), var in self.variables.items():
                pairing = self.pairings[(pairing_id, crew_id)]
                if pairing.covers_flight(flight_id):
                    covering_vars.append(var)

            if covering_vars:
                constraint_name = f"FlightCoverage_{flight_id}"
                constraint = pulp.lpSum(covering_vars) == 1
                self.model += constraint, constraint_name
                # Store reference for dual extraction
                self.flight_constraints[flight_id] = self.model.constraints[constraint_name]

        # Constraint 2: Crew assignment (each crew exactly one pairing)
        for crew_id in self.crew:
            crew_vars = [
                var for (pid, cid), var in self.variables.items()
                if cid == crew_id
            ]

            if crew_vars:
                constraint_name = f"CrewAssignment_{crew_id}"
                constraint = pulp.lpSum(crew_vars) == 1
                self.model += constraint, constraint_name
                # Store reference for dual extraction
                self.crew_constraints[crew_id] = self.model.constraints[constraint_name]

        return self.model

    def solve(self, solver: str = "CBC") -> Tuple[float, Dict[Tuple[str, str], float]]:
        """
        Solve the current model.

        Args:
            solver: Solver to use ("CBC", "GUROBI", "CPLEX")

        Returns:
            (objective_value, solution_dict)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Select solver
        if solver.upper() == "GUROBI":
            try:
                slv = pulp.GUROBI_CMD(msg=0)
            except Exception:
                slv = pulp.PULP_CBC_CMD(msg=0)
        elif solver.upper() == "CPLEX":
            try:
                slv = pulp.CPLEX_CMD(msg=0)
            except Exception:
                slv = pulp.PULP_CBC_CMD(msg=0)
        else:
            slv = pulp.PULP_CBC_CMD(msg=0)

        # Solve
        self.model.solve(slv)

        # Check status
        if self.model.status != pulp.LpStatusOptimal:
            status_name = pulp.LpStatus[self.model.status]
            raise RuntimeError(
                f"Solver did not find optimal solution. Status: {status_name}"
            )

        # Extract solution
        self._last_objective = pulp.value(self.model.objective)
        self._last_solution = {
            (pid, cid): pulp.value(var) if pulp.value(var) is not None else 0.0
            for (pid, cid), var in self.variables.items()
        }

        return self._last_objective, self._last_solution

    def get_duals(self) -> DualValues:
        """
        Extract dual values after solving LP relaxation.

        Must be called after solve() on an LP relaxation.

        Returns:
            DualValues object with flight and crew duals
        """
        if self.model is None or self.model.status != pulp.LpStatusOptimal:
            raise ValueError("Model must be solved optimally first.")

        # Extract flight duals (pi_i)
        flight_duals = {}
        for flight_id, constraint in self.flight_constraints.items():
            # For PuLP with CBC, duals are accessed via the constraint's pi attribute
            try:
                dual = constraint.pi
                flight_duals[flight_id] = dual if dual is not None else 0.0
            except AttributeError:
                flight_duals[flight_id] = 0.0

        # Extract crew duals (sigma_k)
        crew_duals = {}
        for crew_id, constraint in self.crew_constraints.items():
            try:
                dual = constraint.pi
                crew_duals[crew_id] = dual if dual is not None else 0.0
            except AttributeError:
                crew_duals[crew_id] = 0.0

        return DualValues(flight_duals=flight_duals, crew_duals=crew_duals)

    def get_solution_pairings(
        self,
        threshold: float = 0.5
    ) -> Dict[str, Pairing]:
        """
        Get pairings selected in solution (value > threshold).

        Returns:
            Dict mapping crew_id to their assigned pairing
        """
        if self._last_solution is None:
            raise ValueError("Must call solve() first.")

        assignments = {}
        for (pairing_id, crew_id), value in self._last_solution.items():
            if value is not None and value > threshold:
                assignments[crew_id] = self.pairings[(pairing_id, crew_id)]

        return assignments

    @property
    def num_columns(self) -> int:
        """Number of columns (pairings) in the pool."""
        return len(self.pairings)

    @property
    def num_flights(self) -> int:
        """Number of flights to cover."""
        return len(self.flights)

    @property
    def num_crew(self) -> int:
        """Number of crew members."""
        return len(self.crew)

    @property
    def last_objective(self) -> Optional[float]:
        """Get the last objective value."""
        return self._last_objective

    def __repr__(self) -> str:
        return (
            f"MasterProblem(flights={self.num_flights}, "
            f"crew={self.num_crew}, columns={self.num_columns})"
        )
