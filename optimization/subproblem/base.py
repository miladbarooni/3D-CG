"""Base class for pricing subproblems."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from dataclasses import dataclass

from models import Flight, Crew, Pairing, LegalRules, FlightNetwork


@dataclass
class SubproblemResult:
    """Result from solving a pricing subproblem."""
    pairing: Optional[Pairing]
    reduced_cost: float
    solve_time_ms: float
    nodes_explored: int = 0
    labels_created: int = 0

    @property
    def has_negative_reduced_cost(self) -> bool:
        """Check if a column with negative reduced cost was found."""
        return self.pairing is not None and self.reduced_cost < 0


class PricingSubproblem(ABC):
    """
    Abstract base class for pricing subproblems.

    Each subproblem finds the minimum reduced cost pairing
    for a specific crew member.
    """

    def __init__(
        self,
        flights: List[Flight],
        crew: Crew,
        rules: LegalRules
    ):
        self.flights = flights
        self.crew = crew
        self.rules = rules
        self.network = FlightNetwork(flights, crew, rules)

    @abstractmethod
    def solve(
        self,
        flight_duals: Dict[str, float],
        crew_dual: float,
        time_limit_ms: int = 10000
    ) -> SubproblemResult:
        """
        Solve the pricing problem.

        Args:
            flight_duals: Dual values pi_i for each flight
            crew_dual: Dual value sigma_k for this crew
            time_limit_ms: Maximum solve time in milliseconds

        Returns:
            SubproblemResult containing best pairing found (if any)
        """
        pass

    def compute_reduced_cost(
        self,
        pairing: Pairing,
        flight_duals: Dict[str, float],
        crew_dual: float
    ) -> float:
        """
        Compute reduced cost for a pairing.

        c_bar_{jk} = c_{jk} - Sum(pi_i for i in j) - sigma_k
        """
        # Direct cost
        cost = pairing.compute_cost(self.crew)

        # Subtract flight duals
        for flight in pairing.flights:
            cost -= flight_duals.get(flight.id, 0.0)

        # Subtract crew dual
        cost -= crew_dual

        return cost
