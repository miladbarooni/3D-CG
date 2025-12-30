"""Solution data model."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from models.pairing import Pairing
from models.flight import Flight
from models.crew import Crew


@dataclass
class SolutionStatistics:
    """Statistics about a solution."""
    total_cost: float
    total_flight_coverage: int
    total_crew_assigned: int
    total_duty_hours: float
    average_duty_hours: float
    iterations: int
    solve_time_seconds: float
    columns_generated: int
    is_integer: bool
    optimality_gap: Optional[float] = None


@dataclass
class CrewAssignment:
    """Assignment of a crew member to a pairing."""
    crew: Crew
    pairing: Pairing
    cost: float

    @property
    def flights(self) -> List[Flight]:
        """Get flights in this assignment."""
        return self.pairing.flights

    @property
    def duty_hours(self) -> float:
        """Get duty hours for this assignment."""
        return self.pairing.total_duty_hours


@dataclass
class Solution:
    """
    Complete solution to the crew scheduling problem.
    """
    assignments: Dict[str, CrewAssignment]  # crew_id -> assignment
    statistics: SolutionStatistics
    timestamp: datetime = field(default_factory=datetime.now)

    # Iteration history for analysis
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        """Get total solution cost."""
        return self.statistics.total_cost

    @property
    def is_feasible(self) -> bool:
        """Check if solution is feasible."""
        return self.statistics.is_integer

    def get_crew_assignment(self, crew_id: str) -> Optional[CrewAssignment]:
        """Get assignment for a specific crew member."""
        return self.assignments.get(crew_id)

    def get_flight_coverage(self) -> Dict[str, str]:
        """Get mapping of flight_id -> crew_id covering it."""
        coverage: Dict[str, str] = {}
        for crew_id, assignment in self.assignments.items():
            for flight in assignment.flights:
                coverage[flight.id] = crew_id
        return coverage

    def verify_constraints(
        self,
        flights: List[Flight],
        crew: List[Crew]
    ) -> Dict[str, bool]:
        """
        Verify all constraints are satisfied.

        Returns dict of constraint_name -> satisfied
        """
        coverage = self.get_flight_coverage()

        return {
            "all_flights_covered": all(f.id in coverage for f in flights),
            "each_flight_once": len(coverage) == len(flights),
            "all_crew_assigned": len(self.assignments) == len(crew),
            "base_constraints": all(
                a.pairing.start_base == a.crew.base and
                a.pairing.end_base == a.crew.base
                for a in self.assignments.values()
            ),
            "duty_limits": all(
                a.duty_hours <= a.crew.max_duty_hours
                for a in self.assignments.values()
            )
        }

    def to_dict(self) -> dict:
        """Serialize solution to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_cost": self.total_cost,
            "assignments": {
                crew_id: {
                    "crew_id": assignment.crew.id,
                    "crew_name": assignment.crew.name,
                    "crew_base": assignment.crew.base,
                    "pairing_id": assignment.pairing.id,
                    "flights": [
                        {
                            "id": f.id,
                            "origin": f.origin,
                            "destination": f.destination,
                            "departure": f.departure.isoformat(),
                            "arrival": f.arrival.isoformat()
                        }
                        for f in assignment.flights
                    ],
                    "cost": assignment.cost,
                    "duty_hours": assignment.duty_hours
                }
                for crew_id, assignment in self.assignments.items()
            },
            "statistics": {
                "total_cost": self.statistics.total_cost,
                "total_flight_coverage": self.statistics.total_flight_coverage,
                "total_crew_assigned": self.statistics.total_crew_assigned,
                "total_duty_hours": self.statistics.total_duty_hours,
                "average_duty_hours": self.statistics.average_duty_hours,
                "iterations": self.statistics.iterations,
                "solve_time_seconds": self.statistics.solve_time_seconds,
                "columns_generated": self.statistics.columns_generated,
                "is_integer": self.statistics.is_integer
            }
        }

    def print_summary(self) -> None:
        """Print a formatted summary of the solution."""
        print("\n" + "=" * 60)
        print("                    OPTIMAL SOLUTION")
        print("=" * 60)
        print(f"Total Cost: {self.total_cost:.2f}")
        print(f"Solve Time: {self.statistics.solve_time_seconds:.2f} seconds")
        print(f"Iterations: {self.statistics.iterations}")
        print(f"Columns Generated: {self.statistics.columns_generated}")
        print()

        for crew_id, assignment in sorted(self.assignments.items()):
            print(f"Crew {crew_id} (Base: {assignment.crew.base}):")
            for flight in assignment.flights:
                print(
                    f"  {flight.id}: {flight.origin} -> {flight.destination} "
                    f"({flight.departure.strftime('%m/%d %H:%M')}-"
                    f"{flight.arrival.strftime('%H:%M')})"
                )
            print(f"  Duty: {assignment.duty_hours:.1f} hours, Cost: {assignment.cost:.2f}")
            print()

        print("=" * 60)

    def __repr__(self) -> str:
        return (
            f"Solution(cost={self.total_cost:.2f}, "
            f"crew={len(self.assignments)}, "
            f"feasible={self.is_feasible})"
        )
