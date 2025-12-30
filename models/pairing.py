"""Pairing data model."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, FrozenSet, TYPE_CHECKING
import hashlib

if TYPE_CHECKING:
    from models.flight import Flight
    from models.crew import Crew
    from models.legal_rules import LegalRules


@dataclass
class Pairing:
    """
    Represents a crew pairing (sequence of flights).

    In the 3D model, each pairing is associated with a specific crew member.
    This is the key difference from traditional 2D pairing models.

    Attributes:
        id: Unique pairing identifier
        flights: Ordered list of flights in this pairing
        crew_id: ID of the crew member this pairing is assigned to
        reduced_cost: Reduced cost at time of generation (for CG)
    """
    id: str
    flights: List['Flight']
    crew_id: str
    reduced_cost: Optional[float] = None

    @classmethod
    def create(cls, flights: List['Flight'], crew: 'Crew') -> 'Pairing':
        """Factory method to create a pairing with auto-generated ID."""
        # Generate deterministic ID from flight sequence and crew
        flight_ids = "-".join(f.id for f in flights)
        hash_input = f"{crew.id}:{flight_ids}"
        pairing_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return cls(id=f"P_{pairing_id}", flights=list(flights), crew_id=crew.id)

    @property
    def flight_ids(self) -> FrozenSet[str]:
        """Set of flight IDs covered by this pairing."""
        return frozenset(f.id for f in self.flights)

    @property
    def start_base(self) -> Optional[str]:
        """Starting airport of the pairing."""
        return self.flights[0].origin if self.flights else None

    @property
    def end_base(self) -> Optional[str]:
        """Ending airport of the pairing."""
        return self.flights[-1].destination if self.flights else None

    @property
    def start_time(self) -> Optional[datetime]:
        """Departure time of first flight."""
        return self.flights[0].departure if self.flights else None

    @property
    def end_time(self) -> Optional[datetime]:
        """Arrival time of last flight."""
        return self.flights[-1].arrival if self.flights else None

    @property
    def total_duty_hours(self) -> float:
        """Total duty time from first departure to last arrival."""
        if not self.flights:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() / 3600

    @property
    def total_flight_hours(self) -> float:
        """Total time spent actually flying."""
        return sum(f.duration_hours for f in self.flights)

    @property
    def total_base_cost(self) -> float:
        """Sum of base costs for all flights."""
        return sum(f.base_cost for f in self.flights)

    def compute_cost(self, crew: 'Crew') -> float:
        """
        Compute total cost for this pairing assigned to given crew.

        Cost = base_flight_cost + crew_duty_cost
        """
        base_cost = self.total_base_cost
        duty_cost = crew.compute_pairing_cost(self.total_duty_hours)
        return base_cost + duty_cost

    def is_legal(self, crew: 'Crew', rules: 'LegalRules') -> bool:
        """
        Validate this pairing against crew constraints and legal rules.

        Checks:
        1. Starts and ends at crew's home base
        2. Duty time within limits
        3. All connections are legal
        4. Crew is qualified for all flights
        """
        if not self.flights:
            return False

        # Check base constraint
        if self.start_base != crew.base or self.end_base != crew.base:
            return False

        # Check duty time
        if self.total_duty_hours > crew.max_duty_hours:
            return False

        # Check flight time
        if self.total_flight_hours > rules.max_flight_hours:
            return False

        # Check number of flights
        if len(self.flights) > rules.max_flights_per_duty:
            return False

        # Check connections
        for i in range(len(self.flights) - 1):
            if not self.flights[i].can_connect_to(
                self.flights[i + 1],
                rules.min_connection_time
            ):
                return False
            # Check max connection time
            conn_time = self.flights[i + 1].departure - self.flights[i].arrival
            if conn_time > rules.max_connection_time:
                return False

        # Check qualifications
        for flight in self.flights:
            if not crew.can_operate(flight):
                return False

        return True

    def covers_flight(self, flight_id: str) -> bool:
        """Check if this pairing covers a specific flight."""
        return flight_id in self.flight_ids

    def get_coefficient(self, flight_id: str) -> int:
        """
        Get the A_ij coefficient for constraint matrix.
        Returns 1 if this pairing covers the flight, 0 otherwise.
        """
        return 1 if self.covers_flight(flight_id) else 0

    def __hash__(self) -> int:
        return hash((self.id, self.crew_id))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Pairing):
            return self.id == other.id and self.crew_id == other.crew_id
        return False

    def __repr__(self) -> str:
        flight_str = " -> ".join(
            f"{f.origin}-{f.destination}" for f in self.flights
        )
        return f"Pairing({self.id} for {self.crew_id}: {flight_str})"
