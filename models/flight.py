"""Flight data model."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class AircraftType(Enum):
    """Types of aircraft."""
    NARROW_BODY = "narrow"
    WIDE_BODY = "wide"
    REGIONAL = "regional"


@dataclass
class Flight:
    """
    Represents a single flight leg.

    Attributes:
        id: Unique flight identifier
        flight_number: Airline flight number (e.g., "AA123")
        origin: Departure airport code
        destination: Arrival airport code
        departure: Scheduled departure time (UTC)
        arrival: Scheduled arrival time (UTC)
        base_cost: Operating cost for this flight
        aircraft_type: Type of aircraft
        required_crew: Number of crew needed (for future extension)
    """
    id: str
    flight_number: str
    origin: str
    destination: str
    departure: datetime
    arrival: datetime
    base_cost: float
    aircraft_type: AircraftType = AircraftType.NARROW_BODY
    required_crew: int = 1

    @property
    def duration(self) -> timedelta:
        """Flight duration."""
        return self.arrival - self.departure

    @property
    def duration_hours(self) -> float:
        """Flight duration in hours."""
        return self.duration.total_seconds() / 3600

    def can_connect_to(
        self,
        other: 'Flight',
        min_connection: timedelta
    ) -> bool:
        """
        Check if this flight can legally connect to another flight.

        Args:
            other: The potential subsequent flight
            min_connection: Minimum required connection time

        Returns:
            True if connection is legal
        """
        # Must arrive at same airport as next departure
        if self.destination != other.origin:
            return False

        # Must have enough connection time
        connection_time = other.departure - self.arrival
        if connection_time < min_connection:
            return False

        return True

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Flight):
            return self.id == other.id
        return False

    def __repr__(self) -> str:
        return (
            f"Flight({self.id}: {self.origin}→{self.destination} "
            f"{self.departure.strftime('%m/%d %H:%M')}-"
            f"{self.arrival.strftime('%H:%M')})"
        )
