"""Crew data model."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from models.flight import Flight


class CrewRank(Enum):
    """Crew member ranks."""
    CAPTAIN = "captain"
    FIRST_OFFICER = "first_officer"
    FLIGHT_ATTENDANT = "flight_attendant"


class Qualification(Enum):
    """Crew qualifications for aircraft types."""
    NARROW_BODY = "narrow"
    WIDE_BODY = "wide"
    REGIONAL = "regional"
    INTERNATIONAL = "international"


@dataclass
class Crew:
    """
    Represents a crew member.

    Attributes:
        id: Unique crew identifier
        name: Crew member name
        base: Home base airport code
        rank: Crew rank/position
        hourly_cost: Cost per hour of duty
        qualifications: Set of qualifications
        max_duty_hours: Maximum duty period length
        min_rest_hours: Minimum rest between duties
        availability_start: Start of availability window
        availability_end: End of availability window
        seniority: Seniority level (higher = more senior)
    """
    id: str
    name: str
    base: str
    rank: CrewRank
    hourly_cost: float
    qualifications: Set[Qualification] = field(default_factory=lambda: {Qualification.NARROW_BODY})
    max_duty_hours: float = 10.0
    min_rest_hours: float = 10.0
    availability_start: Optional[datetime] = None
    availability_end: Optional[datetime] = None
    seniority: int = 0

    def can_operate(self, flight: 'Flight') -> bool:
        """Check if crew can operate given flight based on qualifications."""
        from models.flight import AircraftType

        type_to_qual = {
            AircraftType.NARROW_BODY: Qualification.NARROW_BODY,
            AircraftType.WIDE_BODY: Qualification.WIDE_BODY,
            AircraftType.REGIONAL: Qualification.REGIONAL,
        }
        required_qual = type_to_qual.get(flight.aircraft_type, Qualification.NARROW_BODY)
        return required_qual in self.qualifications

    def is_available(self, start: datetime, end: datetime) -> bool:
        """Check if crew is available during given time window."""
        if self.availability_start and start < self.availability_start:
            return False
        if self.availability_end and end > self.availability_end:
            return False
        return True

    def compute_pairing_cost(self, duty_hours: float) -> float:
        """Compute cost for a pairing with given duty hours."""
        return self.hourly_cost * duty_hours

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Crew):
            return self.id == other.id
        return False

    def __repr__(self) -> str:
        return f"Crew({self.id}: {self.name}, Base={self.base})"
