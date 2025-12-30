"""Legal rules and regulatory constraints model."""

from dataclasses import dataclass
from datetime import timedelta


@dataclass
class LegalRules:
    """
    Regulatory and operational rules for crew scheduling.

    These rules constrain what constitutes a valid pairing.
    """
    # Connection rules
    min_connection_time: timedelta = timedelta(hours=1)
    max_connection_time: timedelta = timedelta(hours=4)

    # Duty rules
    max_duty_period: timedelta = timedelta(hours=10)
    min_rest_period: timedelta = timedelta(hours=10)

    # Flying rules
    max_flight_time_per_duty: timedelta = timedelta(hours=8)
    max_flights_per_duty: int = 4

    # Base rules
    must_return_to_base: bool = True

    # Cost parameters
    deadhead_cost_factor: float = 0.5  # Cost of positioning flights
    overtime_cost_factor: float = 1.5  # Multiplier for overtime

    @property
    def min_connection_hours(self) -> float:
        """Minimum connection time in hours."""
        return self.min_connection_time.total_seconds() / 3600

    @property
    def max_connection_hours(self) -> float:
        """Maximum connection time in hours."""
        return self.max_connection_time.total_seconds() / 3600

    @property
    def max_duty_hours(self) -> float:
        """Maximum duty period in hours."""
        return self.max_duty_period.total_seconds() / 3600

    @property
    def min_rest_hours(self) -> float:
        """Minimum rest period in hours."""
        return self.min_rest_period.total_seconds() / 3600

    @property
    def max_flight_hours(self) -> float:
        """Maximum flight time per duty in hours."""
        return self.max_flight_time_per_duty.total_seconds() / 3600
