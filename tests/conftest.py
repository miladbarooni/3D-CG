"""Pytest fixtures for crew scheduling tests."""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Flight, Crew, LegalRules, AircraftType, CrewRank, Qualification
from data.generators.micro_airline import generate_micro_airline


@pytest.fixture
def simple_flights():
    """Two flights forming a round trip from JFK."""
    return [
        Flight(
            id="F1",
            flight_number="AA100",
            origin="JFK",
            destination="LAX",
            departure=datetime(2024, 1, 1, 8, 0),
            arrival=datetime(2024, 1, 1, 11, 0),
            base_cost=100.0
        ),
        Flight(
            id="F2",
            flight_number="AA101",
            origin="LAX",
            destination="JFK",
            departure=datetime(2024, 1, 1, 14, 0),
            arrival=datetime(2024, 1, 1, 17, 0),
            base_cost=100.0
        )
    ]


@pytest.fixture
def simple_crew():
    """One crew member based at JFK."""
    return [
        Crew(
            id="C1",
            name="Captain Smith",
            base="JFK",
            rank=CrewRank.CAPTAIN,
            hourly_cost=50.0,
            qualifications={Qualification.NARROW_BODY}
        )
    ]


@pytest.fixture
def default_rules():
    """Standard legal rules."""
    return LegalRules(
        min_connection_time=timedelta(hours=1),
        max_duty_period=timedelta(hours=10),
        min_rest_period=timedelta(hours=10)
    )


@pytest.fixture
def micro_airline():
    """Full micro-airline test instance."""
    return generate_micro_airline()
