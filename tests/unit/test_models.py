"""Unit tests for data models."""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models import Flight, Crew, Pairing, LegalRules, AircraftType, CrewRank, Qualification


class TestFlight:
    """Tests for Flight model."""

    def test_duration(self, simple_flights):
        """Test flight duration calculation."""
        f1 = simple_flights[0]
        assert f1.duration_hours == 3.0

    def test_connection_valid(self, simple_flights, default_rules):
        """Test valid connection between flights."""
        f1, f2 = simple_flights
        # f1 arrives at LAX at 11:00, f2 departs LAX at 14:00 = 3 hour connection
        assert f1.can_connect_to(f2, default_rules.min_connection_time)

    def test_connection_invalid_location(self, simple_flights, default_rules):
        """Test invalid connection due to location mismatch."""
        f1 = simple_flights[0]  # JFK -> LAX
        # Create flight from SFO (different location)
        f_sfo = Flight(
            id="F3",
            flight_number="AA102",
            origin="SFO",
            destination="JFK",
            departure=datetime(2024, 1, 1, 14, 0),
            arrival=datetime(2024, 1, 1, 19, 0),
            base_cost=100.0
        )
        assert not f1.can_connect_to(f_sfo, default_rules.min_connection_time)

    def test_connection_invalid_time(self, simple_flights, default_rules):
        """Test invalid connection due to insufficient time."""
        f1 = simple_flights[0]  # arrives at 11:00
        # Create flight departing at 11:30 (only 30 min connection)
        f_tight = Flight(
            id="F3",
            flight_number="AA102",
            origin="LAX",
            destination="JFK",
            departure=datetime(2024, 1, 1, 11, 30),
            arrival=datetime(2024, 1, 1, 14, 30),
            base_cost=100.0
        )
        assert not f1.can_connect_to(f_tight, default_rules.min_connection_time)


class TestCrew:
    """Tests for Crew model."""

    def test_compute_pairing_cost(self, simple_crew):
        """Test pairing cost calculation."""
        crew = simple_crew[0]
        # 8 hours at $50/hour = $400
        assert crew.compute_pairing_cost(8.0) == 400.0

    def test_can_operate_qualified(self, simple_crew, simple_flights):
        """Test crew can operate flights they're qualified for."""
        crew = simple_crew[0]
        flight = simple_flights[0]
        assert crew.can_operate(flight)

    def test_can_operate_unqualified(self, simple_crew):
        """Test crew cannot operate flights they're not qualified for."""
        crew = simple_crew[0]
        wide_body_flight = Flight(
            id="F99",
            flight_number="AA999",
            origin="JFK",
            destination="LAX",
            departure=datetime(2024, 1, 1, 8, 0),
            arrival=datetime(2024, 1, 1, 11, 0),
            base_cost=100.0,
            aircraft_type=AircraftType.WIDE_BODY
        )
        assert not crew.can_operate(wide_body_flight)


class TestPairing:
    """Tests for Pairing model."""

    def test_creation(self, simple_flights, simple_crew):
        """Test pairing creation."""
        pairing = Pairing.create(simple_flights, simple_crew[0])
        assert pairing.crew_id == "C1"
        assert len(pairing.flights) == 2

    def test_base_constraint(self, simple_flights, simple_crew):
        """Test pairing base constraints."""
        pairing = Pairing.create(simple_flights, simple_crew[0])
        assert pairing.start_base == "JFK"
        assert pairing.end_base == "JFK"

    def test_duty_hours(self, simple_flights, simple_crew):
        """Test duty hours calculation."""
        pairing = Pairing.create(simple_flights, simple_crew[0])
        # 8am to 5pm = 9 hours
        assert pairing.total_duty_hours == 9.0

    def test_flight_hours(self, simple_flights, simple_crew):
        """Test flight hours calculation."""
        pairing = Pairing.create(simple_flights, simple_crew[0])
        # Two 3-hour flights = 6 hours
        assert pairing.total_flight_hours == 6.0

    def test_is_legal_valid(self, simple_flights, simple_crew, default_rules):
        """Test valid pairing is recognized as legal."""
        pairing = Pairing.create(simple_flights, simple_crew[0])
        assert pairing.is_legal(simple_crew[0], default_rules)

    def test_is_legal_wrong_base(self, simple_flights, default_rules):
        """Test pairing with wrong base is illegal."""
        crew_lax = Crew(
            id="C2",
            name="LAX Pilot",
            base="LAX",
            rank=CrewRank.CAPTAIN,
            hourly_cost=50.0,
            qualifications={Qualification.NARROW_BODY}
        )
        pairing = Pairing.create(simple_flights, crew_lax)
        # Starts at JFK but crew is based at LAX
        assert not pairing.is_legal(crew_lax, default_rules)

    def test_covers_flight(self, simple_flights, simple_crew):
        """Test flight coverage check."""
        pairing = Pairing.create(simple_flights, simple_crew[0])
        assert pairing.covers_flight("F1")
        assert pairing.covers_flight("F2")
        assert not pairing.covers_flight("F99")

    def test_compute_cost(self, simple_flights, simple_crew):
        """Test pairing cost computation."""
        pairing = Pairing.create(simple_flights, simple_crew[0])
        crew = simple_crew[0]
        # Base cost: 100 + 100 = 200
        # Duty cost: 9 hours * 50 = 450
        # Total: 650
        assert pairing.compute_cost(crew) == 650.0


class TestLegalRules:
    """Tests for LegalRules model."""

    def test_property_conversions(self, default_rules):
        """Test hour conversion properties."""
        assert default_rules.min_connection_hours == 1.0
        assert default_rules.max_duty_hours == 10.0
        assert default_rules.min_rest_hours == 10.0
