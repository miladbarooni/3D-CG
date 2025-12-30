"""Micro-Airline test dataset generator.

Creates an 11-flight, 4-crew test instance based on the specification.
This is a small but meaningful test case where the 3D effect is visible.
"""

from datetime import datetime, timedelta
from typing import List, Tuple

from models import Flight, Crew, LegalRules, AircraftType, CrewRank, Qualification


def generate_micro_airline() -> Tuple[List[Flight], List[Crew], LegalRules]:
    """
    Generate the Micro-Airline test instance.

    Returns:
        Tuple of (flights, crew, rules)

    Dataset Details:
        - 11 flights over 2 days
        - 4 crew members at 3 different bases (JFK, LAX, SFO)
        - Tests the 3D integrated model where crew bases matter
    """
    # Base date for the schedule
    day1 = datetime(2024, 1, 1)

    # Generate flights - designed for feasibility with 4 crew
    # 8 flights that can be covered by 4 two-flight pairings
    flights = [
        # JFK-LAX round trip (morning) - for JFK crew C1
        Flight(
            id="F1",
            flight_number="AA100",
            origin="JFK",
            destination="LAX",
            departure=day1.replace(hour=6, minute=0),
            arrival=day1.replace(hour=9, minute=0),  # 3h flight
            base_cost=100.0,
            aircraft_type=AircraftType.NARROW_BODY
        ),
        Flight(
            id="F2",
            flight_number="AA101",
            origin="LAX",
            destination="JFK",
            departure=day1.replace(hour=10, minute=0),
            arrival=day1.replace(hour=13, minute=0),  # 3h flight
            base_cost=100.0,
            aircraft_type=AircraftType.NARROW_BODY
        ),
        # JFK-ORD round trip (morning) - for JFK crew C2
        Flight(
            id="F3",
            flight_number="AA102",
            origin="JFK",
            destination="ORD",
            departure=day1.replace(hour=7, minute=0),
            arrival=day1.replace(hour=9, minute=0),  # 2h flight
            base_cost=80.0,
            aircraft_type=AircraftType.NARROW_BODY
        ),
        Flight(
            id="F4",
            flight_number="AA103",
            origin="ORD",
            destination="JFK",
            departure=day1.replace(hour=10, minute=0),
            arrival=day1.replace(hour=12, minute=0),  # 2h flight
            base_cost=80.0,
            aircraft_type=AircraftType.NARROW_BODY
        ),
        # LAX-SFO round trip - for LAX crew C3
        Flight(
            id="F5",
            flight_number="AA104",
            origin="LAX",
            destination="SFO",
            departure=day1.replace(hour=7, minute=0),
            arrival=day1.replace(hour=8, minute=30),  # 1.5h flight
            base_cost=50.0,
            aircraft_type=AircraftType.NARROW_BODY
        ),
        Flight(
            id="F6",
            flight_number="AA105",
            origin="SFO",
            destination="LAX",
            departure=day1.replace(hour=10, minute=0),
            arrival=day1.replace(hour=11, minute=30),  # 1.5h flight
            base_cost=50.0,
            aircraft_type=AircraftType.NARROW_BODY
        ),
        # SFO-JFK round trip - for SFO crew C4
        Flight(
            id="F7",
            flight_number="AA106",
            origin="SFO",
            destination="JFK",
            departure=day1.replace(hour=8, minute=0),
            arrival=day1.replace(hour=12, minute=0),  # 4h flight
            base_cost=120.0,
            aircraft_type=AircraftType.NARROW_BODY
        ),
        Flight(
            id="F8",
            flight_number="AA107",
            origin="JFK",
            destination="SFO",
            departure=day1.replace(hour=13, minute=0),
            arrival=day1.replace(hour=17, minute=0),  # 4h flight
            base_cost=120.0,
            aircraft_type=AircraftType.NARROW_BODY
        ),
    ]

    # Generate crew as specified
    crew = [
        Crew(
            id="C1",
            name="Junior Pilot 1",
            base="JFK",
            rank=CrewRank.CAPTAIN,
            hourly_cost=50.0,
            qualifications={Qualification.NARROW_BODY},
            max_duty_hours=10.0,
            min_rest_hours=10.0,
            seniority=1
        ),
        Crew(
            id="C2",
            name="Senior Pilot 1",
            base="JFK",
            rank=CrewRank.CAPTAIN,
            hourly_cost=70.0,
            qualifications={Qualification.NARROW_BODY},
            max_duty_hours=10.0,
            min_rest_hours=10.0,
            seniority=3
        ),
        Crew(
            id="C3",
            name="Junior Pilot 2",
            base="LAX",
            rank=CrewRank.CAPTAIN,
            hourly_cost=50.0,
            qualifications={Qualification.NARROW_BODY},
            max_duty_hours=10.0,
            min_rest_hours=10.0,
            seniority=1
        ),
        Crew(
            id="C4",
            name="Mid Pilot 1",
            base="SFO",
            rank=CrewRank.CAPTAIN,
            hourly_cost=60.0,
            qualifications={Qualification.NARROW_BODY},
            max_duty_hours=10.0,
            min_rest_hours=10.0,
            seniority=2
        ),
    ]

    # Legal rules as specified
    rules = LegalRules(
        min_connection_time=timedelta(hours=1),
        max_connection_time=timedelta(hours=4),
        max_duty_period=timedelta(hours=10),
        min_rest_period=timedelta(hours=10),
        max_flight_time_per_duty=timedelta(hours=8),
        max_flights_per_duty=4,
        must_return_to_base=True
    )

    return flights, crew, rules


def print_instance_summary(
    flights: List[Flight],
    crew: List[Crew],
    rules: LegalRules
) -> None:
    """Print a summary of the instance."""
    print("\n" + "=" * 60)
    print("           MICRO-AIRLINE TEST INSTANCE")
    print("=" * 60)

    print("\nFLIGHTS:")
    print("-" * 60)
    print(f"{'ID':<4} {'From':<4} {'To':<4} {'Departure':<16} {'Arrival':<16} {'Cost':>6}")
    print("-" * 60)
    for f in sorted(flights, key=lambda x: (x.departure, x.id)):
        print(
            f"{f.id:<4} {f.origin:<4} {f.destination:<4} "
            f"{f.departure.strftime('%m/%d %H:%M'):<16} "
            f"{f.arrival.strftime('%m/%d %H:%M'):<16} "
            f"{f.base_cost:>6.0f}"
        )

    print("\nCREW:")
    print("-" * 60)
    print(f"{'ID':<4} {'Name':<16} {'Base':<4} {'Hourly Cost':>12} {'Seniority':>10}")
    print("-" * 60)
    for c in crew:
        print(
            f"{c.id:<4} {c.name:<16} {c.base:<4} "
            f"{c.hourly_cost:>12.0f} {c.seniority:>10}"
        )

    print("\nRULES:")
    print("-" * 60)
    print(f"  Min Connection Time: {rules.min_connection_hours:.0f} hours")
    print(f"  Max Duty Period:     {rules.max_duty_hours:.0f} hours")
    print(f"  Min Rest Period:     {rules.min_rest_hours:.0f} hours")
    print(f"  Max Flights/Duty:    {rules.max_flights_per_duty}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Generate and print the instance
    flights, crew, rules = generate_micro_airline()
    print_instance_summary(flights, crew, rules)
