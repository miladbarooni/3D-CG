"""Small-Airline test dataset generator.

Creates a larger instance with 24 flights and 8 crew members.
Designed to test scalability while still being solvable quickly.

DESIGN PRINCIPLE: Each crew member has exactly 3 dedicated flights
forming a feasible pairing. No flight overlap between intended pairings.
"""

from datetime import datetime, timedelta
from typing import List, Tuple

from models import Flight, Crew, LegalRules, AircraftType, CrewRank, Qualification


def generate_small_airline() -> Tuple[List[Flight], List[Crew], LegalRules]:
    """
    Generate the Small-Airline test instance.

    Returns:
        Tuple of (flights, crew, rules)

    Dataset Details:
        - 24 flights over 1 day (3 flights per crew member)
        - 8 crew members at 4 different bases (JFK, LAX, ORD, SFO)
        - Each crew has a clear 3-flight pairing that fits within limits
        - Total flight time per pairing: 4-6 hours (within 8h limit)
        - Total duty time per pairing: 6-9 hours (within 10h limit)
    """
    day1 = datetime(2024, 1, 1)

    flights = []
    flight_id = 1

    def add_flight(origin, dest, dep_hour, dep_min, duration_hours, cost):
        nonlocal flight_id
        dep = day1.replace(hour=dep_hour, minute=dep_min)
        arr = dep + timedelta(hours=duration_hours)
        flights.append(Flight(
            id=f"F{flight_id}",
            flight_number=f"AA{100 + flight_id}",
            origin=origin,
            destination=dest,
            departure=dep,
            arrival=arr,
            base_cost=cost,
            aircraft_type=AircraftType.NARROW_BODY
        ))
        flight_id += 1

    # === FEASIBILITY BY DESIGN ===
    # Each crew gets exactly 3 dedicated flights forming a valid pairing.
    # No flights are shared between crews.
    # 8 crews x 3 flights = 24 flights total

    # === JFK Hub (2 crew, 6 flights) ===

    # C1: JFK -> BOS -> PHL -> JFK (short regional hops)
    # Dep 06:00, flights: 1.5h + 1h + 1.5h = 4h flight time
    # Duty: 06:00 to 12:00 = 6h
    add_flight("JFK", "BOS", 6, 0, 1.5, 60)    # F1: arr 07:30
    add_flight("BOS", "PHL", 8, 30, 1.0, 50)   # F2: arr 09:30 (1h connection)
    add_flight("PHL", "JFK", 10, 30, 1.5, 60)  # F3: arr 12:00 (1h connection)

    # C2: JFK -> ORD -> DCA -> JFK (midwest + DC)
    # Dep 07:00, flights: 2h + 1.5h + 1h = 4.5h flight time
    # Duty: 07:00 to 14:30 = 7.5h
    add_flight("JFK", "ORD", 7, 0, 2.0, 80)    # F4: arr 09:00
    add_flight("ORD", "DCA", 10, 0, 1.5, 70)   # F5: arr 11:30 (1h connection)
    add_flight("DCA", "JFK", 12, 30, 1.0, 55)  # F6: arr 13:30 (1h connection)

    # === LAX Hub (2 crew, 6 flights) ===

    # C3: LAX -> PHX -> SAN -> LAX (southwest regional)
    # Dep 06:00, flights: 1h + 1h + 0.5h = 2.5h flight time
    # Duty: 06:00 to 10:30 = 4.5h
    add_flight("LAX", "PHX", 6, 0, 1.0, 45)    # F7: arr 07:00
    add_flight("PHX", "SAN", 8, 0, 1.0, 45)    # F8: arr 09:00 (1h connection)
    add_flight("SAN", "LAX", 10, 0, 0.5, 30)   # F9: arr 10:30 (1h connection)

    # C4: LAX -> SFO -> SEA -> LAX (west coast)
    # Dep 08:00, flights: 1.5h + 2h + 2.5h = 6h flight time
    # Duty: 08:00 to 17:30 = 9.5h (tight but ok)
    add_flight("LAX", "SFO", 8, 0, 1.5, 50)    # F10: arr 09:30
    add_flight("SFO", "SEA", 10, 30, 2.0, 75)  # F11: arr 12:30 (1h connection)
    add_flight("SEA", "LAX", 14, 0, 2.5, 90)   # F12: arr 16:30 (1.5h connection)

    # === ORD Hub (2 crew, 6 flights) ===

    # C5: ORD -> DTW -> CLE -> ORD (great lakes)
    # Dep 06:30, flights: 1h + 0.75h + 1h = 2.75h flight time
    # Duty: 06:30 to 11:15 = 4.75h
    add_flight("ORD", "DTW", 6, 30, 1.0, 50)   # F13: arr 07:30
    add_flight("DTW", "CLE", 8, 30, 0.75, 40)  # F14: arr 09:15 (1h connection)
    add_flight("CLE", "ORD", 10, 15, 1.0, 50)  # F15: arr 11:15 (1h connection)

    # C6: ORD -> MSP -> DEN -> ORD (midwest + mountain)
    # Dep 07:00, flights: 1.5h + 2h + 2h = 5.5h flight time
    # Duty: 07:00 to 16:00 = 9h
    add_flight("ORD", "MSP", 7, 0, 1.5, 55)    # F16: arr 08:30
    add_flight("MSP", "DEN", 9, 30, 2.0, 75)   # F17: arr 11:30 (1h connection)
    add_flight("DEN", "ORD", 13, 0, 2.0, 75)   # F18: arr 15:00 (1.5h connection)

    # === SFO Hub (2 crew, 6 flights) ===

    # C7: SFO -> PDX -> SJC -> SFO (pacific northwest + bay area)
    # Dep 06:00, flights: 1.75h + 1.5h + 0.5h = 3.75h flight time
    # Duty: 06:00 to 12:15 = 6.25h
    add_flight("SFO", "PDX", 6, 0, 1.75, 60)   # F19: arr 07:45
    add_flight("PDX", "SJC", 8, 45, 1.5, 55)   # F20: arr 10:15 (1h connection)
    add_flight("SJC", "SFO", 11, 15, 0.5, 25)  # F21: arr 11:45 (1h connection)

    # C8: SFO -> LAS -> OAK -> SFO (vegas + oakland)
    # Dep 09:00, flights: 1.5h + 1.25h + 0.5h = 3.25h flight time
    # Duty: 09:00 to 15:15 = 6.25h
    add_flight("SFO", "LAS", 9, 0, 1.5, 55)    # F22: arr 10:30
    add_flight("LAS", "OAK", 11, 30, 1.25, 50) # F23: arr 12:45 (1h connection)
    add_flight("OAK", "SFO", 13, 45, 0.5, 25)  # F24: arr 14:15 (1h connection)

    # Generate crew - 2 at each hub
    crew = [
        # JFK crew
        Crew(
            id="C1",
            name="JFK Captain 1",
            base="JFK",
            rank=CrewRank.CAPTAIN,
            hourly_cost=60.0,
            qualifications={Qualification.NARROW_BODY},
            max_duty_hours=10.0,
            min_rest_hours=10.0,
            seniority=3
        ),
        Crew(
            id="C2",
            name="JFK Captain 2",
            base="JFK",
            rank=CrewRank.CAPTAIN,
            hourly_cost=55.0,
            qualifications={Qualification.NARROW_BODY},
            max_duty_hours=10.0,
            min_rest_hours=10.0,
            seniority=2
        ),
        # LAX crew
        Crew(
            id="C3",
            name="LAX Captain 1",
            base="LAX",
            rank=CrewRank.CAPTAIN,
            hourly_cost=58.0,
            qualifications={Qualification.NARROW_BODY},
            max_duty_hours=10.0,
            min_rest_hours=10.0,
            seniority=2
        ),
        Crew(
            id="C4",
            name="LAX Captain 2",
            base="LAX",
            rank=CrewRank.CAPTAIN,
            hourly_cost=52.0,
            qualifications={Qualification.NARROW_BODY},
            max_duty_hours=10.0,
            min_rest_hours=10.0,
            seniority=1
        ),
        # ORD crew
        Crew(
            id="C5",
            name="ORD Captain 1",
            base="ORD",
            rank=CrewRank.CAPTAIN,
            hourly_cost=56.0,
            qualifications={Qualification.NARROW_BODY},
            max_duty_hours=10.0,
            min_rest_hours=10.0,
            seniority=2
        ),
        Crew(
            id="C6",
            name="ORD Captain 2",
            base="ORD",
            rank=CrewRank.CAPTAIN,
            hourly_cost=50.0,
            qualifications={Qualification.NARROW_BODY},
            max_duty_hours=10.0,
            min_rest_hours=10.0,
            seniority=1
        ),
        # SFO crew
        Crew(
            id="C7",
            name="SFO Captain 1",
            base="SFO",
            rank=CrewRank.CAPTAIN,
            hourly_cost=57.0,
            qualifications={Qualification.NARROW_BODY},
            max_duty_hours=10.0,
            min_rest_hours=10.0,
            seniority=2
        ),
        Crew(
            id="C8",
            name="SFO Captain 2",
            base="SFO",
            rank=CrewRank.CAPTAIN,
            hourly_cost=51.0,
            qualifications={Qualification.NARROW_BODY},
            max_duty_hours=10.0,
            min_rest_hours=10.0,
            seniority=1
        ),
    ]

    # Legal rules
    rules = LegalRules(
        min_connection_time=timedelta(minutes=45),
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
    print("\n" + "=" * 70)
    print("              SMALL-AIRLINE TEST INSTANCE")
    print("=" * 70)

    print("\nFLIGHTS:")
    print("-" * 70)
    print(f"{'ID':<4} {'From':<4} {'To':<4} {'Departure':<16} {'Arrival':<16} {'Dur':>5} {'Cost':>6}")
    print("-" * 70)
    for f in sorted(flights, key=lambda x: (x.departure, x.id)):
        duration = (f.arrival - f.departure).total_seconds() / 3600
        print(
            f"{f.id:<4} {f.origin:<4} {f.destination:<4} "
            f"{f.departure.strftime('%m/%d %H:%M'):<16} "
            f"{f.arrival.strftime('%m/%d %H:%M'):<16} "
            f"{duration:>5.1f} {f.base_cost:>6.0f}"
        )

    print(f"\nTotal flights: {len(flights)}")

    # Count by origin
    origins = {}
    for f in flights:
        origins[f.origin] = origins.get(f.origin, 0) + 1
    print("Flights by origin:", dict(sorted(origins.items())))

    print("\nCREW:")
    print("-" * 70)
    print(f"{'ID':<4} {'Name':<16} {'Base':<4} {'Hourly Cost':>12} {'Seniority':>10}")
    print("-" * 70)
    for c in crew:
        print(
            f"{c.id:<4} {c.name:<16} {c.base:<4} "
            f"${c.hourly_cost:>11.0f} {c.seniority:>10}"
        )

    print(f"\nTotal crew: {len(crew)}")

    # Count by base
    bases = {}
    for c in crew:
        bases[c.base] = bases.get(c.base, 0) + 1
    print("Crew by base:", dict(sorted(bases.items())))

    print("\nRULES:")
    print("-" * 70)
    print(f"  Min Connection Time: {rules.min_connection_hours:.2f} hours")
    print(f"  Max Connection Time: {rules.max_connection_hours:.1f} hours")
    print(f"  Max Duty Period:     {rules.max_duty_hours:.0f} hours")
    print(f"  Max Flight Time:     {rules.max_flight_hours:.0f} hours")
    print(f"  Max Flights/Duty:    {rules.max_flights_per_duty}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    flights, crew, rules = generate_small_airline()
    print_instance_summary(flights, crew, rules)
