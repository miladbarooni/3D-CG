#!/usr/bin/env python3
"""Debug script to check what pairings can be generated."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.generators.micro_airline import generate_micro_airline
from models import FlightNetwork, Pairing
from optimization.subproblem.exact_rcspp import ExactRCSPP

def main():
    flights, crew, rules = generate_micro_airline()

    print("=" * 60)
    print("DEBUGGING PAIRING GENERATION")
    print("=" * 60)

    print(f"\nRules:")
    print(f"  Min connection: {rules.min_connection_hours} hours")
    print(f"  Max connection: {rules.max_connection_hours} hours")
    print(f"  Max duty: {rules.max_duty_hours} hours")
    print(f"  Max flight time: {rules.max_flight_hours} hours")

    # Check connections between flights
    print("\n" + "=" * 60)
    print("VALID CONNECTIONS:")
    print("=" * 60)

    for f1 in flights:
        for f2 in flights:
            if f1.id != f2.id and f1.can_connect_to(f2, rules.min_connection_time):
                conn_time = (f2.departure - f1.arrival).total_seconds() / 3600
                if conn_time <= rules.max_connection_hours:
                    print(f"  {f1.id} ({f1.destination}) -> {f2.id} ({f2.origin}): {conn_time:.1f}h connection")

    # For each crew, check what pairings RCSPP finds
    print("\n" + "=" * 60)
    print("RCSPP RESULTS PER CREW:")
    print("=" * 60)

    for c in crew:
        print(f"\nCrew {c.id} (base: {c.base}):")

        # Build network
        network = FlightNetwork(flights, c, rules)
        print(f"  Network: {network.num_nodes} nodes, {network.num_arcs} arcs")

        # Show source connections
        source_succs = network.get_successors("SOURCE")
        print(f"  SOURCE connects to: {source_succs}")

        # Show sink connections
        sink_preds = [arc[0] for arc in network.arcs if arc[1] == "SINK"]
        print(f"  SINK connected from: {sink_preds}")

        # Run RCSPP
        subproblem = ExactRCSPP(flights, c, rules)
        zero_duals = {f.id: 0.0 for f in flights}
        result = subproblem.solve(zero_duals, 0.0, time_limit_ms=10000)

        if result.pairing and result.pairing.flights:
            print(f"  Found pairing: {result.pairing}")
            print(f"    Flights: {[f.id for f in result.pairing.flights]}")
            print(f"    Cost: {result.reduced_cost:.2f}")
        else:
            print(f"  NO PAIRING FOUND!")
            print(f"    Explored {result.nodes_explored} nodes, created {result.labels_created} labels")

if __name__ == "__main__":
    main()
