#!/usr/bin/env python3
"""Debug script to check what pairings can be generated."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.generators.micro_airline import generate_micro_airline
from models import FlightNetwork, Pairing
from optimization.subproblem.exact_rcspp import ExactRCSPP, Label
from heapq import heappush, heappop

def trace_rcspp(network, flights, crew, rules):
    """Trace RCSPP step by step to find the bug."""
    print("\n" + "=" * 60)
    print(f"TRACING RCSPP FOR CREW {crew.id} (base: {crew.base})")
    print("=" * 60)

    flight_duals = {f.id: 0.0 for f in flights}

    # Initialize
    labels_at_node = {node_id: [] for node_id in network.nodes}
    heap = []

    # Create initial label at source
    initial_label = Label(
        cost=0.0,
        node="SOURCE",
        path=("SOURCE",),
        flight_time=0.0,
        start_timestamp=0.0,
        end_timestamp=0.0
    )
    heappush(heap, initial_label)
    labels_at_node["SOURCE"].append(initial_label)

    print(f"\nInitial label: {initial_label}")
    print(f"SOURCE successors: {network.get_successors('SOURCE')}")

    step = 0
    max_steps = 50  # Limit for debugging

    while heap and step < max_steps:
        step += 1
        current = heappop(heap)

        print(f"\n--- Step {step} ---")
        print(f"Processing label at node: {current.node}")
        print(f"  Path: {' -> '.join(current.path)}")
        print(f"  Cost: {current.cost:.2f}")
        print(f"  Flight time: {current.flight_time:.2f}h")
        print(f"  Start timestamp: {current.start_timestamp}")
        print(f"  End timestamp: {current.end_timestamp}")
        print(f"  Duty time: {current.duty_time:.2f}h")

        if current.node == "SINK":
            print(f"  >>> REACHED SINK! <<<")
            return current

        # Check if dominated
        is_dominated = False
        for other in labels_at_node[current.node]:
            if other.dominates(current):
                print(f"  DOMINATED by existing label")
                is_dominated = True
                break

        if is_dominated:
            continue

        # Get successors
        successors = network.get_successors(current.node)
        print(f"  Successors: {successors}")

        for neighbor in successors:
            arc = network.get_arc(current.node, neighbor)
            if arc is None:
                print(f"    {neighbor}: No arc found!")
                continue

            print(f"    Extending to {neighbor} via {arc.arc_type} arc...")

            # Get arc cost
            arc_cost = network.get_arc_cost(current.node, neighbor, flight_duals)

            # Get node info
            to_node_obj = network.nodes[neighbor]
            from_node_obj = network.nodes[current.node]

            print(f"      Arc cost: {arc_cost:.2f}")
            print(f"      To node type: {to_node_obj.node_type}, airport: {to_node_obj.airport}")
            if to_node_obj.time:
                print(f"      To node time: {to_node_obj.time}")

            # Compute new resources
            new_flight_time = current.flight_time
            if arc.arc_type == "flight":
                flight_hours = arc.duration.total_seconds() / 3600
                new_flight_time += flight_hours
                print(f"      Adding {flight_hours:.2f}h flight time -> {new_flight_time:.2f}h total")

            # Check flight time limit
            if new_flight_time > rules.max_flight_hours:
                print(f"      REJECTED: Flight time {new_flight_time:.2f}h exceeds max {rules.max_flight_hours}h")
                continue

            # Update timestamps
            new_start_timestamp = current.start_timestamp
            new_end_timestamp = current.end_timestamp

            if to_node_obj.time is not None:
                new_end_timestamp = to_node_obj.time.timestamp()
                if new_start_timestamp == 0 and to_node_obj.node_type == "departure":
                    new_start_timestamp = to_node_obj.time.timestamp()
                    print(f"      Setting start timestamp: {new_start_timestamp}")

            print(f"      New timestamps: start={new_start_timestamp}, end={new_end_timestamp}")

            # Check duty time limit
            if new_start_timestamp > 0 and new_end_timestamp > 0:
                duty_hours = (new_end_timestamp - new_start_timestamp) / 3600
                print(f"      Duty time: {duty_hours:.2f}h (max: {crew.max_duty_hours}h)")
                if duty_hours > crew.max_duty_hours:
                    print(f"      REJECTED: Duty time exceeds limit")
                    continue

            # Create new label
            new_label = Label(
                cost=current.cost + arc_cost,
                node=neighbor,
                path=current.path + (neighbor,),
                flight_time=new_flight_time,
                start_timestamp=new_start_timestamp,
                end_timestamp=new_end_timestamp
            )

            # Check dominance
            is_dominated = False
            for other in labels_at_node[neighbor]:
                if other.dominates(new_label):
                    print(f"      REJECTED: Dominated by existing label at {neighbor}")
                    is_dominated = True
                    break

            if is_dominated:
                continue

            print(f"      ACCEPTED: Adding label to {neighbor}")
            labels_at_node[neighbor].append(new_label)
            heappush(heap, new_label)

            # Remove dominated labels
            labels_at_node[neighbor] = [
                label for label in labels_at_node[neighbor]
                if not new_label.dominates(label)
            ]

    print(f"\nNo path to SINK found after {step} steps")
    return None


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

        # Check specific paths manually
        # Look for F5->F6 path for JFK crew
        if c.base == "JFK":
            print(f"  Checking F5->F6 path manually:")
            f5_dep = "F5_DEP"
            f5_arr = "F5_ARR"
            f6_dep = "F6_DEP"
            f6_arr = "F6_ARR"

            # Check arcs exist
            arc_f5 = network.get_arc(f5_dep, f5_arr)
            arc_conn = network.get_arc(f5_arr, f6_dep)
            arc_f6 = network.get_arc(f6_dep, f6_arr)
            arc_sink = network.get_arc(f6_arr, "SINK")
            arc_source = network.get_arc("SOURCE", f5_dep)

            print(f"    SOURCE->F5_DEP: {arc_source is not None}")
            print(f"    F5_DEP->F5_ARR: {arc_f5 is not None}")
            print(f"    F5_ARR->F6_DEP: {arc_conn is not None}")
            print(f"    F6_DEP->F6_ARR: {arc_f6 is not None}")
            print(f"    F6_ARR->SINK: {arc_sink is not None}")

            # Trace the RCSPP algorithm step by step
            trace_rcspp(network, flights, c, rules)
            break  # Only trace for first JFK crew

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
