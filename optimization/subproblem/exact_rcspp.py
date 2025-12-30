"""Exact Resource-Constrained Shortest Path Problem solver."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from heapq import heappush, heappop
import time

from models import Flight, Crew, Pairing, LegalRules
from optimization.subproblem.base import PricingSubproblem, SubproblemResult


@dataclass(order=True)
class Label:
    """
    Label for dynamic programming in RCSPP.

    A label represents a partial path through the network
    with accumulated resources.
    """
    cost: float  # For heap ordering
    node: str = field(compare=False)
    path: Tuple[str, ...] = field(compare=False)
    flight_time: float = field(compare=False)  # Total flight hours
    start_timestamp: float = field(compare=False)  # Timestamp of first departure
    end_timestamp: float = field(compare=False)  # Timestamp of current position

    @property
    def duty_time(self) -> float:
        """Total duty time (wall clock from start to current)."""
        if self.start_timestamp == 0:
            return 0.0
        return (self.end_timestamp - self.start_timestamp) / 3600  # Convert to hours

    def dominates(self, other: 'Label') -> bool:
        """Check if this label dominates another at the same node."""
        if self.node != other.node:
            return False
        return (
            self.cost <= other.cost and
            self.flight_time <= other.flight_time and
            self.end_timestamp <= other.end_timestamp and
            (self.cost < other.cost or self.flight_time < other.flight_time or
             self.end_timestamp < other.end_timestamp)
        )


class ExactRCSPP(PricingSubproblem):
    """
    Exact Resource-Constrained Shortest Path solver.

    Uses a label-setting algorithm with dominance rules
    to find the minimum reduced cost path.
    """

    def solve(
        self,
        flight_duals: Dict[str, float],
        crew_dual: float,
        time_limit_ms: int = 10000
    ) -> SubproblemResult:
        """
        Solve RCSPP using label-setting algorithm.
        """
        start_time = time.time()

        # Initialize
        labels_at_node: Dict[str, List[Label]] = {
            node_id: [] for node_id in self.network.nodes
        }

        # Priority queue of labels to explore
        heap: List[Label] = []

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

        # Best label reaching sink
        best_sink_label: Optional[Label] = None

        # Statistics
        nodes_explored = 0
        labels_created = 1

        # Main loop
        while heap:
            # Check time limit
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > time_limit_ms:
                break

            # Get next label
            current = heappop(heap)
            nodes_explored += 1

            # Skip if dominated by existing labels at this node
            if self._is_dominated(current, labels_at_node[current.node]):
                continue

            # Check if reached sink
            if current.node == "SINK":
                if best_sink_label is None or current.cost < best_sink_label.cost:
                    best_sink_label = current
                continue

            # Extend label along all outgoing arcs
            for neighbor in self.network.get_successors(current.node):
                new_label = self._extend_label(
                    current,
                    neighbor,
                    flight_duals
                )

                if new_label is None:
                    continue  # Extension not feasible

                # Check dominance
                if self._is_dominated(new_label, labels_at_node[neighbor]):
                    continue

                # Add new label
                labels_at_node[neighbor].append(new_label)
                heappush(heap, new_label)
                labels_created += 1

                # Remove dominated labels
                labels_at_node[neighbor] = [
                    label for label in labels_at_node[neighbor]
                    if not new_label.dominates(label)
                ]

        solve_time_ms = (time.time() - start_time) * 1000

        # Build result
        if best_sink_label is None:
            return SubproblemResult(
                pairing=None,
                reduced_cost=float('inf'),
                solve_time_ms=solve_time_ms,
                nodes_explored=nodes_explored,
                labels_created=labels_created
            )

        # Convert path to pairing
        pairing = self._path_to_pairing(best_sink_label.path)
        reduced_cost = best_sink_label.cost - crew_dual

        return SubproblemResult(
            pairing=pairing,
            reduced_cost=reduced_cost,
            solve_time_ms=solve_time_ms,
            nodes_explored=nodes_explored,
            labels_created=labels_created
        )

    def _extend_label(
        self,
        label: Label,
        to_node: str,
        flight_duals: Dict[str, float]
    ) -> Optional[Label]:
        """
        Try to extend label to a new node.

        Returns None if extension is not feasible.
        """
        arc = self.network.get_arc(label.node, to_node)

        if arc is None:
            return None

        # Compute arc cost (with reduced cost for flight arcs)
        arc_cost = self.network.get_arc_cost(label.node, to_node, flight_duals)

        # Get timestamps from the network nodes
        to_node_obj = self.network.nodes[to_node]
        from_node_obj = self.network.nodes[label.node]

        # Update flight time
        new_flight_time = label.flight_time
        if arc.arc_type == "flight":
            new_flight_time += arc.duration.total_seconds() / 3600

        # Update timestamps
        new_start_timestamp = label.start_timestamp
        new_end_timestamp = label.end_timestamp

        # Don't update timestamps for SOURCE/SINK - they have artificial times
        if to_node_obj.time is not None and to_node_obj.node_type not in ("source", "sink"):
            new_end_timestamp = to_node_obj.time.timestamp()
            # Set start time when we hit first departure
            if new_start_timestamp == 0 and to_node_obj.node_type == "departure":
                new_start_timestamp = to_node_obj.time.timestamp()

        # Check feasibility: flight time limit
        if new_flight_time > self.rules.max_flight_hours:
            return None

        # Check feasibility: duty period (wall clock time from start to end)
        if new_start_timestamp > 0 and new_end_timestamp > 0:
            duty_hours = (new_end_timestamp - new_start_timestamp) / 3600
            if duty_hours > self.crew.max_duty_hours:
                return None

        return Label(
            cost=label.cost + arc_cost,
            node=to_node,
            path=label.path + (to_node,),
            flight_time=new_flight_time,
            start_timestamp=new_start_timestamp,
            end_timestamp=new_end_timestamp
        )

    def _is_dominated(
        self,
        label: Label,
        existing: List[Label]
    ) -> bool:
        """Check if label is dominated by any existing label."""
        for other in existing:
            if other.dominates(label):
                return True
        return False

    def _path_to_pairing(self, path: Tuple[str, ...]) -> Pairing:
        """Convert a path through the network to a Pairing object."""
        flight_ids = []

        for node_id in path:
            if node_id.endswith("_DEP"):
                flight_id = node_id[:-4]  # Remove "_DEP"
                flight_ids.append(flight_id)

        # Get flights in order
        flight_dict = {f.id: f for f in self.flights}
        flights = [flight_dict[fid] for fid in flight_ids if fid in flight_dict]

        # Sort by departure time
        flights.sort(key=lambda f: f.departure)

        return Pairing.create(flights, self.crew)
