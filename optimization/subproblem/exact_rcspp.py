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
    duty_time: float = field(compare=False)
    current_time: float = field(compare=False)  # In hours from start

    def dominates(self, other: 'Label') -> bool:
        """Check if this label dominates another at the same node."""
        if self.node != other.node:
            return False
        return (
            self.cost <= other.cost and
            self.duty_time <= other.duty_time and
            (self.cost < other.cost or self.duty_time < other.duty_time)
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
            duty_time=0.0,
            current_time=0.0
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

        # Update resources
        new_duty = label.duty_time
        duration_hours = arc.duration.total_seconds() / 3600

        if arc.arc_type == "flight":
            new_duty += duration_hours

        new_time = label.current_time + duration_hours

        # Check feasibility: duty time limit
        if new_duty > self.crew.max_duty_hours:
            return None

        # Check feasibility: max duty period constraint
        if new_time > self.rules.max_duty_hours:
            return None

        return Label(
            cost=label.cost + arc_cost,
            node=to_node,
            path=label.path + (to_node,),
            duty_time=new_duty,
            current_time=new_time
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
