"""Time-space network model for RCSPP."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import networkx as nx

from models.flight import Flight
from models.crew import Crew
from models.legal_rules import LegalRules


@dataclass
class NetworkNode:
    """Node in the time-space network."""
    id: str
    node_type: str  # 'source', 'sink', 'departure', 'arrival'
    airport: str
    time: Optional[datetime]
    flight_id: Optional[str] = None


@dataclass
class NetworkArc:
    """Arc in the time-space network."""
    from_node: str
    to_node: str
    arc_type: str  # 'flight', 'connection', 'source', 'sink', 'wait'
    base_cost: float
    flight_id: Optional[str] = None
    duration: timedelta = timedelta(0)


class FlightNetwork:
    """
    Time-space network for RCSPP.

    Nodes represent events (departures, arrivals) at specific times.
    Arcs represent flights, connections, and waiting.

    This network is crew-specific because:
    - Source/sink are at the crew's home base
    - Only feasible connections for this crew are included
    """

    def __init__(
        self,
        flights: List[Flight],
        crew: Crew,
        rules: LegalRules
    ):
        self.flights = flights
        self.crew = crew
        self.rules = rules

        self.graph = nx.DiGraph()
        self.nodes: Dict[str, NetworkNode] = {}
        self.arcs: Dict[Tuple[str, str], NetworkArc] = {}

        self._build_network()

    def _build_network(self) -> None:
        """Construct the time-space network."""
        if not self.flights:
            return

        # Determine time horizon
        min_time = min(f.departure for f in self.flights)
        max_time = max(f.arrival for f in self.flights)

        # Add source node (at crew's base, start of horizon)
        source = NetworkNode(
            id="SOURCE",
            node_type="source",
            airport=self.crew.base,
            time=min_time - timedelta(hours=1)
        )
        self.nodes["SOURCE"] = source
        self.graph.add_node("SOURCE", **self._node_features(source))

        # Add sink node (at crew's base, end of horizon)
        sink = NetworkNode(
            id="SINK",
            node_type="sink",
            airport=self.crew.base,
            time=max_time + timedelta(hours=1)
        )
        self.nodes["SINK"] = sink
        self.graph.add_node("SINK", **self._node_features(sink))

        # Add flight nodes and arcs
        for flight in self.flights:
            self._add_flight(flight)

        # Add connection arcs between compatible flights
        self._add_connections()

        # Add source/sink arcs
        self._add_source_sink_arcs()

    def _add_flight(self, flight: Flight) -> None:
        """Add departure and arrival nodes and flight arc."""
        # Departure node
        dep_id = f"{flight.id}_DEP"
        dep_node = NetworkNode(
            id=dep_id,
            node_type="departure",
            airport=flight.origin,
            time=flight.departure,
            flight_id=flight.id
        )
        self.nodes[dep_id] = dep_node
        self.graph.add_node(dep_id, **self._node_features(dep_node))

        # Arrival node
        arr_id = f"{flight.id}_ARR"
        arr_node = NetworkNode(
            id=arr_id,
            node_type="arrival",
            airport=flight.destination,
            time=flight.arrival,
            flight_id=flight.id
        )
        self.nodes[arr_id] = arr_node
        self.graph.add_node(arr_id, **self._node_features(arr_node))

        # Flight arc
        arc = NetworkArc(
            from_node=dep_id,
            to_node=arr_id,
            arc_type="flight",
            base_cost=flight.base_cost,
            flight_id=flight.id,
            duration=flight.duration
        )
        self.arcs[(dep_id, arr_id)] = arc
        self.graph.add_edge(dep_id, arr_id, **self._arc_features(arc))

    def _add_connections(self) -> None:
        """Add connection arcs between compatible flights."""
        for f1 in self.flights:
            for f2 in self.flights:
                if f1.id == f2.id:
                    continue

                if f1.can_connect_to(f2, self.rules.min_connection_time):
                    # Connection time check
                    conn_time = f2.departure - f1.arrival
                    if conn_time <= self.rules.max_connection_time:
                        arr_id = f"{f1.id}_ARR"
                        dep_id = f"{f2.id}_DEP"

                        arc = NetworkArc(
                            from_node=arr_id,
                            to_node=dep_id,
                            arc_type="connection",
                            base_cost=0,  # Connection has no direct cost
                            duration=conn_time
                        )
                        self.arcs[(arr_id, dep_id)] = arc
                        self.graph.add_edge(
                            arr_id, dep_id,
                            **self._arc_features(arc)
                        )

    def _add_source_sink_arcs(self) -> None:
        """Add arcs from source and to sink."""
        for flight in self.flights:
            # Source to departures at crew's base
            if flight.origin == self.crew.base:
                dep_id = f"{flight.id}_DEP"
                arc = NetworkArc(
                    from_node="SOURCE",
                    to_node=dep_id,
                    arc_type="source",
                    base_cost=0
                )
                self.arcs[("SOURCE", dep_id)] = arc
                self.graph.add_edge("SOURCE", dep_id, **self._arc_features(arc))

            # Arrivals at crew's base to sink
            if flight.destination == self.crew.base:
                arr_id = f"{flight.id}_ARR"
                arc = NetworkArc(
                    from_node=arr_id,
                    to_node="SINK",
                    arc_type="sink",
                    base_cost=0
                )
                self.arcs[(arr_id, "SINK")] = arc
                self.graph.add_edge(arr_id, "SINK", **self._arc_features(arc))

    def _node_features(self, node: NetworkNode) -> dict:
        """Extract features for a node (for GNN)."""
        return {
            "node_type": node.node_type,
            "airport": node.airport,
            "time": node.time.timestamp() if node.time else 0,
            "flight_id": node.flight_id,
            "is_crew_base": node.airport == self.crew.base
        }

    def _arc_features(self, arc: NetworkArc) -> dict:
        """Extract features for an arc (for GNN)."""
        return {
            "arc_type": arc.arc_type,
            "base_cost": arc.base_cost,
            "flight_id": arc.flight_id,
            "duration_hours": arc.duration.total_seconds() / 3600
        }

    def get_arc_cost(
        self,
        from_node: str,
        to_node: str,
        dual_prices: Dict[str, float]
    ) -> float:
        """
        Get reduced cost for an arc given dual prices.

        For flight arcs: base_cost - pi_i
        For other arcs: 0
        """
        arc = self.arcs.get((from_node, to_node))
        if not arc:
            return float('inf')

        if arc.arc_type == "flight" and arc.flight_id:
            pi = dual_prices.get(arc.flight_id, 0)
            return arc.base_cost - pi

        return arc.base_cost

    def get_successors(self, node_id: str) -> List[str]:
        """Get list of successor node IDs."""
        return list(self.graph.successors(node_id))

    def get_arc(self, from_node: str, to_node: str) -> Optional[NetworkArc]:
        """Get arc between two nodes."""
        return self.arcs.get((from_node, to_node))

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the network."""
        return len(self.nodes)

    @property
    def num_arcs(self) -> int:
        """Number of arcs in the network."""
        return len(self.arcs)

    def __repr__(self) -> str:
        return (
            f"FlightNetwork(crew={self.crew.id}, "
            f"nodes={self.num_nodes}, arcs={self.num_arcs})"
        )
