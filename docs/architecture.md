# Architecture Document: 3D Integrated Crew Scheduling System

## Table of Contents

1. [System Overview](#system-overview)
2. [Directory Structure](#directory-structure)
3. [Core Data Models](#core-data-models)
4. [Optimization Engine](#optimization-engine)
5. [Graph Neural Network Module](#graph-neural-network-module)
6. [Integration Layer](#integration-layer)
7. [Data Pipeline](#data-pipeline)
8. [API Specifications](#api-specifications)
9. [Configuration Management](#configuration-management)
10. [Testing Strategy](#testing-strategy)

---

## 1. System Overview

### 1.1 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   CLI App   │  │  REST API   │  │  Dashboard  │  │  Notebooks  │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
└─────────┼────────────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │                │
          └────────────────┴────────────────┴────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATION LAYER                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                     ColumnGenerationOrchestrator                        │  │
│  │  • Manages iteration loop                                               │  │
│  │  • Coordinates master/subproblems                                       │  │
│  │  • Handles convergence detection                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           OPTIMIZATION LAYER                                  │
│                                                                              │
│  ┌──────────────────────┐    ┌──────────────────────────────────────────┐   │
│  │    MASTER PROBLEM    │    │           PRICING SUBPROBLEM             │   │
│  │                      │    │                                          │   │
│  │  • RMP Formulation   │◄──►│  ┌────────────┐    ┌─────────────────┐  │   │
│  │  • Dual Extraction   │    │  │   Exact    │    │   GNN-Guided    │  │   │
│  │  • Column Pool Mgmt  │    │  │   RCSPP    │    │     RCSPP       │  │   │
│  │  • Integer Rounding  │    │  │  (Label)   │    │   (Hybrid)      │  │   │
│  └──────────────────────┘    │  └────────────┘    └─────────────────┘  │   │
│                              │          │                 │            │   │
│                              │          └────────┬────────┘            │   │
│                              │                   ▼                     │   │
│                              │         ┌─────────────────┐             │   │
│                              │         │  FlightNetwork  │             │   │
│                              │         │  (Time-Space)   │             │   │
│                              │         └─────────────────┘             │   │
│                              └──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           ML LAYER (Optional)                                 │
│  ┌──────────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │    GNN Encoder       │  │   Training Loop   │  │   Inference Engine   │   │
│  │  (GAT/GCN/MPNN)      │  │  (Supervised/RL)  │  │   (TorchScript)      │   │
│  └──────────────────────┘  └──────────────────┘  └──────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │  Flight Data     │  │  Crew Data       │  │  Regulatory Rules        │   │
│  │  (Schedule DB)   │  │  (HR System)     │  │  (Configurable)          │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Interaction Flow

```
┌─────────┐      ┌─────────┐      ┌──────────┐      ┌──────────┐
│  Input  │─────►│ Parser  │─────►│  Solver  │─────►│  Output  │
└─────────┘      └─────────┘      └──────────┘      └──────────┘
                                        │
                         ┌──────────────┼──────────────┐
                         │              │              │
                         ▼              ▼              ▼
                   ┌──────────┐  ┌──────────┐  ┌──────────┐
                   │  Master  │  │  Pricing │  │   GNN    │
                   │  Problem │  │  RCSPP   │  │  Module  │
                   └──────────┘  └──────────┘  └──────────┘
```

---

## 2. Directory Structure

```
crew_scheduling_3d/
│
├── README.md                       # Project overview and quick start
├── initial_prompt.md               # Full problem specification
├── architecture.md                 # This document
├── requirements.txt                # Python dependencies
├── requirements-dev.txt            # Development dependencies
├── pyproject.toml                  # Project configuration
├── setup.py                        # Package installation
│
├── config/
│   ├── __init__.py
│   ├── default.yaml                # Default configuration
│   ├── micro_airline.yaml          # Test instance config
│   └── production.yaml             # Production settings
│
├── data/
│   ├── __init__.py
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── micro_airline.py        # Micro-airline dataset generator
│   │   ├── random_instance.py      # Random instance generator
│   │   └── real_schedule.py        # Real schedule parser
│   ├── datasets/
│   │   ├── micro_airline.json      # Serialized micro-airline
│   │   └── training/               # GNN training data
│   │       ├── instances/          # Problem instances
│   │       └── solutions/          # Optimal solutions
│   └── loaders.py                  # Data loading utilities
│
├── models/
│   ├── __init__.py
│   ├── base.py                     # Base model classes
│   ├── flight.py                   # Flight data model
│   ├── crew.py                     # Crew data model
│   ├── pairing.py                  # Pairing data model
│   ├── solution.py                 # Solution data model
│   ├── legal_rules.py              # Legal constraints model
│   └── network.py                  # Time-space network graph
│
├── optimization/
│   ├── __init__.py
│   ├── master_problem.py           # Restricted Master Problem
│   ├── subproblem/
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract subproblem interface
│   │   ├── exact_rcspp.py          # Exact label-setting algorithm
│   │   ├── gnn_guided.py           # GNN-guided RCSPP
│   │   └── gnn_direct.py           # Direct GNN prediction
│   ├── column_generation.py        # Main CG orchestrator
│   ├── integer_solution.py         # Rounding and MIP solving
│   └── utils/
│       ├── __init__.py
│       ├── dual_stabilization.py   # Dual smoothing techniques
│       └── column_pool.py          # Column management
│
├── gnn/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_gnn.py             # Abstract GNN interface
│   │   ├── gat_pricing.py          # Graph Attention Network
│   │   ├── gcn_pricing.py          # Graph Convolutional Network
│   │   └── mpnn_pricing.py         # Message Passing NN
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── attention.py            # Custom attention layers
│   │   └── aggregation.py          # Aggregation functions
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # PyG Dataset class
│   │   ├── transforms.py           # Graph transformations
│   │   └── collate.py              # Batch collation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training loop
│   │   ├── losses.py               # Loss functions
│   │   └── metrics.py              # Evaluation metrics
│   └── inference/
│       ├── __init__.py
│       └── predictor.py            # Inference wrapper
│
├── visualization/
│   ├── __init__.py
│   ├── network_viz.py              # Flight network visualization
│   ├── solution_viz.py             # Solution timeline
│   ├── convergence_plot.py         # CG convergence plots
│   └── dashboard.py                # Interactive dashboard
│
├── api/
│   ├── __init__.py
│   ├── rest/
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI application
│   │   ├── routes.py               # API endpoints
│   │   └── schemas.py              # Request/response models
│   └── cli/
│       ├── __init__.py
│       └── main.py                 # CLI application
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_master_problem.py
│   │   ├── test_rcspp.py
│   │   └── test_gnn.py
│   ├── integration/
│   │   ├── test_column_generation.py
│   │   └── test_end_to_end.py
│   └── benchmarks/
│       ├── benchmark_rcspp.py
│       └── benchmark_gnn.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_algorithm_demo.ipynb
│   ├── 03_gnn_training.ipynb
│   └── 04_results_analysis.ipynb
│
├── scripts/
│   ├── generate_data.py            # Data generation script
│   ├── train_gnn.py                # GNN training script
│   ├── evaluate.py                 # Evaluation script
│   └── benchmark.py                # Benchmarking script
│
└── main.py                         # Main entry point
```

---

## 3. Core Data Models

### 3.1 Flight Model

```python
# models/flight.py

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum

class AircraftType(Enum):
    NARROW_BODY = "narrow"
    WIDE_BODY = "wide"
    REGIONAL = "regional"

@dataclass(frozen=True)
class Airport:
    """Airport representation."""
    code: str           # IATA code (e.g., "JFK")
    name: str           # Full name
    timezone: str       # Timezone identifier
    
    def __hash__(self):
        return hash(self.code)
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.code == other
        return self.code == other.code

@dataclass
class Flight:
    """
    Represents a single flight leg.
    
    Attributes:
        id: Unique flight identifier
        flight_number: Airline flight number (e.g., "AA123")
        origin: Departure airport
        destination: Arrival airport
        departure: Scheduled departure time (UTC)
        arrival: Scheduled arrival time (UTC)
        base_cost: Operating cost for this flight
        aircraft_type: Type of aircraft
        required_crew: Number of crew needed (for future extension)
    """
    id: str
    flight_number: str
    origin: str
    destination: str
    departure: datetime
    arrival: datetime
    base_cost: float
    aircraft_type: AircraftType = AircraftType.NARROW_BODY
    required_crew: int = 1
    
    # Computed properties
    @property
    def duration(self) -> timedelta:
        """Flight duration."""
        return self.arrival - self.departure
    
    @property
    def duration_hours(self) -> float:
        """Flight duration in hours."""
        return self.duration.total_seconds() / 3600
    
    def can_connect_to(
        self, 
        other: 'Flight', 
        min_connection: timedelta
    ) -> bool:
        """
        Check if this flight can legally connect to another flight.
        
        Args:
            other: The potential subsequent flight
            min_connection: Minimum required connection time
            
        Returns:
            True if connection is legal
        """
        # Must arrive at same airport as next departure
        if self.destination != other.origin:
            return False
        
        # Must have enough connection time
        connection_time = other.departure - self.arrival
        if connection_time < min_connection:
            return False
        
        return True
    
    def __hash__(self):
        return hash(self.id)
    
    def __repr__(self):
        return (
            f"Flight({self.id}: {self.origin}→{self.destination} "
            f"{self.departure.strftime('%m/%d %H:%M')}-"
            f"{self.arrival.strftime('%H:%M')})"
        )
```

### 3.2 Crew Model

```python
# models/crew.py

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Set
from enum import Enum

class CrewRank(Enum):
    CAPTAIN = "captain"
    FIRST_OFFICER = "first_officer"
    FLIGHT_ATTENDANT = "flight_attendant"

class Qualification(Enum):
    NARROW_BODY = "narrow"
    WIDE_BODY = "wide"
    REGIONAL = "regional"
    INTERNATIONAL = "international"

@dataclass
class Crew:
    """
    Represents a crew member.
    
    Attributes:
        id: Unique crew identifier
        name: Crew member name
        base: Home base airport code
        rank: Crew rank/position
        hourly_cost: Cost per hour of duty
        qualifications: Set of qualifications
        max_duty_hours: Maximum duty period length
        min_rest_hours: Minimum rest between duties
        availability_start: Start of availability window
        availability_end: End of availability window
        preferences: Optional preferences (for future use)
    """
    id: str
    name: str
    base: str
    rank: CrewRank
    hourly_cost: float
    qualifications: Set[Qualification] = field(default_factory=set)
    max_duty_hours: float = 10.0
    min_rest_hours: float = 10.0
    availability_start: Optional[datetime] = None
    availability_end: Optional[datetime] = None
    seniority: int = 0  # Higher = more senior
    
    def can_operate(self, flight: 'Flight') -> bool:
        """Check if crew can operate given flight based on qualifications."""
        required_qual = Qualification(flight.aircraft_type.value)
        return required_qual in self.qualifications
    
    def is_available(self, start: datetime, end: datetime) -> bool:
        """Check if crew is available during given time window."""
        if self.availability_start and start < self.availability_start:
            return False
        if self.availability_end and end > self.availability_end:
            return False
        return True
    
    def compute_pairing_cost(self, duty_hours: float) -> float:
        """Compute cost for a pairing with given duty hours."""
        return self.hourly_cost * duty_hours
    
    def __hash__(self):
        return hash(self.id)
    
    def __repr__(self):
        return f"Crew({self.id}: {self.name}, Base={self.base})"
```

### 3.3 Pairing Model

```python
# models/pairing.py

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, FrozenSet
import hashlib

from .flight import Flight
from .crew import Crew
from .legal_rules import LegalRules

@dataclass
class Pairing:
    """
    Represents a crew pairing (sequence of flights).
    
    In the 3D model, each pairing is associated with a specific crew member.
    This is the key difference from traditional 2D pairing models.
    
    Attributes:
        id: Unique pairing identifier
        flights: Ordered list of flights in this pairing
        crew_id: ID of the crew member this pairing is assigned to
        reduced_cost: Reduced cost at time of generation (for CG)
    """
    id: str
    flights: List[Flight]
    crew_id: str
    reduced_cost: Optional[float] = None
    
    @classmethod
    def create(cls, flights: List[Flight], crew: Crew) -> 'Pairing':
        """Factory method to create a pairing with auto-generated ID."""
        # Generate deterministic ID from flight sequence and crew
        flight_ids = "-".join(f.id for f in flights)
        hash_input = f"{crew.id}:{flight_ids}"
        pairing_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return cls(id=f"P_{pairing_id}", flights=flights, crew_id=crew.id)
    
    @property
    def flight_ids(self) -> FrozenSet[str]:
        """Set of flight IDs covered by this pairing."""
        return frozenset(f.id for f in self.flights)
    
    @property
    def start_base(self) -> str:
        """Starting airport of the pairing."""
        return self.flights[0].origin if self.flights else None
    
    @property
    def end_base(self) -> str:
        """Ending airport of the pairing."""
        return self.flights[-1].destination if self.flights else None
    
    @property
    def start_time(self) -> datetime:
        """Departure time of first flight."""
        return self.flights[0].departure if self.flights else None
    
    @property
    def end_time(self) -> datetime:
        """Arrival time of last flight."""
        return self.flights[-1].arrival if self.flights else None
    
    @property
    def total_duty_hours(self) -> float:
        """Total duty time from first departure to last arrival."""
        if not self.flights:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() / 3600
    
    @property
    def total_flight_hours(self) -> float:
        """Total time spent actually flying."""
        return sum(f.duration_hours for f in self.flights)
    
    @property
    def total_base_cost(self) -> float:
        """Sum of base costs for all flights."""
        return sum(f.base_cost for f in self.flights)
    
    def compute_cost(self, crew: Crew) -> float:
        """
        Compute total cost for this pairing assigned to given crew.
        
        Cost = base_flight_cost + crew_duty_cost
        """
        base_cost = self.total_base_cost
        duty_cost = crew.compute_pairing_cost(self.total_duty_hours)
        return base_cost + duty_cost
    
    def is_legal(self, crew: Crew, rules: LegalRules) -> bool:
        """
        Validate this pairing against crew constraints and legal rules.
        
        Checks:
        1. Starts and ends at crew's home base
        2. Duty time within limits
        3. All connections are legal
        4. Crew is qualified for all flights
        """
        # Check base constraint
        if self.start_base != crew.base or self.end_base != crew.base:
            return False
        
        # Check duty time
        if self.total_duty_hours > crew.max_duty_hours:
            return False
        
        # Check connections
        for i in range(len(self.flights) - 1):
            if not self.flights[i].can_connect_to(
                self.flights[i + 1], 
                rules.min_connection_time
            ):
                return False
        
        # Check qualifications
        for flight in self.flights:
            if not crew.can_operate(flight):
                return False
        
        return True
    
    def covers_flight(self, flight_id: str) -> bool:
        """Check if this pairing covers a specific flight."""
        return flight_id in self.flight_ids
    
    def get_coefficient(self, flight_id: str) -> int:
        """
        Get the A_ij coefficient for constraint matrix.
        Returns 1 if this pairing covers the flight, 0 otherwise.
        """
        return 1 if self.covers_flight(flight_id) else 0
    
    def __hash__(self):
        return hash((self.id, self.crew_id))
    
    def __repr__(self):
        flight_str = " → ".join(
            f"{f.origin}-{f.destination}" for f in self.flights
        )
        return f"Pairing({self.id} for {self.crew_id}: {flight_str})"
```

### 3.4 Legal Rules Model

```python
# models/legal_rules.py

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
        return self.min_connection_time.total_seconds() / 3600
    
    @property
    def max_duty_hours(self) -> float:
        return self.max_duty_period.total_seconds() / 3600
    
    @property
    def min_rest_hours(self) -> float:
        return self.min_rest_period.total_seconds() / 3600
```

### 3.5 Solution Model

```python
# models/solution.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

from .pairing import Pairing
from .flight import Flight
from .crew import Crew

@dataclass
class SolutionStatistics:
    """Statistics about a solution."""
    total_cost: float
    total_flight_coverage: int
    total_crew_assigned: int
    total_duty_hours: float
    average_duty_hours: float
    iterations: int
    solve_time_seconds: float
    columns_generated: int
    is_integer: bool
    optimality_gap: Optional[float] = None

@dataclass
class CrewAssignment:
    """Assignment of a crew member to a pairing."""
    crew: Crew
    pairing: Pairing
    cost: float
    
    @property
    def flights(self) -> List[Flight]:
        return self.pairing.flights
    
    @property
    def duty_hours(self) -> float:
        return self.pairing.total_duty_hours

@dataclass
class Solution:
    """
    Complete solution to the crew scheduling problem.
    """
    assignments: Dict[str, CrewAssignment]  # crew_id -> assignment
    statistics: SolutionStatistics
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Iteration history for analysis
    iteration_history: List[Dict] = field(default_factory=list)
    
    @property
    def total_cost(self) -> float:
        return self.statistics.total_cost
    
    @property
    def is_feasible(self) -> bool:
        """Check if solution is feasible."""
        return (
            self.statistics.total_flight_coverage == self.statistics.total_crew_assigned
            and self.statistics.is_integer
        )
    
    def get_crew_assignment(self, crew_id: str) -> Optional[CrewAssignment]:
        """Get assignment for a specific crew member."""
        return self.assignments.get(crew_id)
    
    def get_flight_coverage(self) -> Dict[str, str]:
        """Get mapping of flight_id -> crew_id covering it."""
        coverage = {}
        for crew_id, assignment in self.assignments.items():
            for flight in assignment.flights:
                coverage[flight.id] = crew_id
        return coverage
    
    def verify_constraints(
        self, 
        flights: List[Flight], 
        crew: List[Crew]
    ) -> Dict[str, bool]:
        """
        Verify all constraints are satisfied.
        
        Returns dict of constraint_name -> satisfied
        """
        coverage = self.get_flight_coverage()
        
        return {
            "all_flights_covered": all(f.id in coverage for f in flights),
            "each_flight_once": len(coverage) == len(flights),
            "all_crew_assigned": len(self.assignments) == len(crew),
            "base_constraints": all(
                a.pairing.start_base == a.crew.base and 
                a.pairing.end_base == a.crew.base
                for a in self.assignments.values()
            ),
            "duty_limits": all(
                a.duty_hours <= a.crew.max_duty_hours
                for a in self.assignments.values()
            )
        }
    
    def to_dict(self) -> dict:
        """Serialize solution to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_cost": self.total_cost,
            "assignments": {
                crew_id: {
                    "crew_id": assignment.crew.id,
                    "pairing_id": assignment.pairing.id,
                    "flights": [f.id for f in assignment.flights],
                    "cost": assignment.cost,
                    "duty_hours": assignment.duty_hours
                }
                for crew_id, assignment in self.assignments.items()
            },
            "statistics": {
                "total_cost": self.statistics.total_cost,
                "iterations": self.statistics.iterations,
                "solve_time": self.statistics.solve_time_seconds,
                "columns_generated": self.statistics.columns_generated
            }
        }
    
    def __repr__(self):
        return (
            f"Solution(cost={self.total_cost:.2f}, "
            f"crew={len(self.assignments)}, "
            f"feasible={self.is_feasible})"
        )
```

### 3.6 Network Model

```python
# models/network.py

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime, timedelta
import networkx as nx

from .flight import Flight
from .crew import Crew
from .legal_rules import LegalRules

@dataclass
class NetworkNode:
    """Node in the time-space network."""
    id: str
    node_type: str  # 'source', 'sink', 'departure', 'arrival'
    airport: str
    time: datetime
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
    
    def _build_network(self):
        """Construct the time-space network."""
        # Determine time horizon
        if not self.flights:
            return
            
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
    
    def _add_flight(self, flight: Flight):
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
    
    def _add_connections(self):
        """Add connection arcs between compatible flights."""
        flight_dict = {f.id: f for f in self.flights}
        
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
    
    def _add_source_sink_arcs(self):
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
        
        For flight arcs: base_cost - π_i
        For other arcs: 0
        """
        arc = self.arcs.get((from_node, to_node))
        if not arc:
            return float('inf')
        
        if arc.arc_type == "flight" and arc.flight_id:
            pi = dual_prices.get(arc.flight_id, 0)
            return arc.base_cost - pi
        
        return arc.base_cost
    
    def to_pyg_data(
        self, 
        dual_prices: Dict[str, float], 
        crew_dual: float
    ):
        """
        Convert to PyTorch Geometric Data object for GNN.
        
        Returns a PyG Data object with node features, edge index,
        and edge features suitable for GNN processing.
        """
        import torch
        from torch_geometric.data import Data
        
        # Node mapping
        node_list = list(self.nodes.keys())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        
        # Node features
        node_features = []
        for node_id in node_list:
            node = self.nodes[node_id]
            features = self._encode_node_features(node, dual_prices)
            node_features.append(features)
        
        # Edge index and features
        edge_index = [[], []]
        edge_features = []
        
        for (from_node, to_node), arc in self.arcs.items():
            edge_index[0].append(node_to_idx[from_node])
            edge_index[1].append(node_to_idx[to_node])
            
            reduced_cost = self.get_arc_cost(from_node, to_node, dual_prices)
            features = self._encode_arc_features(arc, reduced_cost)
            edge_features.append(features)
        
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            crew_dual=torch.tensor([crew_dual], dtype=torch.float),
            num_nodes=len(node_list)
        )
    
    def _encode_node_features(
        self, 
        node: NetworkNode, 
        dual_prices: Dict[str, float]
    ) -> List[float]:
        """Encode node features for GNN."""
        # One-hot for node type
        type_encoding = [0, 0, 0, 0]  # source, sink, dep, arr
        type_map = {"source": 0, "sink": 1, "departure": 2, "arrival": 3}
        type_encoding[type_map[node.node_type]] = 1
        
        # Dual price (if flight node)
        dual = dual_prices.get(node.flight_id, 0) if node.flight_id else 0
        
        # Is crew base
        is_base = 1.0 if node.airport == self.crew.base else 0.0
        
        return type_encoding + [dual, is_base]
    
    def _encode_arc_features(
        self, 
        arc: NetworkArc, 
        reduced_cost: float
    ) -> List[float]:
        """Encode arc features for GNN."""
        # One-hot for arc type
        type_encoding = [0, 0, 0, 0, 0]  # flight, conn, source, sink, wait
        type_map = {
            "flight": 0, "connection": 1, "source": 2, "sink": 3, "wait": 4
        }
        type_encoding[type_map.get(arc.arc_type, 0)] = 1
        
        return type_encoding + [
            arc.base_cost / 100,  # Normalized
            reduced_cost / 100,   # Normalized
            arc.duration.total_seconds() / 3600  # Hours
        ]
```

---

## 4. Optimization Engine

### 4.1 Master Problem

```python
# optimization/master_problem.py

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pulp

from models import Flight, Crew, Pairing, LegalRules

@dataclass
class DualValues:
    """Container for dual values from LP relaxation."""
    flight_duals: Dict[str, float]   # π_i for each flight
    crew_duals: Dict[str, float]     # σ_k for each crew
    
    def get_flight_dual(self, flight_id: str) -> float:
        return self.flight_duals.get(flight_id, 0.0)
    
    def get_crew_dual(self, crew_id: str) -> float:
        return self.crew_duals.get(crew_id, 0.0)

class MasterProblem:
    """
    Restricted Master Problem (RMP) for column generation.
    
    Manages:
    - LP/MIP formulation
    - Column (pairing) pool
    - Dual value extraction
    - Solution retrieval
    """
    
    def __init__(
        self, 
        flights: List[Flight], 
        crew: List[Crew],
        rules: LegalRules
    ):
        self.flights = {f.id: f for f in flights}
        self.crew = {c.id: c for c in crew}
        self.rules = rules
        
        # Column pool: (pairing_id, crew_id) -> Pairing
        self.pairings: Dict[Tuple[str, str], Pairing] = {}
        
        # Model objects (rebuilt on each solve)
        self.model: Optional[pulp.LpProblem] = None
        self.variables: Dict[Tuple[str, str], pulp.LpVariable] = {}
        
        # Constraint references (for dual extraction)
        self.flight_constraints: Dict[str, pulp.LpConstraint] = {}
        self.crew_constraints: Dict[str, pulp.LpConstraint] = {}
    
    def add_pairing(self, pairing: Pairing) -> bool:
        """
        Add a new pairing to the column pool.
        
        Returns True if pairing was added (not a duplicate).
        """
        key = (pairing.id, pairing.crew_id)
        if key in self.pairings:
            return False
        
        self.pairings[key] = pairing
        return True
    
    def add_pairings(self, pairings: List[Pairing]) -> int:
        """Add multiple pairings. Returns count of new pairings added."""
        return sum(1 for p in pairings if self.add_pairing(p))
    
    def build_model(self, relax: bool = True) -> pulp.LpProblem:
        """
        Build or rebuild the optimization model.
        
        Args:
            relax: If True, use continuous variables (LP relaxation)
                   If False, use binary variables (MIP)
        
        Returns:
            The PuLP model object
        """
        self.model = pulp.LpProblem("CrewScheduling3D", pulp.LpMinimize)
        self.variables = {}
        self.flight_constraints = {}
        self.crew_constraints = {}
        
        # Create variables for each (pairing, crew) combination
        for (pairing_id, crew_id), pairing in self.pairings.items():
            crew = self.crew[crew_id]
            cost = pairing.compute_cost(crew)
            
            var_name = f"x_{pairing_id}_{crew_id}"
            if relax:
                var = pulp.LpVariable(var_name, 0, 1, cat=pulp.LpContinuous)
            else:
                var = pulp.LpVariable(var_name, cat=pulp.LpBinary)
            
            self.variables[(pairing_id, crew_id)] = var
        
        # Objective: minimize total cost
        self.model += pulp.lpSum(
            pairing.compute_cost(self.crew[crew_id]) * var
            for (pairing_id, crew_id), var in self.variables.items()
            for pairing in [self.pairings[(pairing_id, crew_id)]]
        ), "TotalCost"
        
        # Constraint 1: Flight coverage (each flight exactly once)
        for flight_id in self.flights:
            covering_vars = []
            for (pairing_id, crew_id), var in self.variables.items():
                pairing = self.pairings[(pairing_id, crew_id)]
                if pairing.covers_flight(flight_id):
                    covering_vars.append(var)
            
            if covering_vars:
                constraint = pulp.lpSum(covering_vars) == 1
                constraint_name = f"FlightCoverage_{flight_id}"
                self.model += constraint, constraint_name
                self.flight_constraints[flight_id] = constraint
        
        # Constraint 2: Crew assignment (each crew exactly one pairing)
        for crew_id in self.crew:
            crew_vars = [
                var for (pid, cid), var in self.variables.items()
                if cid == crew_id
            ]
            
            if crew_vars:
                constraint = pulp.lpSum(crew_vars) == 1
                constraint_name = f"CrewAssignment_{crew_id}"
                self.model += constraint, constraint_name
                self.crew_constraints[crew_id] = constraint
        
        return self.model
    
    def solve(self, solver: str = "CBC") -> Tuple[float, Dict]:
        """
        Solve the current model.
        
        Args:
            solver: Solver to use ("CBC", "GUROBI", "CPLEX")
        
        Returns:
            (objective_value, solution_dict)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Select solver
        if solver == "GUROBI":
            slv = pulp.GUROBI_CMD(msg=0)
        elif solver == "CPLEX":
            slv = pulp.CPLEX_CMD(msg=0)
        else:
            slv = pulp.PULP_CBC_CMD(msg=0)
        
        # Solve
        self.model.solve(slv)
        
        # Check status
        if self.model.status != pulp.LpStatusOptimal:
            raise RuntimeError(
                f"Solver did not find optimal solution. "
                f"Status: {pulp.LpStatus[self.model.status]}"
            )
        
        # Extract solution
        objective = pulp.value(self.model.objective)
        solution = {
            (pid, cid): pulp.value(var)
            for (pid, cid), var in self.variables.items()
        }
        
        return objective, solution
    
    def get_duals(self) -> DualValues:
        """
        Extract dual values after solving LP relaxation.
        
        Must be called after solve() on an LP relaxation.
        
        Returns:
            DualValues object with flight and crew duals
        """
        if self.model is None or self.model.status != pulp.LpStatusOptimal:
            raise ValueError("Model must be solved optimally first.")
        
        # Extract flight duals (π_i)
        flight_duals = {}
        for flight_id, constraint in self.flight_constraints.items():
            # PuLP stores dual in constraint.pi
            dual = constraint.pi if hasattr(constraint, 'pi') else 0.0
            flight_duals[flight_id] = dual if dual is not None else 0.0
        
        # Extract crew duals (σ_k)
        crew_duals = {}
        for crew_id, constraint in self.crew_constraints.items():
            dual = constraint.pi if hasattr(constraint, 'pi') else 0.0
            crew_duals[crew_id] = dual if dual is not None else 0.0
        
        return DualValues(flight_duals=flight_duals, crew_duals=crew_duals)
    
    def get_solution_pairings(
        self, 
        threshold: float = 0.5
    ) -> Dict[str, Pairing]:
        """
        Get pairings selected in solution (value > threshold).
        
        Returns:
            Dict mapping crew_id to their assigned pairing
        """
        _, solution = self.solve()
        
        assignments = {}
        for (pairing_id, crew_id), value in solution.items():
            if value > threshold:
                assignments[crew_id] = self.pairings[(pairing_id, crew_id)]
        
        return assignments
    
    @property
    def num_columns(self) -> int:
        """Number of columns (pairings) in the pool."""
        return len(self.pairings)
    
    @property
    def num_flights(self) -> int:
        """Number of flights to cover."""
        return len(self.flights)
    
    @property
    def num_crew(self) -> int:
        """Number of crew members."""
        return len(self.crew)
```

### 4.2 Subproblem Interface

```python
# optimization/subproblem/base.py

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from dataclasses import dataclass

from models import Flight, Crew, Pairing, LegalRules, FlightNetwork

@dataclass
class SubproblemResult:
    """Result from solving a pricing subproblem."""
    pairing: Optional[Pairing]
    reduced_cost: float
    solve_time_ms: float
    nodes_explored: int = 0
    labels_created: int = 0

class PricingSubproblem(ABC):
    """
    Abstract base class for pricing subproblems.
    
    Each subproblem finds the minimum reduced cost pairing
    for a specific crew member.
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
        self.network = FlightNetwork(flights, crew, rules)
    
    @abstractmethod
    def solve(
        self,
        flight_duals: Dict[str, float],
        crew_dual: float,
        time_limit_ms: int = 10000
    ) -> SubproblemResult:
        """
        Solve the pricing problem.
        
        Args:
            flight_duals: Dual values π_i for each flight
            crew_dual: Dual value σ_k for this crew
            time_limit_ms: Maximum solve time in milliseconds
        
        Returns:
            SubproblemResult containing best pairing found (if any)
        """
        pass
    
    def compute_reduced_cost(
        self,
        pairing: Pairing,
        flight_duals: Dict[str, float],
        crew_dual: float
    ) -> float:
        """
        Compute reduced cost for a pairing.
        
        c̄_{jk} = c_{jk} - Σ(π_i for i in j) - σ_k
        """
        # Direct cost
        cost = pairing.compute_cost(self.crew)
        
        # Subtract flight duals
        for flight in pairing.flights:
            cost -= flight_duals.get(flight.id, 0.0)
        
        # Subtract crew dual
        cost -= crew_dual
        
        return cost
```

### 4.3 Exact RCSPP Solver

```python
# optimization/subproblem/exact_rcspp.py

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from heapq import heappush, heappop
import time

from models import Flight, Crew, Pairing, LegalRules
from .base import PricingSubproblem, SubproblemResult

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
    current_time: float = field(compare=False)
    
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
            for neighbor in self.network.graph.successors(current.node):
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
                    l for l in labels_at_node[neighbor]
                    if not new_label.dominates(l)
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
        arc_key = (label.node, to_node)
        arc = self.network.arcs.get(arc_key)
        
        if arc is None:
            return None
        
        # Compute arc cost
        arc_cost = self.network.get_arc_cost(
            label.node, to_node, flight_duals
        )
        
        # Update resources
        new_duty = label.duty_time
        if arc.arc_type == "flight":
            new_duty += arc.duration.total_seconds() / 3600
        
        # Check feasibility
        if new_duty > self.crew.max_duty_hours:
            return None
        
        return Label(
            cost=label.cost + arc_cost,
            node=to_node,
            path=label.path + (to_node,),
            duty_time=new_duty,
            current_time=label.current_time + arc.duration.total_seconds() / 3600
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
        
        flights = [
            f for f in self.flights 
            if f.id in flight_ids
        ]
        
        # Sort by departure time
        flights.sort(key=lambda f: f.departure)
        
        return Pairing.create(flights, self.crew)
```

### 4.4 Column Generation Orchestrator

```python
# optimization/column_generation.py

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import time
import logging

from models import Flight, Crew, Pairing, LegalRules, Solution
from .master_problem import MasterProblem, DualValues
from .subproblem.base import PricingSubproblem
from .subproblem.exact_rcspp import ExactRCSPP

logger = logging.getLogger(__name__)

@dataclass
class IterationResult:
    """Result of a single column generation iteration."""
    iteration: int
    objective: float
    columns_added: int
    best_reduced_cost: float
    solve_time_ms: float
    duals: Optional[DualValues] = None

class ColumnGeneration:
    """
    Main column generation orchestrator.
    
    Coordinates the master problem and pricing subproblems
    to solve the integrated crew scheduling problem.
    """
    
    def __init__(
        self,
        flights: List[Flight],
        crew: List[Crew],
        rules: LegalRules,
        subproblem_class: type = ExactRCSPP,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        max_columns_per_iter: int = 10,
        callback: Optional[Callable[[IterationResult], None]] = None
    ):
        self.flights = flights
        self.crew = crew
        self.rules = rules
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_columns_per_iter = max_columns_per_iter
        self.callback = callback
        
        # Initialize master problem
        self.master = MasterProblem(flights, crew, rules)
        
        # Initialize subproblems (one per crew member)
        self.subproblems: Dict[str, PricingSubproblem] = {}
        for c in crew:
            self.subproblems[c.id] = subproblem_class(flights, c, rules)
        
        # History
        self.iteration_history: List[IterationResult] = []
    
    def initialize(self):
        """
        Generate initial feasible pairings.
        
        Creates simple pairings to ensure feasibility:
        - For each crew, find a simple out-and-back pairing from their base
        - Or a "dummy" pairing with high cost if no valid pairing exists
        """
        logger.info("Generating initial pairings...")
        
        for crew in self.crew:
            # Find flights from/to crew's base
            base_departures = [
                f for f in self.flights 
                if f.origin == crew.base
            ]
            base_arrivals = [
                f for f in self.flights
                if f.destination == crew.base
            ]
            
            # Try to find a valid round-trip pairing
            found_pairing = False
            for dep_flight in base_departures:
                for arr_flight in base_arrivals:
                    if dep_flight.can_connect_to(arr_flight, self.rules.min_connection_time):
                        pairing = Pairing.create([dep_flight, arr_flight], crew)
                        if pairing.is_legal(crew, self.rules):
                            self.master.add_pairing(pairing)
                            found_pairing = True
                            logger.debug(f"Initial pairing for {crew.id}: {pairing}")
                            break
                if found_pairing:
                    break
            
            if not found_pairing:
                # Create a dummy/artificial pairing with high cost
                logger.warning(
                    f"No valid initial pairing for {crew.id}. "
                    "Adding artificial pairing."
                )
                # This would need special handling - omitted for brevity
        
        logger.info(f"Initialized with {self.master.num_columns} pairings")
    
    def run(self) -> Solution:
        """
        Run the column generation algorithm.
        
        Returns:
            Solution object with final assignments
        """
        start_time = time.time()
        
        # Initialize
        self.initialize()
        
        # Main loop
        for iteration in range(1, self.max_iterations + 1):
            iter_start = time.time()
            
            # Step 1: Solve master problem (LP relaxation)
            self.master.build_model(relax=True)
            objective, solution = self.master.solve()
            duals = self.master.get_duals()
            
            logger.info(
                f"Iteration {iteration}: "
                f"Objective = {objective:.2f}, "
                f"Columns = {self.master.num_columns}"
            )
            
            # Step 2: Solve pricing subproblems
            new_columns = []
            best_reduced_cost = float('inf')
            
            for crew_id, subproblem in self.subproblems.items():
                result = subproblem.solve(
                    duals.flight_duals,
                    duals.get_crew_dual(crew_id)
                )
                
                if result.pairing and result.reduced_cost < -self.tolerance:
                    new_columns.append(result.pairing)
                    best_reduced_cost = min(best_reduced_cost, result.reduced_cost)
                    
                    logger.debug(
                        f"  {crew_id}: reduced cost = {result.reduced_cost:.4f}"
                    )
            
            # Step 3: Add columns and check convergence
            columns_added = self.master.add_pairings(
                new_columns[:self.max_columns_per_iter]
            )
            
            iter_time = (time.time() - iter_start) * 1000
            
            # Record iteration
            iter_result = IterationResult(
                iteration=iteration,
                objective=objective,
                columns_added=columns_added,
                best_reduced_cost=best_reduced_cost,
                solve_time_ms=iter_time,
                duals=duals
            )
            self.iteration_history.append(iter_result)
            
            if self.callback:
                self.callback(iter_result)
            
            # Check convergence
            if columns_added == 0:
                logger.info(f"Converged at iteration {iteration}")
                break
        
        # Step 4: Get integer solution
        solution = self._get_integer_solution()
        
        total_time = time.time() - start_time
        solution.statistics.solve_time_seconds = total_time
        solution.iteration_history = self.iteration_history
        
        return solution
    
    def _get_integer_solution(self) -> Solution:
        """
        Obtain integer solution from LP solution.
        
        Options:
        1. Simple rounding (if LP solution is nearly integral)
        2. Solve MIP with all generated columns
        """
        # Solve MIP
        self.master.build_model(relax=False)
        objective, solution = self.master.solve()
        
        # Build solution object
        from models import Solution, SolutionStatistics, CrewAssignment
        
        assignments = {}
        for (pairing_id, crew_id), value in solution.items():
            if value > 0.5:
                pairing = self.master.pairings[(pairing_id, crew_id)]
                crew = self.master.crew[crew_id]
                assignments[crew_id] = CrewAssignment(
                    crew=crew,
                    pairing=pairing,
                    cost=pairing.compute_cost(crew)
                )
        
        stats = SolutionStatistics(
            total_cost=objective,
            total_flight_coverage=sum(
                len(a.flights) for a in assignments.values()
            ),
            total_crew_assigned=len(assignments),
            total_duty_hours=sum(
                a.duty_hours for a in assignments.values()
            ),
            average_duty_hours=sum(
                a.duty_hours for a in assignments.values()
            ) / max(len(assignments), 1),
            iterations=len(self.iteration_history),
            solve_time_seconds=0,  # Updated by caller
            columns_generated=self.master.num_columns,
            is_integer=True
        )
        
        return Solution(assignments=assignments, statistics=stats)
```

---

## 5. Graph Neural Network Module

### 5.1 GNN Architecture

```python
# gnn/models/gat_pricing.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional

class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder for flight networks.
    
    Processes the time-space network to produce node embeddings
    that capture structural and dual-value information.
    """
    
    def __init__(
        self,
        node_features: int = 6,
        edge_features: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        
        # Initial embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=hidden_dim
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            batch: Batch assignment [num_nodes] (for batched graphs)
        
        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        # Initial encoding
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        
        # GAT layers with residual connections
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            h_new = gat(h, edge_index, edge_attr=e)
            h = norm(h + self.dropout(h_new))  # Residual + norm
        
        return h


class ArcScorer(nn.Module):
    """
    Scores arcs to predict probability of being in optimal path.
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Score each arc.
        
        Args:
            node_embeddings: [num_nodes, hidden_dim]
            edge_index: [2, num_edges]
        
        Returns:
            Arc scores [num_edges, 1]
        """
        src = node_embeddings[edge_index[0]]  # [num_edges, hidden_dim]
        dst = node_embeddings[edge_index[1]]  # [num_edges, hidden_dim]
        
        edge_features = torch.cat([src, dst], dim=-1)
        return self.scorer(edge_features)


class PricingGNN(nn.Module):
    """
    Complete GNN model for pricing subproblem.
    
    Can be used in two modes:
    1. Arc scoring: Predict probability each arc is in optimal path
    2. Autoregressive: Generate path one step at a time
    """
    
    def __init__(
        self,
        node_features: int = 6,
        edge_features: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = GATEncoder(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.arc_scorer = ArcScorer(hidden_dim)
        
        # Global context
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for crew_dual
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            data: PyG Data object with:
                - x: node features
                - edge_index: edges
                - edge_attr: edge features
                - crew_dual: scalar crew dual value
        
        Returns:
            (arc_scores, global_embedding)
        """
        # Encode nodes
        node_embeddings = self.encoder(
            data.x,
            data.edge_index,
            data.edge_attr,
            getattr(data, 'batch', None)
        )
        
        # Score arcs
        arc_scores = self.arc_scorer(node_embeddings, data.edge_index)
        
        # Global embedding (for value prediction)
        global_emb = global_mean_pool(
            node_embeddings, 
            getattr(data, 'batch', None)
        )
        
        # Incorporate crew dual
        crew_dual = data.crew_dual.view(-1, 1)
        if crew_dual.size(0) != global_emb.size(0):
            crew_dual = crew_dual.expand(global_emb.size(0), 1)
        
        global_emb = self.global_mlp(
            torch.cat([global_emb, crew_dual], dim=-1)
        )
        
        return arc_scores, global_emb
    
    def predict_arc_probabilities(self, data: Data) -> torch.Tensor:
        """Predict probability each arc is in optimal path."""
        arc_scores, _ = self.forward(data)
        return arc_scores.squeeze(-1)


class AutoregressiveDecoder(nn.Module):
    """
    Autoregressive decoder for path generation.
    
    Generates paths one step at a time, selecting the next
    node based on current state and node embeddings.
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        self.state_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        self.next_node_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        current_node_idx: int,
        state: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next node in path.
        
        Args:
            node_embeddings: [num_nodes, hidden_dim]
            current_node_idx: Index of current node
            state: Hidden state [1, hidden_dim]
            valid_mask: [num_nodes] boolean mask of valid next nodes
        
        Returns:
            (logits over nodes, new_state)
        """
        # Update state with current node
        current_emb = node_embeddings[current_node_idx].unsqueeze(0).unsqueeze(0)
        _, new_state = self.state_encoder(current_emb, state.unsqueeze(0))
        new_state = new_state.squeeze(0)
        
        # Score all nodes
        state_expanded = new_state.expand(node_embeddings.size(0), -1)
        combined = torch.cat([node_embeddings, state_expanded], dim=-1)
        logits = self.next_node_scorer(combined).squeeze(-1)
        
        # Mask invalid nodes
        logits[~valid_mask] = float('-inf')
        
        return logits, new_state
```

### 5.2 Training Pipeline

```python
# gnn/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from typing import Dict, List, Optional, Callable
import logging
from tqdm import tqdm

from ..models.gat_pricing import PricingGNN
from ..data.dataset import PricingDataset

logger = logging.getLogger(__name__)

class PricingGNNTrainer:
    """
    Trainer for the Pricing GNN model.
    
    Supports:
    - Supervised training on optimal solutions
    - Reinforcement learning fine-tuning
    - Mixed training strategies
    """
    
    def __init__(
        self,
        model: PricingGNN,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_arc_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            arc_scores, global_emb = self.model(batch)
            
            # Compute arc classification loss
            # batch.arc_labels contains 1 for arcs in optimal path, 0 otherwise
            arc_loss = self.bce_loss(
                arc_scores.squeeze(),
                batch.arc_labels.float()
            )
            
            # Total loss
            loss = arc_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_arc_loss += arc_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': loss.item(),
                'arc_loss': arc_loss.item()
            })
        
        return {
            'loss': total_loss / num_batches,
            'arc_loss': total_arc_loss / num_batches
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_arcs = 0
        
        for batch in val_loader:
            batch = batch.to(self.device)
            
            arc_scores, _ = self.model(batch)
            
            # Loss
            arc_loss = self.bce_loss(
                arc_scores.squeeze(),
                batch.arc_labels.float()
            )
            total_loss += arc_loss.item()
            
            # Accuracy
            predictions = (arc_scores.squeeze() > 0.5).float()
            total_correct += (predictions == batch.arc_labels).sum().item()
            total_arcs += batch.arc_labels.size(0)
        
        return {
            'val_loss': total_loss / len(val_loader),
            'arc_accuracy': total_correct / total_arcs
        }
    
    def train(
        self,
        train_dataset: PricingDataset,
        val_dataset: PricingDataset,
        num_epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Returns:
            Dictionary with training history
        """
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size
        )
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'arc_accuracy': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_metrics['val_loss'])
            
            # Log
            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['val_loss']:.4f}, "
                f"arc_acc={val_metrics['arc_accuracy']:.4f}"
            )
            
            # History
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['arc_accuracy'].append(val_metrics['arc_accuracy'])
            
            # Early stopping
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                
                if checkpoint_path:
                    torch.save(self.model.state_dict(), checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        return history
```

### 5.3 GNN-Guided Subproblem

```python
# optimization/subproblem/gnn_guided.py

import torch
from typing import Dict, List, Optional
import time

from models import Flight, Crew, Pairing, LegalRules
from gnn.models.gat_pricing import PricingGNN
from .base import PricingSubproblem, SubproblemResult
from .exact_rcspp import Label

class GNNGuidedRCSPP(PricingSubproblem):
    """
    GNN-guided Resource-Constrained Shortest Path solver.
    
    Uses a trained GNN to predict which arcs are likely to be
    in the optimal path, then guides the search accordingly.
    """
    
    def __init__(
        self,
        flights: List[Flight],
        crew: Crew,
        rules: LegalRules,
        gnn_model: PricingGNN,
        pruning_threshold: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(flights, crew, rules)
        
        self.gnn = gnn_model.to(device)
        self.gnn.eval()
        self.device = device
        self.pruning_threshold = pruning_threshold
    
    def solve(
        self,
        flight_duals: Dict[str, float],
        crew_dual: float,
        time_limit_ms: int = 10000
    ) -> SubproblemResult:
        """
        Solve RCSPP using GNN-guided search.
        """
        start_time = time.time()
        
        # Convert network to PyG format
        pyg_data = self.network.to_pyg_data(flight_duals, crew_dual)
        pyg_data = pyg_data.to(self.device)
        
        # Get GNN predictions
        with torch.no_grad():
            arc_probs = self.gnn.predict_arc_probabilities(pyg_data)
        
        arc_probs = arc_probs.cpu().numpy()
        
        # Build arc probability lookup
        edge_index = pyg_data.edge_index.cpu().numpy()
        node_list = list(self.network.nodes.keys())
        
        arc_prob_dict = {}
        for i, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
            src_node = node_list[src]
            dst_node = node_list[dst]
            arc_prob_dict[(src_node, dst_node)] = arc_probs[i]
        
        # Run guided label-setting
        result = self._guided_label_setting(
            flight_duals,
            crew_dual,
            arc_prob_dict,
            time_limit_ms - (time.time() - start_time) * 1000
        )
        
        result.solve_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _guided_label_setting(
        self,
        flight_duals: Dict[str, float],
        crew_dual: float,
        arc_probs: Dict,
        time_limit_ms: float
    ) -> SubproblemResult:
        """
        Label-setting algorithm guided by GNN predictions.
        
        Key differences from exact RCSPP:
        1. Priority is based on arc probability * (-cost)
        2. Arcs with very low probability can be pruned
        """
        from heapq import heappush, heappop
        
        start_time = time.time()
        
        # Initialize
        labels_at_node = {node: [] for node in self.network.nodes}
        heap = []
        
        # Initial label at source
        initial_label = Label(
            cost=0.0,
            node="SOURCE",
            path=("SOURCE",),
            duty_time=0.0,
            current_time=0.0
        )
        heappush(heap, (0.0, initial_label))  # (priority, label)
        labels_at_node["SOURCE"].append(initial_label)
        
        best_sink_label = None
        nodes_explored = 0
        labels_created = 1
        
        while heap:
            # Check time limit
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > time_limit_ms:
                break
            
            _, current = heappop(heap)
            nodes_explored += 1
            
            # Skip dominated labels
            if self._is_dominated(current, labels_at_node[current.node]):
                continue
            
            # Check if reached sink
            if current.node == "SINK":
                if best_sink_label is None or current.cost < best_sink_label.cost:
                    best_sink_label = current
                continue
            
            # Extend to neighbors
            for neighbor in self.network.graph.successors(current.node):
                # Get arc probability
                arc_prob = arc_probs.get((current.node, neighbor), 0.0)
                
                # Prune low-probability arcs
                if arc_prob < self.pruning_threshold:
                    continue
                
                new_label = self._extend_label(current, neighbor, flight_duals)
                
                if new_label is None:
                    continue
                
                if self._is_dominated(new_label, labels_at_node[neighbor]):
                    continue
                
                # Priority: higher probability = lower priority value
                priority = -arc_prob * (1.0 / (new_label.cost + 1e-6))
                
                labels_at_node[neighbor].append(new_label)
                heappush(heap, (priority, new_label))
                labels_created += 1
                
                # Clean dominated
                labels_at_node[neighbor] = [
                    l for l in labels_at_node[neighbor]
                    if not new_label.dominates(l)
                ]
        
        # Build result
        if best_sink_label is None:
            return SubproblemResult(
                pairing=None,
                reduced_cost=float('inf'),
                solve_time_ms=0,
                nodes_explored=nodes_explored,
                labels_created=labels_created
            )
        
        pairing = self._path_to_pairing(best_sink_label.path)
        reduced_cost = best_sink_label.cost - crew_dual
        
        return SubproblemResult(
            pairing=pairing,
            reduced_cost=reduced_cost,
            solve_time_ms=0,
            nodes_explored=nodes_explored,
            labels_created=labels_created
        )
    
    def _extend_label(self, label, to_node, flight_duals):
        """Same as ExactRCSPP._extend_label"""
        # (Implementation omitted - same as exact version)
        pass
    
    def _is_dominated(self, label, existing):
        """Same as ExactRCSPP._is_dominated"""
        pass
    
    def _path_to_pairing(self, path):
        """Same as ExactRCSPP._path_to_pairing"""
        pass
```

---

## 6. Configuration Management

```yaml
# config/default.yaml

# Problem settings
problem:
  horizon_days: 2
  min_connection_minutes: 60
  max_duty_hours: 10
  min_rest_hours: 10

# Solver settings
solver:
  engine: "CBC"  # CBC, GUROBI, CPLEX
  time_limit_seconds: 3600
  mip_gap: 0.01

# Column generation settings
column_generation:
  max_iterations: 100
  tolerance: 1e-6
  max_columns_per_iteration: 20
  pricing_time_limit_ms: 5000

# GNN settings
gnn:
  enabled: false
  model_path: "checkpoints/best_gnn.pt"
  architecture:
    node_features: 6
    edge_features: 8
    hidden_dim: 64
    num_layers: 4
    num_heads: 4
    dropout: 0.1
  training:
    learning_rate: 0.001
    batch_size: 32
    num_epochs: 100
    early_stopping_patience: 10
  inference:
    pruning_threshold: 0.01
    use_cuda: true

# Logging
logging:
  level: INFO
  file: "logs/solver.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Output
output:
  solution_file: "outputs/solution.json"
  visualization: true
  save_iterations: true
```

---

## 7. Testing Strategy

### 7.1 Test Categories

```python
# tests/conftest.py

import pytest
from datetime import datetime, timedelta
from models import Flight, Crew, LegalRules, AircraftType, CrewRank, Qualification

@pytest.fixture
def simple_flights():
    """Two flights forming a round trip."""
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
    from data.generators.micro_airline import generate_micro_airline
    return generate_micro_airline()
```

### 7.2 Unit Tests

```python
# tests/unit/test_models.py

import pytest
from datetime import datetime, timedelta
from models import Flight, Crew, Pairing, LegalRules

class TestFlight:
    def test_duration(self, simple_flights):
        f1 = simple_flights[0]
        assert f1.duration_hours == 3.0
    
    def test_connection_valid(self, simple_flights):
        f1, f2 = simple_flights
        min_conn = timedelta(hours=1)
        assert f1.can_connect_to(f2, min_conn) == True
    
    def test_connection_invalid_location(self, simple_flights):
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
        assert f1.can_connect_to(f_sfo, timedelta(hours=1)) == False

class TestPairing:
    def test_creation(self, simple_flights, simple_crew):
        pairing = Pairing.create(simple_flights, simple_crew[0])
        assert pairing.crew_id == "C1"
        assert len(pairing.flights) == 2
    
    def test_base_constraint(self, simple_flights, simple_crew):
        pairing = Pairing.create(simple_flights, simple_crew[0])
        assert pairing.start_base == "JFK"
        assert pairing.end_base == "JFK"
    
    def test_duty_hours(self, simple_flights, simple_crew):
        pairing = Pairing.create(simple_flights, simple_crew[0])
        # 8am to 5pm = 9 hours
        assert pairing.total_duty_hours == 9.0
```

### 7.3 Integration Tests

```python
# tests/integration/test_column_generation.py

import pytest
from optimization.column_generation import ColumnGeneration
from optimization.subproblem.exact_rcspp import ExactRCSPP

class TestColumnGeneration:
    def test_micro_airline_converges(self, micro_airline):
        flights, crew, rules = micro_airline
        
        cg = ColumnGeneration(
            flights=flights,
            crew=crew,
            rules=rules,
            subproblem_class=ExactRCSPP,
            max_iterations=50
        )
        
        solution = cg.run()
        
        # Should converge
        assert len(cg.iteration_history) < 50
        
        # Should be feasible
        assert solution.is_feasible
        
        # All flights covered
        coverage = solution.get_flight_coverage()
        assert len(coverage) == len(flights)
    
    def test_constraints_satisfied(self, micro_airline):
        flights, crew, rules = micro_airline
        
        cg = ColumnGeneration(flights, crew, rules)
        solution = cg.run()
        
        verification = solution.verify_constraints(flights, crew)
        
        assert verification["all_flights_covered"]
        assert verification["each_flight_once"]
        assert verification["all_crew_assigned"]
        assert verification["base_constraints"]
        assert verification["duty_limits"]
```

---

## 8. Deployment Considerations

### 8.1 Scalability

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SCALING STRATEGIES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HORIZONTAL SCALING (Parallel Subproblems)                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Crew 1 SP   │  │ Crew 2 SP   │  │ Crew 3 SP   │  │ Crew N SP   │        │
│  │ (Worker 1)  │  │ (Worker 2)  │  │ (Worker 3)  │  │ (Worker N)  │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         └────────────────┴────────────────┴────────────────┘                │
│                                    │                                        │
│                                    ▼                                        │
│                          ┌─────────────────┐                                │
│                          │  Column Pool    │                                │
│                          │  Aggregator     │                                │
│                          └────────┬────────┘                                │
│                                   │                                         │
│                                   ▼                                         │
│                          ┌─────────────────┐                                │
│                          │  Master Problem │                                │
│                          │  (Single Node)  │                                │
│                          └─────────────────┘                                │
│                                                                             │
│  GPU ACCELERATION (GNN Batch Inference)                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Batch multiple crew subproblems → Single GNN forward pass          │   │
│  │  Throughput: ~1000 pricing problems per second on RTX 3090          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Production Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PRODUCTION PIPELINE                                     │
│                                                                             │
│  1. DATA INGESTION                                                          │
│     • Flight schedule from OAG/Sabre                                        │
│     • Crew data from HR system                                              │
│     • Regulatory rules from compliance database                             │
│                                                                             │
│  2. PREPROCESSING                                                           │
│     • Validate and clean data                                               │
│     • Build flight connection graph                                         │
│     • Compute initial bounds                                                │
│                                                                             │
│  3. OPTIMIZATION                                                            │
│     • Run column generation                                                 │
│     • Apply GNN acceleration if enabled                                     │
│     • Obtain integer solution                                               │
│                                                                             │
│  4. POST-PROCESSING                                                         │
│     • Validate solution feasibility                                         │
│     • Generate reports                                                      │
│     • Export to crew management system                                      │
│                                                                             │
│  5. MONITORING                                                              │
│     • Track solution quality metrics                                        │
│     • Log performance data                                                  │
│     • Alert on infeasibility                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Future Extensions

1. **Multi-day horizons**: Extend from 2-day to weekly/monthly scheduling
2. **Crew qualifications**: Detailed qualification matching (aircraft types, routes)
3. **Preferences**: Incorporate crew preferences and seniority bidding
4. **Disruption handling**: Real-time re-optimization when disruptions occur
5. **Fairness constraints**: Ensure equitable distribution of desirable trips
6. **Training/checking**: Account for training and check rides in pairings
7. **Reserve crew**: Model reserve crew and open-time assignment
8. **Multi-objective**: Pareto optimization of cost, fairness, and robustness

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-XX-XX | AI Agent | Initial architecture document |
