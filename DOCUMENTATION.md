# 3D Integrated Crew Scheduling System

## Complete Technical Documentation

---

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [Architecture](#3-architecture)
4. [Core Data Models](#4-core-data-models)
5. [Time-Space Network](#5-time-space-network)
6. [Column Generation Algorithm](#6-column-generation-algorithm)
7. [Master Problem](#7-master-problem)
8. [Pricing Subproblem (RCSPP)](#8-pricing-subproblem-rcspp)
9. [Test Instances](#9-test-instances)
10. [Usage Guide](#10-usage-guide)
11. [Optimality Guarantees](#11-optimality-guarantees)
12. [Future Extensions](#12-future-extensions)

---

## 1. Overview

### 1.1 Problem Description

The **3D Integrated Crew Scheduling Problem** assigns airline crew members to flight sequences (pairings) while:

1. **Covering all flights** exactly once
2. **Respecting crew constraints** (duty limits, rest periods, base requirements)
3. **Minimizing total cost** (flight costs + crew labor costs)

The "3D" refers to the three integrated dimensions:
- **Flights** (what needs to be covered)
- **Crew** (who can fly)
- **Time** (when things happen)

### 1.2 Why Column Generation?

The number of possible pairings grows exponentially with flights. For `n` flights:
- Naive enumeration: O(n!) possible sequences
- With 100 flights: astronomical number of variables

**Column Generation** solves this by:
1. Starting with a small subset of pairings
2. Iteratively adding only "promising" pairings (negative reduced cost)
3. Proving optimality when no improving pairings exist

### 1.3 Project Structure

```
crew_scheduling_3d/
├── models/                      # Core data structures
│   ├── __init__.py
│   ├── flight.py               # Flight, AircraftType
│   ├── crew.py                 # Crew, CrewRank, Qualification
│   ├── pairing.py              # Pairing (flight sequence)
│   ├── rules.py                # LegalRules (constraints)
│   ├── network.py              # FlightNetwork (time-space graph)
│   └── solution.py             # Solution, CrewAssignment, Statistics
├── optimization/
│   ├── __init__.py
│   ├── column_generation.py    # Main CG orchestrator
│   ├── master_problem.py       # LP/MIP formulation
│   └── subproblem/
│       ├── __init__.py
│       ├── base.py             # Abstract pricing interface
│       └── exact_rcspp.py      # Label-setting RCSPP solver
├── data/
│   └── generators/
│       ├── __init__.py
│       ├── micro_airline.py    # 8-flight test instance
│       └── small_airline.py    # 24-flight test instance
├── api/
│   └── cli/
│       └── main.py             # Command-line interface
├── tests/                      # Unit tests
└── DOCUMENTATION.md            # This file
```

---

## 2. Mathematical Formulation

### 2.1 Sets and Indices

| Symbol | Description |
|--------|-------------|
| F | Set of flights, indexed by i |
| K | Set of crew members, indexed by k |
| P_k | Set of feasible pairings for crew k |
| p | A pairing (sequence of flights) |

### 2.2 Parameters

| Symbol | Description |
|--------|-------------|
| c_pk | Cost of pairing p for crew k |
| a_ip | 1 if pairing p covers flight i, 0 otherwise |

### 2.3 Decision Variables

| Symbol | Description |
|--------|-------------|
| x_pk | 1 if crew k is assigned pairing p, 0 otherwise |

### 2.4 Formulation

**Objective:** Minimize total cost
```
min  Σ_k Σ_p∈P_k  c_pk · x_pk
```

**Subject to:**

1. **Flight Coverage** (each flight exactly once):
```
Σ_k Σ_p∈P_k  a_ip · x_pk = 1    ∀i ∈ F
```

2. **Crew Assignment** (each crew exactly one pairing):
```
Σ_p∈P_k  x_pk = 1    ∀k ∈ K
```

3. **Binary Variables**:
```
x_pk ∈ {0, 1}    ∀k ∈ K, ∀p ∈ P_k
```

### 2.5 Dual Variables

From LP relaxation:
- **π_i**: Dual for flight coverage constraint i (value of covering flight i)
- **σ_k**: Dual for crew assignment constraint k (cost of using crew k)

### 2.6 Reduced Cost

For a pairing p assigned to crew k:
```
reduced_cost(p, k) = c_pk - Σ_{i∈p} π_i - σ_k
```

A negative reduced cost means adding this pairing could improve the objective.

---

## 3. Architecture

### 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Column Generation                         │
│                   (Orchestrator)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────────┐     │
│  │  Master Problem  │◄───────►│  Pricing Subproblem  │     │
│  │     (PuLP)       │  duals  │      (RCSPP)         │     │
│  │                  │◄───────►│                      │     │
│  │  LP Relaxation   │ columns │  Per-crew shortest   │     │
│  │  → MIP Final     │         │  path search         │     │
│  └──────────────────┘         └──────────────────────┘     │
│           │                            │                    │
│           ▼                            ▼                    │
│  ┌──────────────────┐         ┌──────────────────────┐     │
│  │  Column Pool     │         │  Flight Network      │     │
│  │  (Pairings)      │         │  (Time-Space Graph)  │     │
│  └──────────────────┘         └──────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

```
1. Initialize
   ├── Generate initial pairings (heuristic)
   └── Build master problem with initial columns

2. Main Loop (until convergence)
   ├── Solve LP relaxation
   ├── Extract dual values (π, σ)
   ├── For each crew k:
   │   └── Solve RCSPP with duals → find min reduced cost pairing
   ├── If best reduced cost < 0:
   │   └── Add new columns to master
   └── Else: CONVERGED

3. Finalize
   ├── Solve MIP with all generated columns
   └── Return integer solution
```

---

## 4. Core Data Models

### 4.1 Flight (`models/flight.py`)

```python
@dataclass
class Flight:
    id: str                      # Unique identifier (e.g., "F1")
    flight_number: str           # Display number (e.g., "AA100")
    origin: str                  # Departure airport (e.g., "JFK")
    destination: str             # Arrival airport (e.g., "LAX")
    departure: datetime          # Departure time
    arrival: datetime            # Arrival time
    base_cost: float             # Operating cost
    aircraft_type: AircraftType  # NARROW_BODY, WIDE_BODY, REGIONAL
```

**Key Methods:**
- `duration` → `timedelta`: Flight duration
- `can_connect_to(other, min_connection)` → `bool`: Check if connection is feasible

**Connection Logic:**
```python
def can_connect_to(self, other: Flight, min_connection: timedelta) -> bool:
    # Must arrive before other departs
    if self.arrival >= other.departure:
        return False
    # Must be at same airport
    if self.destination != other.origin:
        return False
    # Must have minimum connection time
    connection_time = other.departure - self.arrival
    return connection_time >= min_connection
```

### 4.2 Crew (`models/crew.py`)

```python
@dataclass
class Crew:
    id: str                           # Unique identifier (e.g., "C1")
    name: str                         # Display name
    base: str                         # Home airport (e.g., "JFK")
    rank: CrewRank                    # CAPTAIN, FIRST_OFFICER, FLIGHT_ATTENDANT
    hourly_cost: float                # Labor cost per hour
    qualifications: Set[Qualification] # Aircraft type certifications
    max_duty_hours: float             # Maximum duty period (e.g., 10h)
    min_rest_hours: float             # Minimum rest between duties
    seniority: int                    # For preference ordering
```

**Qualifications:**
```python
class Qualification(Enum):
    NARROW_BODY = "narrow_body"
    WIDE_BODY = "wide_body"
    REGIONAL = "regional"
```

### 4.3 Pairing (`models/pairing.py`)

A **pairing** is a sequence of flights that:
- Starts and ends at crew's home base
- Respects all connection and duty constraints

```python
@dataclass
class Pairing:
    id: str                  # Unique identifier
    flights: List[Flight]    # Ordered flight sequence
    crew_id: str             # Assigned crew member
```

**Key Methods:**

```python
def compute_cost(self, crew: Crew) -> float:
    """Total cost = flight costs + crew labor cost."""
    flight_cost = sum(f.base_cost for f in self.flights)
    labor_cost = self.duty_hours * crew.hourly_cost
    return flight_cost + labor_cost

def is_legal(self, crew: Crew, rules: LegalRules) -> bool:
    """Check all constraints."""
    return (
        self._check_base_constraint(crew) and
        self._check_connections(rules) and
        self._check_duty_limit(crew) and
        self._check_flight_time_limit(rules) and
        self._check_max_flights(rules)
    )
```

**Pairing Properties:**
- `flight_time`: Sum of actual flying time
- `duty_hours`: Wall-clock time from first departure to last arrival
- `total_cost`: Flight costs + labor costs

### 4.4 LegalRules (`models/rules.py`)

```python
@dataclass
class LegalRules:
    min_connection_time: timedelta   # Minimum between flights (e.g., 45 min)
    max_connection_time: timedelta   # Maximum wait time (e.g., 4 hours)
    max_duty_period: timedelta       # Maximum duty length (e.g., 10 hours)
    min_rest_period: timedelta       # Minimum rest (e.g., 10 hours)
    max_flight_time_per_duty: timedelta  # Max flying time (e.g., 8 hours)
    max_flights_per_duty: int        # Max flight legs (e.g., 4)
    must_return_to_base: bool        # Pairing must end at home base
```

### 4.5 Solution (`models/solution.py`)

```python
@dataclass
class Solution:
    assignments: Dict[str, CrewAssignment]  # crew_id → assignment
    statistics: SolutionStatistics
    iteration_history: List[Dict]           # CG iteration data

@dataclass
class CrewAssignment:
    crew: Crew
    pairing: Pairing
    cost: float

@dataclass
class SolutionStatistics:
    total_cost: float
    total_flight_coverage: int
    total_crew_assigned: int
    total_duty_hours: float
    average_duty_hours: float
    iterations: int
    solve_time_seconds: float
    columns_generated: int
    is_integer: bool
```

---

## 5. Time-Space Network

### 5.1 Concept

The **FlightNetwork** represents all possible crew movements as a directed graph:

```
         ┌─────────────────────────────────────────────────────┐
         │                    TIME →                          │
         │                                                     │
    JFK  │  ○────────────────────────────────────────────○    │
         │  │ SOURCE                              SINK    │    │
         │  │                                             │    │
         │  ▼         flight arc                          │    │
         │  [F1_DEP]═══════════►[F1_ARR]                  │    │
         │     │                   │                      │    │
         │     │ ground            │ connection           │    │
         │     │                   ▼                      │    │
    LAX  │     │              [F2_DEP]═══════►[F2_ARR]───►│    │
         │     │                                          │    │
         └─────┴──────────────────────────────────────────┴────┘
```

### 5.2 Node Types

```python
@dataclass
class NetworkNode:
    node_id: str        # "F1_DEP", "F1_ARR", "SOURCE", "SINK"
    node_type: str      # "departure", "arrival", "source", "sink"
    flight_id: str      # Associated flight (or None)
    airport: str        # Airport code
    time: datetime      # Event time
```

**Node Creation:**
- Each flight creates 2 nodes: `{flight_id}_DEP` and `{flight_id}_ARR`
- `SOURCE`: Virtual start node (connects to all departures from crew's base)
- `SINK`: Virtual end node (connects from all arrivals at crew's base)

### 5.3 Arc Types

```python
@dataclass
class NetworkArc:
    from_node: str
    to_node: str
    arc_type: str       # "flight", "connection", "source", "sink"
    flight_id: str      # For flight arcs
    cost: float         # Base cost
    duration: timedelta # For resource consumption
```

**Arc Types:**

| Type | From | To | Description |
|------|------|-----|-------------|
| `source` | SOURCE | F_DEP | Start pairing at base |
| `flight` | F_DEP | F_ARR | Fly the flight |
| `connection` | F1_ARR | F2_DEP | Wait for next flight |
| `sink` | F_ARR | SINK | End pairing at base |

### 5.4 Network Construction (`models/network.py`)

```python
class FlightNetwork:
    def __init__(self, flights: List[Flight], crew: Crew, rules: LegalRules):
        self.nodes: Dict[str, NetworkNode] = {}
        self.arcs: Dict[Tuple[str, str], NetworkArc] = {}

        # 1. Create SOURCE and SINK
        self._create_source_sink(flights)

        # 2. Create flight nodes and arcs
        for flight in flights:
            self._add_flight(flight)

        # 3. Create connection arcs (respecting constraints)
        self._create_connections(flights, rules)

        # 4. Connect SOURCE to departures from base
        # 5. Connect arrivals at base to SINK
        self._connect_source_sink(flights, crew)
```

### 5.5 Arc Cost Calculation

For pricing subproblem, arc costs incorporate dual values:

```python
def get_arc_cost(self, from_node: str, to_node: str,
                 flight_duals: Dict[str, float]) -> float:
    arc = self.arcs[(from_node, to_node)]

    if arc.arc_type == "flight":
        # Reduced cost: base_cost - dual_value
        dual = flight_duals.get(arc.flight_id, 0.0)
        return arc.cost - dual
    else:
        return arc.cost  # Connection arcs have zero cost
```

---

## 6. Column Generation Algorithm

### 6.1 Overview (`optimization/column_generation.py`)

```python
class ColumnGeneration:
    def __init__(self, flights, crew, rules, ...):
        self.master = MasterProblem(flights, crew, rules)
        self.subproblems = {
            c.id: ExactRCSPP(flights, c, rules)
            for c in crew
        }
```

### 6.2 Initialization

Generate initial pairings to ensure feasibility:

```python
def initialize(self) -> int:
    initial_count = 0

    # Strategy 1: All valid 2-flight round trips
    for crew in self.crew:
        for dep_flight in flights_from_base(crew):
            for arr_flight in flights_to_base(crew):
                if valid_connection(dep_flight, arr_flight):
                    pairing = Pairing.create([dep_flight, arr_flight], crew)
                    if pairing.is_legal(crew, rules):
                        self.master.add_pairing(pairing)
                        initial_count += 1

    # Strategy 2: Multi-leg pairings (3-4 flights)
    for crew in self.crew:
        initial_count += self._find_multi_leg_pairings(crew)

    return initial_count
```

### 6.3 Main Loop

```python
def run(self) -> Solution:
    self.initialize()

    for iteration in range(max_iterations):
        # Step 1: Solve LP relaxation
        self.master.build_model(relax=True)
        objective, solution = self.master.solve()
        duals = self.master.get_duals()

        # Step 2: Solve pricing subproblems
        new_columns = []
        for crew_id, subproblem in self.subproblems.items():
            result = subproblem.solve(
                duals.flight_duals,
                duals.get_crew_dual(crew_id)
            )
            if result.reduced_cost < -tolerance:
                new_columns.append(result.pairing)

        # Step 3: Check convergence
        if not new_columns:
            break  # No improving columns → optimal

        # Step 4: Add columns to master
        self.master.add_pairings(new_columns)

    # Step 5: Solve final MIP
    return self._get_integer_solution()
```

### 6.4 Convergence Criterion

The algorithm converges when:
```
min_{k,p} { c_pk - Σ_{i∈p} π_i - σ_k } ≥ 0
```

No pairing has negative reduced cost → LP relaxation is optimal.

---

## 7. Master Problem

### 7.1 Implementation (`optimization/master_problem.py`)

```python
class MasterProblem:
    def __init__(self, flights, crew, rules):
        self.flights = {f.id: f for f in flights}
        self.crew = {c.id: c for c in crew}
        self.pairings: Dict[Tuple[str, str], Pairing] = {}
        # Key: (pairing_id, crew_id)
```

### 7.2 Model Building

```python
def build_model(self, relax: bool = True):
    self.model = pulp.LpProblem("CrewScheduling3D", pulp.LpMinimize)

    # Variables
    for (pairing_id, crew_id), pairing in self.pairings.items():
        if relax:
            var = pulp.LpVariable(f"x_{pairing_id}_{crew_id}",
                                  0, 1, cat=pulp.LpContinuous)
        else:
            var = pulp.LpVariable(f"x_{pairing_id}_{crew_id}",
                                  cat=pulp.LpBinary)
        self.variables[(pairing_id, crew_id)] = var

    # Objective: minimize total cost
    self.model += pulp.lpSum([
        pairing.compute_cost(crew) * var
        for (pid, cid), var in self.variables.items()
        for pairing in [self.pairings[(pid, cid)]]
        for crew in [self.crew[cid]]
    ])

    # Constraint 1: Flight coverage (= 1)
    for flight_id in self.flights:
        covering_vars = [
            var for (pid, cid), var in self.variables.items()
            if self.pairings[(pid, cid)].covers_flight(flight_id)
        ]
        self.model += pulp.lpSum(covering_vars) == 1

    # Constraint 2: Crew assignment (= 1)
    for crew_id in self.crew:
        crew_vars = [
            var for (pid, cid), var in self.variables.items()
            if cid == crew_id
        ]
        self.model += pulp.lpSum(crew_vars) == 1
```

### 7.3 Dual Extraction

After solving LP relaxation, extract duals for pricing:

```python
def get_duals(self) -> DualValues:
    flight_duals = {}
    for flight_id, constraint in self.flight_constraints.items():
        flight_duals[flight_id] = constraint.pi or 0.0

    crew_duals = {}
    for crew_id, constraint in self.crew_constraints.items():
        crew_duals[crew_id] = constraint.pi or 0.0

    return DualValues(flight_duals, crew_duals)
```

---

## 8. Pricing Subproblem (RCSPP)

### 8.1 Problem Definition

**Resource-Constrained Shortest Path Problem (RCSPP):**

Find the minimum cost path from SOURCE to SINK in the time-space network, subject to:
- **Flight time** ≤ max_flight_hours (e.g., 8h)
- **Duty time** ≤ max_duty_hours (e.g., 10h)

### 8.2 Label-Setting Algorithm

A **label** represents a partial path with accumulated resources:

```python
@dataclass
class Label:
    cost: float           # Accumulated reduced cost
    node: str             # Current node
    path: Tuple[str, ...] # Path taken
    flight_time: float    # Total flying hours
    start_timestamp: float # First departure time
    end_timestamp: float   # Current time

    @property
    def duty_time(self) -> float:
        if self.start_timestamp == 0:
            return 0.0
        return (self.end_timestamp - self.start_timestamp) / 3600
```

### 8.3 Dominance Rules

Label A dominates Label B at the same node if:
```python
def dominates(self, other: Label) -> bool:
    return (
        self.cost <= other.cost and
        self.flight_time <= other.flight_time and
        self.end_timestamp <= other.end_timestamp and
        # Strictly better in at least one dimension
        (self.cost < other.cost or
         self.flight_time < other.flight_time or
         self.end_timestamp < other.end_timestamp)
    )
```

**Why dominance?** If A dominates B, any extension of B can be replaced by the same extension of A with equal or better result. We can prune B.

### 8.4 Algorithm Implementation

```python
def solve(self, flight_duals, crew_dual) -> SubproblemResult:
    # Initialize
    labels_at_node = {node: [] for node in self.network.nodes}
    heap = []  # Priority queue by cost

    # Start at SOURCE
    initial = Label(cost=0, node="SOURCE", path=("SOURCE",),
                   flight_time=0, start_timestamp=0, end_timestamp=0)
    heappush(heap, initial)

    best_sink_label = None

    while heap:
        current = heappop(heap)

        # Skip if dominated
        if self._is_dominated(current, labels_at_node[current.node]):
            continue

        # Reached SINK?
        if current.node == "SINK":
            if best_sink_label is None or current.cost < best_sink_label.cost:
                best_sink_label = current
            continue

        # Extend to all neighbors
        for neighbor in self.network.get_successors(current.node):
            new_label = self._extend_label(current, neighbor, flight_duals)

            if new_label is None:  # Infeasible extension
                continue

            if not self._is_dominated(new_label, labels_at_node[neighbor]):
                labels_at_node[neighbor].append(new_label)
                heappush(heap, new_label)

    # Convert best path to pairing
    if best_sink_label:
        pairing = self._path_to_pairing(best_sink_label.path)
        reduced_cost = best_sink_label.cost - crew_dual
        return SubproblemResult(pairing, reduced_cost, ...)
```

### 8.5 Label Extension

```python
def _extend_label(self, label, to_node, flight_duals) -> Optional[Label]:
    arc = self.network.get_arc(label.node, to_node)
    to_node_obj = self.network.nodes[to_node]

    # Compute arc cost with dual
    arc_cost = self.network.get_arc_cost(label.node, to_node, flight_duals)

    # Update flight time
    new_flight_time = label.flight_time
    if arc.arc_type == "flight":
        new_flight_time += arc.duration.total_seconds() / 3600

    # Check flight time limit
    if new_flight_time > self.rules.max_flight_hours:
        return None  # Infeasible

    # Update timestamps (skip SOURCE/SINK artificial nodes)
    new_start = label.start_timestamp
    new_end = label.end_timestamp

    if to_node_obj.node_type not in ("source", "sink"):
        new_end = to_node_obj.time.timestamp()
        if new_start == 0 and to_node_obj.node_type == "departure":
            new_start = to_node_obj.time.timestamp()

    # Check duty time limit
    if new_start > 0 and new_end > 0:
        duty = (new_end - new_start) / 3600
        if duty > self.crew.max_duty_hours:
            return None  # Infeasible

    return Label(
        cost=label.cost + arc_cost,
        node=to_node,
        path=label.path + (to_node,),
        flight_time=new_flight_time,
        start_timestamp=new_start,
        end_timestamp=new_end
    )
```

---

## 9. Test Instances

### 9.1 Micro-Airline (8 flights, 4 crew)

**File:** `data/generators/micro_airline.py`

**Design:**
- 4 round-trip pairings, one per crew
- JFK: 2 crew (C1, C2)
- LAX: 1 crew (C3)
- SFO: 1 crew (C4)

**Flights:**
| ID | Route | Duration | Crew |
|----|-------|----------|------|
| F1 | JFK→LAX | 3h | C1 |
| F2 | LAX→JFK | 3h | C1 |
| F3 | JFK→ORD | 2h | C2 |
| F4 | ORD→JFK | 2h | C2 |
| F5 | LAX→SFO | 1.5h | C3 |
| F6 | SFO→LAX | 1.5h | C3 |
| F7 | SFO→JFK | 4h | C4 |
| F8 | JFK→SFO | 4h | C4 |

**Optimal Solution:** $2,165.00

### 9.2 Small-Airline (24 flights, 8 crew)

**File:** `data/generators/small_airline.py`

**Design:**
- 8 three-flight pairings
- 4 hubs: JFK, LAX, ORD, SFO (2 crew each)
- Each crew has regional routes

**Hubs and Routes:**

| Hub | Crew | Route | Flight Time |
|-----|------|-------|-------------|
| JFK | C1 | JFK→BOS→PHL→JFK | 4h |
| JFK | C2 | JFK→ORD→DCA→JFK | 4.5h |
| LAX | C3 | LAX→PHX→SAN→LAX | 2.5h |
| LAX | C4 | LAX→SFO→SEA→LAX | 6h |
| ORD | C5 | ORD→DTW→CLE→ORD | 2.75h |
| ORD | C6 | ORD→MSP→DEN→ORD | 5.5h |
| SFO | C7 | SFO→PDX→SJC→SFO | 3.75h |
| SFO | C8 | SFO→LAS→OAK→SFO | 3.25h |

**Optimal Solution:** $4,004.00

**Optimization Insight:** The solver assigns cheaper crews to longer duties:
- C1 ($60/hr) gets 6h duty vs C2 ($55/hr) gets 6.5h duty
- Saves $2.50 compared to swapped assignment

---

## 10. Usage Guide

### 10.1 Command Line Interface

```bash
# Activate environment
conda activate crew3d

# Run micro-airline instance
python -m api.cli.main --instance micro_airline

# Run small-airline instance
python -m api.cli.main --instance small_airline

# Save solution to file
python -m api.cli.main --instance small_airline --output solution.json

# Quiet mode (no instance summary)
python -m api.cli.main --instance small_airline --quiet

# Custom iteration limit
python -m api.cli.main --instance small_airline --max-iterations 50
```

### 10.2 Programmatic Usage

```python
from data.generators.small_airline import generate_small_airline
from optimization.column_generation import ColumnGeneration
from optimization.subproblem.exact_rcspp import ExactRCSPP

# Generate instance
flights, crew, rules = generate_small_airline()

# Create solver
cg = ColumnGeneration(
    flights=flights,
    crew=crew,
    rules=rules,
    subproblem_class=ExactRCSPP,
    max_iterations=100,
    tolerance=1e-6
)

# Solve
solution = cg.run(verbose=True)

# Access results
print(f"Total Cost: ${solution.statistics.total_cost:.2f}")
for crew_id, assignment in solution.assignments.items():
    print(f"{crew_id}: {[f.id for f in assignment.pairing.flights]}")
```

### 10.3 Solution Output Format

```json
{
  "statistics": {
    "total_cost": 4004.0,
    "solve_time_seconds": 0.05,
    "iterations": 1,
    "columns_generated": 16
  },
  "assignments": {
    "C1": {
      "crew_id": "C1",
      "pairing_id": "P_C1_001",
      "flights": ["F1", "F2", "F3"],
      "cost": 530.0,
      "duty_hours": 6.0
    }
  }
}
```

---

## 11. Optimality Guarantees

### 11.1 Why Column Generation Finds Optimal Solutions

1. **LP Relaxation Bound:**
   - The LP relaxation provides a lower bound on the optimal integer solution
   - Column generation solves the LP to optimality

2. **Reduced Cost Optimality:**
   - When no negative reduced cost columns exist, LP is optimal
   - The pricing subproblem proves this by exhaustive search

3. **Integer Solution:**
   - If LP solution is integer → optimal MIP solution
   - Otherwise, MIP solve with all generated columns

### 11.2 Proof of Optimality

For the small-airline instance:

```
LP Optimal Value: $4,004.00
MIP Solution:     $4,004.00  (same → no integrality gap)
Reduced Costs:    All ≥ 0   (no improving columns)
```

The solution is **provably optimal** because:
1. Column generation converged (no negative reduced costs)
2. LP and MIP values match (no integrality gap)

### 11.3 When Optimality May Not Hold

- **Time limit reached** before convergence
- **Numerical tolerance issues** (very small reduced costs)
- **Heuristic pricing** (if used instead of exact RCSPP)

---

## 12. Future Extensions

### 12.1 GNN-Accelerated Pricing

Replace or augment exact RCSPP with Graph Neural Network:

```
┌─────────────────────────────────────────┐
│         GNN Pricing Module              │
├─────────────────────────────────────────┤
│  Input: FlightNetwork + Dual Values     │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Graph Convolution Layers       │   │
│  │  - Node features: time, cost    │   │
│  │  - Edge features: connection    │   │
│  └─────────────────────────────────┘   │
│                  ↓                      │
│  ┌─────────────────────────────────┐   │
│  │  Path Probability Prediction    │   │
│  └─────────────────────────────────┘   │
│                  ↓                      │
│  Output: Top-k candidate paths         │
│  → Verify with exact feasibility check │
└─────────────────────────────────────────┘
```

### 12.2 Multi-Day Pairings

Extend to pairings spanning multiple days with rest periods:

```python
@dataclass
class MultiDayPairing:
    duties: List[Duty]      # Each duty is a single-day pairing
    rest_periods: List[timedelta]
    hotel_costs: List[float]
```

### 12.3 Robust Optimization

Handle uncertainty in flight times:

```python
class RobustPairing:
    def slack_time(self) -> timedelta:
        """Buffer time for delays."""
        pass

    def delay_propagation_risk(self) -> float:
        """Probability of missing connections."""
        pass
```

### 12.4 Larger Instances

For 100+ flights:
- Parallel pricing (solve subproblems concurrently)
- Column pool management (remove dominated columns)
- Stabilization techniques (prevent dual oscillation)

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Pairing** | A sequence of flights assigned to one crew member |
| **Duty Period** | Wall-clock time from first departure to last arrival |
| **Flight Time** | Actual time spent flying (block hours) |
| **Connection** | Time between flights at same airport |
| **Base** | Crew member's home airport |
| **Reduced Cost** | Pairing cost minus dual values; negative = improving |
| **Column** | A variable in the master problem (represents one pairing) |
| **Dual Value** | Shadow price from LP relaxation |
| **RCSPP** | Resource-Constrained Shortest Path Problem |

---

## Appendix B: Algorithm Complexity

| Component | Complexity |
|-----------|------------|
| Master Problem (LP) | O(n²) per solve with n columns |
| RCSPP (per crew) | O(L × A) where L=labels, A=arcs |
| Label dominance | O(L²) worst case |
| Total CG iteration | O(K × L × A) where K=crews |

Practical performance:
- Micro-airline: ~50ms total
- Small-airline: ~50ms total
- 100 flights: estimated 1-5 seconds

---

*Document Version: 1.0*
*Last Updated: 2024-01-01*
