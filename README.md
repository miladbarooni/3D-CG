# Crew Scheduling 3D

Exact column generation for the **3D integrated airline crew scheduling problem** — pairings are generated *per crew member*, so flight coverage, crew assignment, and time feasibility are solved as one problem instead of the usual two-stage pairing-then-rostering pipeline.

Pure Python, no commercial solver required (PuLP + CBC). A GNN-accelerated pricing module is planned but not yet implemented.

---

## The 3D idea

Classical crew scheduling is solved in two stages: first build anonymous *pairings* (legal flight sequences), then assign crew to them. That decomposition throws away information — a pairing that looks cheap in stage 1 may be expensive or infeasible for whoever actually gets it in stage 2.

Here a pairing carries a `crew_id` from the moment it is created ([models/pairing.py](models/pairing.py)). The three dimensions are:

| Dimension | Meaning |
|-----------|---------|
| **Flights** | what must be covered |
| **Crew** | who can fly it (base, qualification, hourly cost) |
| **Time** | when it happens (connections, duty windows, rest) |

Each crew member gets their **own time-space network** rooted at their home base ([models/network.py](models/network.py)) and their **own pricing subproblem**, so cost and legality are crew-specific throughout.

## Mathematical formulation

Set-partitioning master problem over crew-indexed pairings:

```
min   Σ_k Σ_{p∈P_k}  c_pk · x_pk

s.t.  Σ_k Σ_{p∈P_k}  a_ip · x_pk = 1     ∀ i ∈ F     (each flight covered exactly once)
      Σ_{p∈P_k}      x_pk        = 1     ∀ k ∈ K     (each crew flies exactly one pairing)
                     x_pk ∈ {0,1}
```

Dual values from the LP relaxation — `π_i` per flight, `σ_k` per crew — drive the pricing problem:

```
reduced_cost(p, k) = c_pk − Σ_{i∈p} π_i − σ_k
```

`c_pk = Σ_{i∈p} base_cost_i + hourly_cost_k × duty_hours(p)`. The column generation loop terminates when no crew has a pairing of negative reduced cost.

> **Known issue:** the pricer currently scores arcs with `base_cost_i − π_i` only — the `hourly_cost_k × duty_hours` term is missing, so its reduced costs are understated by the full crew labor cost and termination does *not* prove LP optimality. See [Status and roadmap](#status-and-roadmap).

## Architecture

```
main.py  →  api/cli/main.py
                 │
                 ▼
     optimization/column_generation.py        ColumnGeneration — the CG loop
                 │
        ┌────────┴─────────┐
        ▼                  ▼
  master_problem.py   subproblem/exact_rcspp.py
  (PuLP set-           (label-setting RCSPP,
   partitioning LP/     one instance per crew)
   MIP, dual                    │
   extraction)                  ▼
                        models/network.py
                        (crew-specific time-space graph)
```

| Module | Contents |
|--------|----------|
| [models/flight.py](models/flight.py) | `Flight`, `AircraftType`, connection legality |
| [models/crew.py](models/crew.py) | `Crew`, `CrewRank`, `Qualification`, cost model |
| [models/pairing.py](models/pairing.py) | `Pairing` — crew-indexed flight sequence, `is_legal()`, `compute_cost()` |
| [models/legal_rules.py](models/legal_rules.py) | `LegalRules` — connection, duty, flight-time, base rules |
| [models/network.py](models/network.py) | `FlightNetwork` — per-crew time-space DAG (source → departures/arrivals → sink) |
| [models/solution.py](models/solution.py) | `Solution`, `CrewAssignment`, `SolutionStatistics`, constraint verification |
| [optimization/master_problem.py](optimization/master_problem.py) | Restricted master problem, column pool, dual extraction |
| [optimization/column_generation.py](optimization/column_generation.py) | Initialization, CG iterations, final MIP solve |
| [optimization/subproblem/exact_rcspp.py](optimization/subproblem/exact_rcspp.py) | Label-setting RCSPP pricer with dominance (see known issues) |
| [data/generators/](data/generators/) | Micro- and small-airline benchmark instances |
| [gnn/](gnn/), [visualization/](visualization/), [config/](config/) | Placeholders for planned features |

### How the loop runs

1. **Initialize** — enumerate every legal 2-, 3-, and 4-leg round trip per crew so the master problem starts feasible; any still-uncovered flight triggers an extra RCSPP call with an inflated dual.
2. **Master** — solve the LP relaxation over the current column pool, read off `π` and `σ`.
3. **Price** — for each crew, run the exact RCSPP on their network with arc costs `base_cost_i − π_i`; a path reaching the sink with cost below `σ_k` is an improving column.
4. **Add & repeat** — add up to `max_columns_per_iter` columns; stop when an iteration adds none (convergence proof) or `max_iterations` is hit.
5. **Integralize** — re-solve as a binary MIP over all generated columns and verify constraints.

The pricer is a label-setting algorithm. Labels track `(cost, flight_time, start/end timestamp)`; extensions are pruned by max flight hours and max duty hours, and dominated labels are discarded rather than expanded.

> **Known issue:** the dominance test compares cost, flight time, and *end* timestamp, but not `start_timestamp` — i.e. not remaining duty budget. Since every non-source node has a fixed event time, all labels at a node share an end timestamp, so a cheap-but-early label can dominate an expensive-but-late one and then die on the duty limit, losing a feasible pairing. The algorithm is therefore not exact as written.

## Installation

Requires Python ≥ 3.10.

```bash
cd crew_scheduling_3d
python -m venv .venv && source .venv/bin/activate
pip install -e .              # or: pip install -r requirements.txt
```

Optional extras:

```bash
pip install -e ".[dev]"       # pytest, black, isort, mypy
pip install -e ".[gnn]"       # torch, torch-geometric (module not implemented yet)
```

CBC ships with PuLP, so no external solver install is needed. Gurobi and CPLEX are used automatically if available and requested via `MasterProblem.solve(solver=...)`, falling back to CBC otherwise.

## Usage

### Command line

```bash
python main.py                                    # micro-airline (default)
python main.py --instance small_airline
python main.py --instance small_airline --output outputs/solution.json
python main.py --max-iterations 50 --log-level DEBUG --log-file logs/solver.log
python main.py -q                                 # skip the instance summary
```

After `pip install -e .` the same entry point is available as `crew-schedule`.

Output is the instance summary, per-iteration objective and column count, the final assignment per crew, and a constraint verification block (coverage, base return, duty limits).

### Programmatic

```python
from data.generators.micro_airline import generate_micro_airline
from optimization.column_generation import ColumnGeneration
from optimization.subproblem.exact_rcspp import ExactRCSPP

flights, crew, rules = generate_micro_airline()

cg = ColumnGeneration(
    flights=flights,
    crew=crew,
    rules=rules,
    subproblem_class=ExactRCSPP,
    max_iterations=100,
    tolerance=1e-6,
    max_columns_per_iter=20,
)

solution = cg.run(verbose=True)

print(f"Total cost: ${solution.statistics.total_cost:,.2f}")
for crew_id, a in solution.assignments.items():
    print(crew_id, [f.id for f in a.pairing.flights], f"{a.duty_hours:.1f}h")

print(solution.verify_constraints(flights, crew))
```

Pass `callback=` to `ColumnGeneration` to receive an `IterationResult` after every iteration (objective, columns added, best reduced cost, timing) — useful for plotting convergence.

### Custom instances

Build `Flight` and `Crew` lists plus a `LegalRules` object and hand them to `ColumnGeneration`; the generators in [data/generators/](data/generators/) are the reference pattern.

```python
from datetime import timedelta
from models import LegalRules

rules = LegalRules(
    min_connection_time=timedelta(hours=1),
    max_connection_time=timedelta(hours=4),
    max_duty_period=timedelta(hours=10),
    max_flight_time_per_duty=timedelta(hours=8),
    max_flights_per_duty=4,
)
```

### Custom pricing

Subclass `PricingSubproblem` ([optimization/subproblem/base.py](optimization/subproblem/base.py)), implement `solve(flight_duals, crew_dual, time_limit_ms) -> SubproblemResult`, and pass the class as `subproblem_class`. This is the hook the planned GNN pricer targets.

## Benchmark instances

| Instance | Flights | Crew | Bases | Reference optimum |
|----------|---------|------|-------|-------------------|
| `micro_airline` | 8 | 4 | JFK, LAX, SFO | $2,165.00 |
| `small_airline` | 24 | 8 | JFK, LAX, ORD, SFO | $4,004.00 |

Both are designed so a feasible partition exists by construction (one dedicated pairing per crew), which makes the optimal objective easy to check.

> **Caveat:** because `initialize()` exhaustively enumerates every legal pairing up to `max_flights_per_duty` (4), and both instances admit only 6 and 16 legal pairings respectively, the entire column pool is built before the first iteration. Pricing adds nothing and the loop reports convergence at iteration 1. These benchmarks exercise the master problem and the data model, but **not** column generation.

## Testing

```bash
pytest                     # runs with coverage, per pyproject config
pytest tests/unit          # model-level tests
pytest tests/integration   # convergence, constraint satisfaction, network structure
```

Shared fixtures (simple two-flight round trip, micro-airline instance, default rules) live in [tests/conftest.py](tests/conftest.py).

[debug_pairings.py](debug_pairings.py) traces the RCSPP label expansion step by step on the micro instance — handy when a pricer change stops producing columns.

## Configuration

[config/default.yaml](config/default.yaml) holds problem, solver, column generation, GNN, logging, and output settings. It documents the intended knobs, but the CLI currently constructs the solver from its own arguments and defaults — the YAML is not yet loaded at runtime.

## Status and roadmap

Working today: data models, per-crew time-space networks, the set-partitioning master with dual extraction, integer recovery, constraint verification, CLI, and tests. Both benchmark instances solve to their documented optima and pass all constraint checks.

**Correctness work needed before the solver can be trusted on new instances:**

- **Pricer ignores crew cost.** `FlightNetwork.get_arc_cost()` returns `base_cost − π` for flight arcs and 0 otherwise, so `hourly_cost × duty_hours` never enters the label cost. Reduced costs are understated by exactly that amount. Fix: charge `hourly_cost_k × arc_duration` on flight *and* connection arcs — duty time is wall-clock from first departure to last arrival, so it decomposes additively along the path.
- **Dominance rule drops feasible paths.** Add `start_timestamp` (remaining duty budget) to `Label.dominates()` as a resource that must also be no worse.
- **Crew constraint is an equality.** `Σ_p x_pk = 1` forces *every* crew member to fly, so any instance without a perfect crew-to-pairing partition is infeasible. Relax to `≤ 1` for realistic instances.
- **Initialization defeats column generation.** The exhaustive 2/3/4-leg enumeration in `initialize()` is O(n⁴) and, at `max_flights_per_duty = 4`, enumerates the complete feasible pairing set. Seed with a small artificial or greedy basis instead and let pricing do the work.

Not yet implemented:

- **GNN-accelerated pricing** ([gnn/](gnn/) is an empty package) — learn to propose top-k candidate paths, then verify exactly, to cut pricing time on large instances.
- **Visualization** ([visualization/](visualization/) is an empty package) — Gantt charts of assignments, convergence plots.
- **Config loading** — wire `config/default.yaml` into the CLI.
- **Multi-day pairings** with overnight rest, deadheading, and reserve crew. `LegalRules` already carries `min_rest_period` and `deadhead_cost_factor` fields that nothing consumes yet.
- **Branch-and-price** for instances where the LP solution is fractional; the current final step solves a MIP over the generated columns only, which is optimal for these benchmarks but not in general.

## Further reading

[DOCUMENTATION.md](DOCUMENTATION.md) is the full technical write-up: derivations, dominance rules, worked algorithm traces, and instance design notes. Note that it refers to `models/rules.py`, which is now [models/legal_rules.py](models/legal_rules.py).

[docs/architecture.md](docs/architecture.md) is the original system design document. Treat it as a **design record, not a description of the current code** — it predates the implementation and the two diverged. Most usefully it carries a full sketch of the unbuilt GNN pricing module (`gat_pricing.py`, `trainer.py`, `gnn_guided.py`), which is the starting point if you pick that work up. That sketch has never been run and calls at least one method that does not exist (`FlightNetwork.to_pyg_data`).

## License

MIT.
