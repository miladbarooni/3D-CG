"""Integration tests for column generation."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from optimization.column_generation import ColumnGeneration
from optimization.subproblem.exact_rcspp import ExactRCSPP


class TestColumnGeneration:
    """Integration tests for column generation algorithm."""

    def test_simple_instance_converges(self, simple_flights, simple_crew, default_rules):
        """Test that a simple instance converges."""
        cg = ColumnGeneration(
            flights=simple_flights,
            crew=simple_crew,
            rules=default_rules,
            subproblem_class=ExactRCSPP,
            max_iterations=50
        )

        solution = cg.run(verbose=False)

        # Should converge quickly
        assert cg.num_iterations < 50

        # Should find a solution
        assert solution is not None
        assert solution.is_feasible

    def test_micro_airline_converges(self, micro_airline):
        """Test that micro-airline instance converges."""
        flights, crew, rules = micro_airline

        cg = ColumnGeneration(
            flights=flights,
            crew=crew,
            rules=rules,
            subproblem_class=ExactRCSPP,
            max_iterations=50
        )

        solution = cg.run(verbose=False)

        # Should converge
        assert cg.num_iterations < 50

        # Should be feasible
        assert solution.is_feasible

    def test_micro_airline_constraints_satisfied(self, micro_airline):
        """Test that all constraints are satisfied for micro-airline."""
        flights, crew, rules = micro_airline

        cg = ColumnGeneration(flights, crew, rules)
        solution = cg.run(verbose=False)

        verification = solution.verify_constraints(flights, crew)

        assert verification["all_flights_covered"], "Not all flights covered"
        assert verification["each_flight_once"], "Some flights covered multiple times"
        assert verification["all_crew_assigned"], "Not all crew assigned"
        assert verification["base_constraints"], "Base constraints violated"
        assert verification["duty_limits"], "Duty limits violated"

    def test_objective_decreases(self, micro_airline):
        """Test that objective value decreases during optimization."""
        flights, crew, rules = micro_airline

        objectives = []

        def callback(result):
            objectives.append(result.objective)

        cg = ColumnGeneration(
            flights=flights,
            crew=crew,
            rules=rules,
            callback=callback,
            max_iterations=50
        )

        cg.run(verbose=False)

        # Objective should generally decrease (or stay same)
        # Check that final is <= initial
        if len(objectives) > 1:
            assert objectives[-1] <= objectives[0]


class TestFlightNetwork:
    """Tests for the flight network construction."""

    def test_network_has_source_sink(self, simple_flights, simple_crew, default_rules):
        """Test that network has source and sink nodes."""
        from models import FlightNetwork

        crew = simple_crew[0]
        network = FlightNetwork(simple_flights, crew, default_rules)

        assert "SOURCE" in network.nodes
        assert "SINK" in network.nodes

    def test_network_has_flight_nodes(self, simple_flights, simple_crew, default_rules):
        """Test that network has flight departure and arrival nodes."""
        from models import FlightNetwork

        crew = simple_crew[0]
        network = FlightNetwork(simple_flights, crew, default_rules)

        for flight in simple_flights:
            assert f"{flight.id}_DEP" in network.nodes
            assert f"{flight.id}_ARR" in network.nodes

    def test_network_source_connects_to_base_flights(self, micro_airline):
        """Test that source connects only to flights departing from crew base."""
        from models import FlightNetwork

        flights, crew, rules = micro_airline

        for c in crew:
            network = FlightNetwork(flights, c, rules)
            source_successors = network.get_successors("SOURCE")

            for succ in source_successors:
                # Each successor should be a departure node at crew's base
                node = network.nodes[succ]
                assert node.airport == c.base
