"""Core data models for the 3D Crew Scheduling system."""

from models.flight import Flight, AircraftType
from models.crew import Crew, CrewRank, Qualification
from models.pairing import Pairing
from models.legal_rules import LegalRules
from models.solution import Solution, SolutionStatistics, CrewAssignment
from models.network import FlightNetwork, NetworkNode, NetworkArc

__all__ = [
    "Flight",
    "AircraftType",
    "Crew",
    "CrewRank",
    "Qualification",
    "Pairing",
    "LegalRules",
    "Solution",
    "SolutionStatistics",
    "CrewAssignment",
    "FlightNetwork",
    "NetworkNode",
    "NetworkArc",
]
