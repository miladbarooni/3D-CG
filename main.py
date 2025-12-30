#!/usr/bin/env python3
"""
3D Integrated Crew Scheduling with Column Generation

Main entry point for the crew scheduling solver.
"""

import sys
from pathlib import Path

# Ensure the package is in the path
sys.path.insert(0, str(Path(__file__).parent))

from api.cli.main import main

if __name__ == "__main__":
    main()
