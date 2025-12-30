"""Command-line interface for the crew scheduling solver."""

import argparse
import logging
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.generators.micro_airline import generate_micro_airline, print_instance_summary
from data.generators.small_airline import (
    generate_small_airline,
    print_instance_summary as print_small_summary
)
from optimization.column_generation import ColumnGeneration
from optimization.subproblem.exact_rcspp import ExactRCSPP


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    """Configure logging."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def run_micro_airline(
    max_iterations: int = 100,
    verbose: bool = True,
    output_file: str = None
) -> None:
    """Run the solver on the micro-airline instance."""
    logger = logging.getLogger(__name__)

    # Generate instance
    logger.info("Generating micro-airline instance...")
    flights, crew, rules = generate_micro_airline()

    if verbose:
        print_instance_summary(flights, crew, rules)

    # Create column generation solver
    logger.info("Initializing column generation solver...")
    cg = ColumnGeneration(
        flights=flights,
        crew=crew,
        rules=rules,
        subproblem_class=ExactRCSPP,
        max_iterations=max_iterations,
        tolerance=1e-6,
        max_columns_per_iter=20
    )

    # Solve
    logger.info("Starting optimization...")
    solution = cg.run(verbose=verbose)

    # Verify constraints
    verification = solution.verify_constraints(flights, crew)

    # Print solution
    solution.print_summary()

    # Print verification
    print("\nConstraint Verification:")
    print("-" * 40)
    for constraint, satisfied in verification.items():
        status = "PASS" if satisfied else "FAIL"
        print(f"  {constraint}: {status}")

    # Save solution if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(solution.to_dict(), f, indent=2)
        logger.info(f"Solution saved to {output_file}")

    return solution


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="3D Integrated Crew Scheduling with Column Generation"
    )

    parser.add_argument(
        "--instance",
        type=str,
        default="micro_airline",
        choices=["micro_airline", "small_airline"],
        help="Instance to solve (default: micro_airline)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum CG iterations (default: 100)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for solution JSON"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    # Run solver
    if args.instance == "micro_airline":
        run_micro_airline(
            max_iterations=args.max_iterations,
            verbose=not args.quiet,
            output_file=args.output
        )
    elif args.instance == "small_airline":
        run_small_airline(
            max_iterations=args.max_iterations,
            verbose=not args.quiet,
            output_file=args.output
        )


def run_small_airline(
    max_iterations: int = 100,
    verbose: bool = True,
    output_file: str = None
) -> None:
    """Run the solver on the small-airline instance."""
    logger = logging.getLogger(__name__)

    # Generate instance
    logger.info("Generating small-airline instance...")
    flights, crew, rules = generate_small_airline()

    if verbose:
        print_small_summary(flights, crew, rules)

    # Create column generation solver
    logger.info("Initializing column generation solver...")
    cg = ColumnGeneration(
        flights=flights,
        crew=crew,
        rules=rules,
        subproblem_class=ExactRCSPP,
        max_iterations=max_iterations,
        tolerance=1e-6,
        max_columns_per_iter=20
    )

    # Solve
    logger.info("Starting optimization...")
    solution = cg.run(verbose=verbose)

    # Verify constraints
    verification = solution.verify_constraints(flights, crew)

    # Print solution
    solution.print_summary()

    # Print verification
    print("\nConstraint Verification:")
    print("-" * 40)
    for constraint, satisfied in verification.items():
        status = "PASS" if satisfied else "FAIL"
        print(f"  {constraint}: {status}")

    # Save solution if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(solution.to_dict(), f, indent=2)
        logger.info(f"Solution saved to {output_file}")

    return solution


if __name__ == "__main__":
    main()
