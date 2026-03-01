"""
Command-line interface for the Adaptive Memory-Driven Regime Engine.

Usage:
    python -m gravitas_engine.cli [command] [options]

Commands:
    run         Run a simulation and print summary statistics.
    validate    Run the full validation suite (exit 0 on pass).
    info        Print system information and default parameters.
"""

from __future__ import annotations

import argparse
import json
import sys

from .analysis.metrics import summary_statistics
from .analysis.validation import run_all_tests
from .core.factions import create_balanced_factions, create_dominant_factions
from .core.parameters import SystemParameters
from .core.state import RegimeState, SystemState
from .simulation.runner import SimulationRunner
from .systems.crisis_classifier import ClassifierThresholds


def _build_subparsers(parser: argparse.ArgumentParser) -> None:
    """Register all sub-commands on the root parser."""
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ run --
    run_p = sub.add_parser("run", help="Run a simulation")
    run_p.add_argument(
        "--n-factions", type=int, default=3, metavar="N",
        help="Number of factions [2–6] (default: 3)"
    )
    run_p.add_argument(
        "--steps", type=int, default=500, metavar="S",
        help="Number of simulation steps (default: 500)"
    )
    run_p.add_argument(
        "--dt", type=float, default=0.01, metavar="DT",
        help="Integration time step (default: 0.01)"
    )
    run_p.add_argument(
        "--seed", type=int, default=0, metavar="SEED",
        help="Random seed for reproducibility (default: 0)"
    )
    run_p.add_argument(
        "--dominant", type=int, default=None, metavar="IDX",
        help="If set, create one dominant faction at this index"
    )
    run_p.add_argument(
        "--dominant-power", type=float, default=0.6, metavar="P",
        help="Power share of dominant faction (default: 0.6)"
    )
    run_p.add_argument(
        "--json", action="store_true",
        help="Output summary statistics as JSON"
    )

    # -------------------------------------------------------------- validate --
    sub.add_parser("validate", help="Run the full validation suite")

    # ---------------------------------------------------------------- info --
    sub.add_parser("info", help="Print default parameters and system info")


def _cmd_run(args: argparse.Namespace) -> None:
    """Execute the run sub-command."""
    params = SystemParameters(
        n_factions=args.n_factions,
        max_steps=args.steps,
        dt=args.dt,
        seed=args.seed,
    )

    if args.dominant is not None:
        factions = create_dominant_factions(
            args.n_factions, args.dominant, args.dominant_power
        )
    else:
        factions = create_balanced_factions(args.n_factions)

    initial_state = RegimeState(
        factions=factions,
        system=SystemState.neutral(),
        step=0,
    )

    runner = SimulationRunner(params=params)
    trajectory = runner.run(
        initial_state=initial_state,
        n_steps=args.steps,
    )

    thresholds = ClassifierThresholds()
    stats = summary_statistics(trajectory, thresholds)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(f"Simulation completed: {int(stats['n_steps'])} steps")
        print(f"  Mean legitimacy:    {stats['mean_legitimacy']:.4f}")
        print(f"  Mean instability:   {stats['mean_instability']:.4f}")
        print(f"  Mean fragmentation: {stats['mean_fragmentation']:.4f}")
        print(f"  Mean volatility:    {stats['mean_volatility']:.4f}")
        print(f"  Mean exhaustion:    {stats['mean_exhaustion']:.4f}")
        print(f"  Max radicalization: {stats['max_radicalization']:.4f}")
        print(f"  Final legitimacy:   {stats['final_legitimacy']:.4f}")
        print(f"  Final exhaustion:   {stats['final_exhaustion']:.4f}")
        print(f"  Crisis fraction ≥CRISIS: {stats['crisis_fraction_crisis']:.4f}")


def _cmd_validate(_args: argparse.Namespace) -> None:
    """Execute the validate sub-command."""
    run_all_tests()
    print("All 10 validation tests passed.")


def _cmd_info(_args: argparse.Namespace) -> None:
    """Execute the info sub-command."""
    params = SystemParameters()
    print("Adaptive Memory-Driven Regime Engine")
    print("Default SystemParameters:")
    for name, value in params.to_dict().items():
        print(f"  {name}: {value}")


def main() -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="gravitas_engine",
        description="Adaptive Memory-Driven Regime Engine CLI",
    )
    _build_subparsers(parser)
    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "validate":
        _cmd_validate(args)
    elif args.command == "info":
        _cmd_info(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
