#!/usr/bin/env python3
"""
cli.py — GRAVITAS Engine CLI entry point.

Provides a unified command-line interface for running scenarios,
listing available scenarios and plugins, and running evaluations.

Usage:
    # Run a scenario with default plugins
    python cli.py run stalingrad --episodes 30

    # Run from a config file
    python cli.py run --config configs/custom.yaml --episodes 10

    # List available scenarios
    python cli.py list scenarios

    # List available plugins
    python cli.py list plugins

    # Run with trained agents
    python cli.py run stalingrad --episodes 30 \\
        --axis-model logs/stalingrad_selfplay_volga/axis_final.zip \\
        --soviet-model logs/stalingrad_selfplay_volga/soviet_final.zip

    # Run with specific plugins
    python cli.py run stalingrad --plugins soviet_reinforcements axis_airlift
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))


def cmd_run(args: argparse.Namespace) -> None:
    """Run a simulation scenario."""
    from gravitas.engine import GravitasEngine

    if args.config:
        engine = GravitasEngine.from_config(args.config, seed=args.seed)
    else:
        scenario = args.scenario or "stalingrad"
        engine = GravitasEngine(
            scenario=scenario,
            seed=args.seed,
            max_steps=args.max_steps,
            verbose=not args.quiet,
        )

        # Load plugins
        plugin_names = args.plugins or []
        if plugin_names:
            engine.load_plugins(plugin_names)

    # Load trained agents if provided
    axis_model = None
    soviet_model = None

    if args.axis_model:
        try:
            from sb3_contrib import RecurrentPPO
            axis_model = RecurrentPPO.load(args.axis_model, device=args.device)
            print(f"  Loaded Axis model: {args.axis_model}")
        except ImportError:
            print("ERROR: sb3-contrib required for trained agents")
            sys.exit(1)

    if args.soviet_model:
        try:
            from sb3_contrib import RecurrentPPO
            soviet_model = RecurrentPPO.load(args.soviet_model, device=args.device)
            print(f"  Loaded Soviet model: {args.soviet_model}")
        except ImportError:
            print("ERROR: sb3-contrib required for trained agents")
            sys.exit(1)

    # Run
    print(f"\n  ⚔️  GRAVITAS Engine — {engine.scenario_name.upper()}")
    print(f"  Plugins: {[p.name for p in engine.plugins] or ['none']}")
    print(f"  Episodes: {args.episodes}, Seed: {args.seed}")
    print()

    results = engine.run(
        episodes=args.episodes,
        seed=args.seed,
        axis_model=axis_model,
        soviet_model=soviet_model,
    )

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Make results JSON-serializable
        serializable = []
        for r in results:
            sr = {k: v for k, v in r.items() if k != "plugin_events"}
            sr["plugin_events"] = [
                {k: v for k, v in e.items() if k != "data"}
                for e in r.get("plugin_events", [])
            ]
            serializable.append(sr)

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")


def cmd_list(args: argparse.Namespace) -> None:
    """List available scenarios or plugins."""
    if args.what == "scenarios":
        from gravitas.engine import list_scenarios
        scenarios = list_scenarios()
        print(f"\n  Available scenarios ({len(scenarios)}):")
        for s in scenarios:
            print(f"    - {s['name']:<25s}  {s['path']}")
        print()

    elif args.what == "plugins":
        from gravitas.plugins import discover_plugins
        plugins = discover_plugins()
        print(f"\n  Available plugins ({len(plugins)}):")
        for name, module in plugins.items():
            print(f"    - {name:<30s}  ({module})")
        print()

        # Try to show descriptions
        from gravitas.plugins import load_plugin
        print("  Plugin details:")
        for name in plugins:
            try:
                p = load_plugin(name)
                print(f"    {p.name}: {p.description}")
            except Exception as e:
                print(f"    {name}: (failed to load: {e})")
        print()

    else:
        print(f"Unknown list target: {args.what}")
        print("Use: python cli.py list scenarios|plugins")
        sys.exit(1)


def cmd_replay(args: argparse.Namespace) -> None:
    """Run a detailed battle replay (delegates to replay script)."""
    import subprocess
    cmd = [
        sys.executable, "tests/replay_stalingrad_battle.py",
        "--axis-model", args.axis_model,
        "--soviet-model", args.soviet_model,
        "--seed", str(args.seed),
        "--interval", str(args.interval),
    ]
    subprocess.run(cmd, cwd=str(_ROOT))


def main():
    parser = argparse.ArgumentParser(
        prog="gravitas",
        description="GRAVITAS Engine — Governance simulation with plugin support",
    )
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── run ──────────────────────────────────────────────────────── #
    p_run = subparsers.add_parser("run", help="Run a simulation scenario")
    p_run.add_argument("scenario", nargs="?", default=None,
                       help="Scenario name (e.g., 'stalingrad')")
    p_run.add_argument("--config", type=str, default=None,
                       help="Path to unified config YAML")
    p_run.add_argument("--episodes", type=int, default=1)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--max-steps", type=int, default=500)
    p_run.add_argument("--plugins", nargs="+", default=None,
                       help="Plugin names to activate")
    p_run.add_argument("--axis-model", type=str, default=None)
    p_run.add_argument("--soviet-model", type=str, default=None)
    p_run.add_argument("--device", type=str, default="cpu")
    p_run.add_argument("--output", type=str, default=None,
                       help="Save results JSON to this path")
    p_run.add_argument("--quiet", action="store_true")
    p_run.set_defaults(func=cmd_run)

    # ── list ─────────────────────────────────────────────────────── #
    p_list = subparsers.add_parser("list", help="List scenarios or plugins")
    p_list.add_argument("what", choices=["scenarios", "plugins"])
    p_list.set_defaults(func=cmd_list)

    # ── replay ───────────────────────────────────────────────────── #
    p_replay = subparsers.add_parser("replay", help="Detailed battle replay")
    p_replay.add_argument("--axis-model", type=str, required=True)
    p_replay.add_argument("--soviet-model", type=str, required=True)
    p_replay.add_argument("--seed", type=int, default=42)
    p_replay.add_argument("--interval", type=int, default=25)
    p_replay.set_defaults(func=cmd_replay)

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
