from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from psd_snn.analysis.dynamics.runner import analyze_dynamics


def _load_json_arg(value: str | None) -> dict[str, Any]:
    """Load a small JSON object from a literal string or a file path."""
    if value is None:
        return {}
    candidate = Path(value)
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    else:
        loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("dynamics JSON inputs must decode to an object")
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m psd_snn.cli.analyze_dynamics",
        description="Summarize already-collected dynamics parameter and internal-state statistics.",
    )
    parser.add_argument(
        "--parameter-stats-json",
        default=None,
        help="JSON object or path to a JSON object containing parameter statistics. Defaults to {}.",
    )
    parser.add_argument(
        "--internal-state-stats-json",
        default=None,
        help="JSON object or path to a JSON object containing internal-state statistics. Defaults to {}.",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print the DynamicsStats result as JSON instead of the dataclass repr.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    parameter_stats = _load_json_arg(args.parameter_stats_json)
    internal_state_stats = _load_json_arg(args.internal_state_stats_json)
    stats = analyze_dynamics(parameter_stats, internal_state_stats)
    if args.as_json:
        print(json.dumps(asdict(stats), ensure_ascii=False, sort_keys=True))
    else:
        print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
