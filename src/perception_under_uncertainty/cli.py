"""
start a smoke run from a configuration file and print the saved record's path.

input and filesystem errors go to standard error with exit code 2.
"""

import argparse
from pathlib import Path

from .runner import run_smoke


def main(argv: list[str] | None = None) -> int:
    """
    read the supplied arguments, or use the process arguments by default.

    return zero on success. input and file errors exit with code 2, while
    unexpected errors keep their traceback for debugging.
    """
    parser = argparse.ArgumentParser(
        prog="ppu", description="check the setup for perception under uncertainty"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run the data-free setup check")
    run_parser.add_argument(
        "--config", type=Path, required=True, help="path to the smoke configuration"
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        help="override the output root; each run gets a new directory inside it",
    )
    arguments = parser.parse_args(argv)

    if arguments.command == "run":
        try:
            destination = run_smoke(arguments.config, arguments.output_dir)
        except (OSError, ValueError) as error:
            parser.exit(2, f"configuration or run error: {error}\n")
        print("smoke check passed; no model evaluation was performed")
        print(f"manifest: {destination}")
        return 0

    return 1
