"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file.

    Returns
    -------
    dict
        Nested configuration dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_project_root() -> Path:
    """Return the project root directory (parent of ``src/``).

    Walks up from this file until it finds ``pyproject.toml``.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


def apply_config_to_parser(
    parser: "argparse.ArgumentParser",
    config_path: str | Path,
) -> None:
    """Load a YAML config file and apply its values as argparse defaults.

    This enables a two-pass parsing pattern::

        args, _ = parser.parse_known_args()
        if args.config:
            apply_config_to_parser(parser, args.config)
        args = parser.parse_args()

    CLI arguments always override config file values because
    ``set_defaults`` only changes the default — any value the user
    provides on the command line takes precedence.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser whose defaults should be updated.
    config_path : str or Path
        Path to a flat YAML config file whose keys match argparse
        destination names.
    """
    config = load_config(config_path)
    if config:
        parser.set_defaults(**config)
