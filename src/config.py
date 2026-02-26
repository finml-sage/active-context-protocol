"""Configuration for the Active Context Protocol.

Loads settings from a YAML config file (``~/.acp/config.yaml`` by default),
falling back to sensible defaults when the file is missing or incomplete.
Partial configs are merged with defaults so users only need to specify
the values they want to override.

YAML parsing: tries ``import yaml`` (PyYAML) first.  If unavailable, falls
back to a minimal YAML-subset parser that handles the flat key-value format
used by the default config template.  This keeps the zero-dependency promise
for environments without PyYAML installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("~/.acp/config.yaml")


# ---------------------------------------------------------------------------
# AcpConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class AcpConfig:
    """All ACP runtime settings.

    Every field has a sensible default so the system works out of the box.
    """

    # Token monitoring
    token_threshold: int = 70_000
    polling_interval: int = 30  # seconds

    # Delivery
    warmdown_interval: int = 120  # seconds between reminders (global)
    grace_period: int = 300  # seconds after session start before any reminders
    tmux_session: str = ""  # target tmux session name
    log_file: str = "~/.acp/acp.log"

    # Triggers — compaction
    compaction_enabled: bool = True
    compaction_threshold: int = 70_000
    compaction_cooldown: int = 120

    # Triggers — memory filing
    memory_filing_enabled: bool = True
    memory_filing_grace_after_event: int = 60
    memory_filing_patterns: list[str] = field(default_factory=list)

    # Idle detection
    idle_threshold: float = 5.0


# ---------------------------------------------------------------------------
# Minimal YAML-subset parser (fallback when PyYAML is not installed)
# ---------------------------------------------------------------------------


def _parse_yaml_minimal(text: str) -> dict[str, Any]:
    """Parse a minimal YAML subset into a flat dict.

    Supports:
    - ``key: value`` pairs (strings, ints, floats, booleans)
    - Comments (``#``) and blank lines
    - Nested sections (one level, with 2-space indent)
    - Lists written as ``[item1, item2]`` (inline) or ``[]`` (empty)
    - Quoted strings (single or double)

    Does NOT support multi-line values, anchors, tags, or block-style lists.
    """
    result: dict[str, Any] = {}
    current_section: str | None = None

    for raw_line in text.splitlines():
        # Strip inline comments (but not inside quotes)
        line = raw_line.split("#")[0].rstrip() if "#" in raw_line else raw_line.rstrip()

        if not line.strip():
            continue

        # Detect indentation
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if ":" not in stripped:
            continue

        key, _, val = stripped.partition(":")
        key = key.strip()
        val = val.strip()

        if indent == 0:
            if val == "" or val == "":
                # Section header (nested block follows)
                current_section = key
                continue
            current_section = None
            result[key] = _coerce_value(val)
        elif indent >= 2 and current_section is not None:
            # Nested key — flatten to section_key
            flat_key = f"{current_section}_{key}"
            result[flat_key] = _coerce_value(val)
        else:
            result[key] = _coerce_value(val)

    return result


def _coerce_value(val: str) -> Any:
    """Coerce a string value to the appropriate Python type."""
    # Remove surrounding quotes
    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("'") and val.endswith("'")
    ):
        return val[1:-1]

    # Booleans
    if val.lower() in ("true", "yes"):
        return True
    if val.lower() in ("false", "no"):
        return False

    # Empty list
    if val == "[]":
        return []

    # Inline list: [item1, item2]
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        items = [_coerce_value(item.strip()) for item in inner.split(",")]
        return items

    # Numbers
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass

    return val


# ---------------------------------------------------------------------------
# YAML loading (PyYAML with fallback)
# ---------------------------------------------------------------------------


def _load_yaml(text: str) -> dict[str, Any]:
    """Load YAML text into a dict, using PyYAML if available."""
    try:
        import yaml

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except ImportError:
        return _parse_yaml_minimal(text)


# ---------------------------------------------------------------------------
# Config flattening (nested YAML -> flat AcpConfig fields)
# ---------------------------------------------------------------------------


_NESTED_MAP: dict[str, dict[str, str]] = {
    "compaction": {
        "enabled": "compaction_enabled",
        "threshold": "compaction_threshold",
        "cooldown": "compaction_cooldown",
    },
    "memory_filing": {
        "enabled": "memory_filing_enabled",
        "grace_after_event": "memory_filing_grace_after_event",
        "patterns": "memory_filing_patterns",
    },
}


def _flatten_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested YAML sections into flat AcpConfig field names.

    For example::

        {"compaction": {"enabled": True, "threshold": 80000}}

    becomes::

        {"compaction_enabled": True, "compaction_threshold": 80000}

    Top-level keys that are not nested sections are passed through as-is.
    """
    flat: dict[str, Any] = {}

    for key, value in raw.items():
        if key in _NESTED_MAP and isinstance(value, dict):
            # Nested section — expand
            for sub_key, sub_val in value.items():
                flat_key = _NESTED_MAP[key].get(sub_key)
                if flat_key is not None:
                    flat[flat_key] = sub_val
                else:
                    logger.warning(
                        "Unknown config key: %s.%s (ignoring)", key, sub_key
                    )
        else:
            flat[key] = value

    return flat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: Path | None = None) -> AcpConfig:
    """Load ACP configuration from a YAML file.

    Args:
        path: Path to the YAML config file.  Defaults to
            ``~/.acp/config.yaml``.  If the file does not exist,
            returns the default configuration.

    Returns:
        An :class:`AcpConfig` instance with settings from the file merged
        over the defaults.  Unknown keys are silently ignored.
    """
    config_path = (path or _DEFAULT_CONFIG_PATH).expanduser()

    if not config_path.is_file():
        logger.debug("Config file not found at %s — using defaults", config_path)
        return AcpConfig()

    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read config file %s: %s — using defaults", config_path, exc)
        return AcpConfig()

    raw = _load_yaml(text)
    flat = _flatten_config(raw)

    # Build config from defaults, overriding with file values
    valid_fields = {f.name for f in fields(AcpConfig)}
    kwargs: dict[str, Any] = {}

    for key, value in flat.items():
        if key in valid_fields:
            kwargs[key] = value
        else:
            logger.debug("Ignoring unknown config key: %s", key)

    return AcpConfig(**kwargs)


def save_default_config(path: Path) -> None:
    """Write the default configuration template to a file.

    Creates parent directories as needed.  The file includes comments
    explaining each setting.

    Args:
        path: Destination path for the config file.
    """
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    template = _get_default_config_template()
    path.write_text(template, encoding="utf-8")
    logger.info("Default config written to %s", path)


def _get_default_config_template() -> str:
    """Return the commented default config as a string."""
    return """\
# Active Context Protocol Configuration
# https://github.com/finml-sage/active-context-protocol

# Token monitoring — how often to check and when to alert
token_threshold: 70000        # Total context tokens before compaction reminder
polling_interval: 30          # Seconds between monitoring cycles

# Delivery — how reminders reach the agent
warmdown_interval: 120        # Minimum seconds between any two reminders (global)
grace_period: 300             # Seconds after session start before any reminders fire
tmux_session: ""              # Target tmux session name (required for delivery)
log_file: "~/.acp/acp.log"   # Audit trail for all delivery attempts

# Idle detection
idle_threshold: 5.0           # Seconds since last JSONL write to consider agent idle

# Compaction trigger
compaction:
  enabled: true               # Whether compaction reminders are active
  threshold: 70000            # Token count that triggers compaction reminder
  cooldown: 120               # Seconds between compaction reminders

# Memory filing trigger
memory_filing:
  enabled: true               # Whether memory filing reminders are active
  grace_after_event: 60       # Seconds to wait after milestone before reminding
  patterns: []                # Additional regex patterns for milestone detection
"""
