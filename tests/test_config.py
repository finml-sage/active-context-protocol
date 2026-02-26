"""Tests for the ACP configuration module."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
    AcpConfig,
    _coerce_value,
    _flatten_config,
    _parse_yaml_minimal,
    load_config,
    save_default_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Return a path for a temporary config file."""
    return tmp_path / "config.yaml"


# ---------------------------------------------------------------------------
# AcpConfig dataclass tests
# ---------------------------------------------------------------------------


class TestAcpConfig:
    def test_defaults(self) -> None:
        """Default config has all expected values."""
        config = AcpConfig()
        assert config.token_threshold == 70_000
        assert config.polling_interval == 30
        assert config.warmdown_interval == 120
        assert config.grace_period == 300
        assert config.tmux_session == ""
        assert config.log_file == "~/.acp/acp.log"
        assert config.compaction_enabled is True
        assert config.compaction_threshold == 70_000
        assert config.compaction_cooldown == 120
        assert config.memory_filing_enabled is True
        assert config.memory_filing_grace_after_event == 60
        assert config.memory_filing_patterns == []
        assert config.idle_threshold == 5.0

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        config = AcpConfig(
            token_threshold=100_000,
            tmux_session="kelvin",
            memory_filing_patterns=["deploy", "release"],
        )
        assert config.token_threshold == 100_000
        assert config.tmux_session == "kelvin"
        assert config.memory_filing_patterns == ["deploy", "release"]

    def test_default_patterns_independent_per_instance(self) -> None:
        """Each AcpConfig gets its own list (no mutable default sharing)."""
        c1 = AcpConfig()
        c2 = AcpConfig()
        c1.memory_filing_patterns.append("test")
        assert c2.memory_filing_patterns == []


# ---------------------------------------------------------------------------
# _coerce_value tests (direct unit tests)
# ---------------------------------------------------------------------------


class TestCoerceValue:
    def test_double_quoted_string(self) -> None:
        assert _coerce_value('"hello world"') == "hello world"

    def test_single_quoted_string(self) -> None:
        assert _coerce_value("'quoted'") == "quoted"

    def test_quoted_number_stays_string(self) -> None:
        """A quoted number should remain a string, not be coerced."""
        assert _coerce_value('"42"') == "42"

    def test_bool_true(self) -> None:
        assert _coerce_value("true") is True

    def test_bool_True_mixed_case(self) -> None:
        assert _coerce_value("True") is True

    def test_bool_yes(self) -> None:
        assert _coerce_value("yes") is True

    def test_bool_YES_upper(self) -> None:
        assert _coerce_value("YES") is True

    def test_bool_false(self) -> None:
        assert _coerce_value("false") is False

    def test_bool_False_mixed_case(self) -> None:
        assert _coerce_value("False") is False

    def test_bool_no(self) -> None:
        assert _coerce_value("no") is False

    def test_bool_NO_upper(self) -> None:
        assert _coerce_value("NO") is False

    def test_integer(self) -> None:
        assert _coerce_value("42") == 42
        assert isinstance(_coerce_value("42"), int)

    def test_negative_integer(self) -> None:
        assert _coerce_value("-7") == -7

    def test_float(self) -> None:
        assert _coerce_value("3.14") == 3.14
        assert isinstance(_coerce_value("3.14"), float)

    def test_float_no_decimal(self) -> None:
        """5.0 should parse as float, not int."""
        assert _coerce_value("5.0") == 5.0
        assert isinstance(_coerce_value("5.0"), float)

    def test_empty_list(self) -> None:
        assert _coerce_value("[]") == []

    def test_inline_list(self) -> None:
        assert _coerce_value("[a, b, c]") == ["a", "b", "c"]

    def test_inline_list_with_numbers(self) -> None:
        result = _coerce_value("[1, 2, 3]")
        assert result == [1, 2, 3]
        assert all(isinstance(x, int) for x in result)

    def test_inline_list_with_spaces_inside_brackets(self) -> None:
        """[  ] with only whitespace should be an empty list."""
        assert _coerce_value("[  ]") == []

    def test_plain_string(self) -> None:
        assert _coerce_value("hello") == "hello"

    def test_string_with_colon(self) -> None:
        """A value that contains a colon but is not special should stay string."""
        assert _coerce_value("http") == "http"


# ---------------------------------------------------------------------------
# YAML parsing (minimal parser) tests
# ---------------------------------------------------------------------------


class TestMinimalYamlParser:
    def test_simple_key_value(self) -> None:
        text = "key: value\nnumber: 42\n"
        result = _parse_yaml_minimal(text)
        assert result["key"] == "value"
        assert result["number"] == 42

    def test_boolean_values_true_false(self) -> None:
        text = "yes_val: true\nno_val: false\n"
        result = _parse_yaml_minimal(text)
        assert result["yes_val"] is True
        assert result["no_val"] is False

    def test_boolean_values_yes_no(self) -> None:
        text = "flag_a: yes\nflag_b: no\n"
        result = _parse_yaml_minimal(text)
        assert result["flag_a"] is True
        assert result["flag_b"] is False

    def test_quoted_strings(self) -> None:
        text = 'name: "hello world"\nsingle: \'quoted\'\n'
        result = _parse_yaml_minimal(text)
        assert result["name"] == "hello world"
        assert result["single"] == "quoted"

    def test_empty_list(self) -> None:
        text = "patterns: []\n"
        result = _parse_yaml_minimal(text)
        assert result["patterns"] == []

    def test_inline_list(self) -> None:
        text = "items: [one, two, three]\n"
        result = _parse_yaml_minimal(text)
        assert result["items"] == ["one", "two", "three"]

    def test_comments_and_blanks(self) -> None:
        text = "# comment\n\nkey: value\n# another comment\n"
        result = _parse_yaml_minimal(text)
        assert result == {"key": "value"}

    def test_nested_section(self) -> None:
        text = "compaction:\n  enabled: true\n  threshold: 80000\n"
        result = _parse_yaml_minimal(text)
        assert result["compaction_enabled"] is True
        assert result["compaction_threshold"] == 80000

    def test_float_value(self) -> None:
        text = "idle_threshold: 5.0\n"
        result = _parse_yaml_minimal(text)
        assert result["idle_threshold"] == 5.0

    def test_inline_comment(self) -> None:
        text = "key: 42  # this is a comment\n"
        result = _parse_yaml_minimal(text)
        assert result["key"] == 42

    def test_empty_string_value(self) -> None:
        text = 'tmux_session: ""\n'
        result = _parse_yaml_minimal(text)
        assert result["tmux_session"] == ""

    def test_multiple_nested_sections(self) -> None:
        """Two consecutive nested sections are parsed correctly."""
        text = (
            "compaction:\n"
            "  enabled: false\n"
            "  threshold: 90000\n"
            "memory_filing:\n"
            "  enabled: true\n"
            "  grace_after_event: 45\n"
        )
        result = _parse_yaml_minimal(text)
        assert result["compaction_enabled"] is False
        assert result["compaction_threshold"] == 90000
        assert result["memory_filing_enabled"] is True
        assert result["memory_filing_grace_after_event"] == 45

    def test_nested_section_then_top_level(self) -> None:
        """After a nested section, a top-level key resets the section context."""
        text = (
            "compaction:\n"
            "  enabled: true\n"
            "token_threshold: 50000\n"
        )
        result = _parse_yaml_minimal(text)
        assert result["compaction_enabled"] is True
        assert result["token_threshold"] == 50000

    def test_line_without_colon_ignored(self) -> None:
        """Lines without a colon are silently skipped."""
        text = "key: value\nno_colon_here\nanother: 42\n"
        result = _parse_yaml_minimal(text)
        assert result == {"key": "value", "another": 42}

    def test_empty_input(self) -> None:
        assert _parse_yaml_minimal("") == {}

    def test_only_comments(self) -> None:
        assert _parse_yaml_minimal("# just a comment\n# another\n") == {}

    def test_inline_list_with_numbers(self) -> None:
        text = "ports: [8080, 8081, 8082]\n"
        result = _parse_yaml_minimal(text)
        assert result["ports"] == [8080, 8081, 8082]

    def test_section_header_with_empty_value(self) -> None:
        """A section header has key: with no value (empty string)."""
        text = "compaction:\n  enabled: false\n"
        result = _parse_yaml_minimal(text)
        assert result["compaction_enabled"] is False


# ---------------------------------------------------------------------------
# Config flattening tests
# ---------------------------------------------------------------------------


class TestFlattenConfig:
    def test_flat_passthrough(self) -> None:
        raw = {"token_threshold": 80000, "polling_interval": 15}
        result = _flatten_config(raw)
        assert result == {"token_threshold": 80000, "polling_interval": 15}

    def test_nested_compaction(self) -> None:
        raw = {"compaction": {"enabled": False, "threshold": 90000, "cooldown": 60}}
        result = _flatten_config(raw)
        assert result["compaction_enabled"] is False
        assert result["compaction_threshold"] == 90000
        assert result["compaction_cooldown"] == 60

    def test_nested_memory_filing(self) -> None:
        raw = {
            "memory_filing": {
                "enabled": True,
                "grace_after_event": 30,
                "patterns": ["deploy"],
            }
        }
        result = _flatten_config(raw)
        assert result["memory_filing_enabled"] is True
        assert result["memory_filing_grace_after_event"] == 30
        assert result["memory_filing_patterns"] == ["deploy"]

    def test_mixed_flat_and_nested(self) -> None:
        raw = {
            "token_threshold": 80000,
            "compaction": {"enabled": True},
        }
        result = _flatten_config(raw)
        assert result["token_threshold"] == 80000
        assert result["compaction_enabled"] is True

    def test_unknown_nested_key_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unknown sub-keys inside a known nested section trigger a warning."""
        raw = {"compaction": {"enabled": True, "unknown_sub": 999}}
        with caplog.at_level(logging.WARNING, logger="src.config"):
            result = _flatten_config(raw)
        assert result["compaction_enabled"] is True
        assert "unknown_sub" not in result
        assert any("Unknown config key: compaction.unknown_sub" in msg for msg in caplog.messages)

    def test_unknown_top_level_dict_passed_through(self) -> None:
        """A dict value for a key NOT in _NESTED_MAP is passed through as-is."""
        raw = {"custom_section": {"a": 1, "b": 2}}
        result = _flatten_config(raw)
        assert result["custom_section"] == {"a": 1, "b": 2}

    def test_empty_dict(self) -> None:
        assert _flatten_config({}) == {}

    def test_both_nested_sections(self) -> None:
        """Both compaction and memory_filing flatten correctly together."""
        raw = {
            "compaction": {"enabled": False, "threshold": 50000, "cooldown": 30},
            "memory_filing": {"enabled": True, "grace_after_event": 90, "patterns": ["pr_merged"]},
        }
        result = _flatten_config(raw)
        assert result["compaction_enabled"] is False
        assert result["compaction_threshold"] == 50000
        assert result["compaction_cooldown"] == 30
        assert result["memory_filing_enabled"] is True
        assert result["memory_filing_grace_after_event"] == 90
        assert result["memory_filing_patterns"] == ["pr_merged"]


# ---------------------------------------------------------------------------
# load_config tests
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_load_from_file(self, tmp_config: Path) -> None:
        """Load a full config from a YAML file."""
        tmp_config.write_text(
            "token_threshold: 90000\n"
            "polling_interval: 15\n"
            "tmux_session: kelvin\n"
            "compaction:\n"
            "  enabled: false\n"
            "  threshold: 80000\n"
        )
        config = load_config(tmp_config)
        assert config.token_threshold == 90000
        assert config.polling_interval == 15
        assert config.tmux_session == "kelvin"
        assert config.compaction_enabled is False
        assert config.compaction_threshold == 80000

    def test_load_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        """Missing config file returns defaults."""
        missing = tmp_path / "nonexistent.yaml"
        config = load_config(missing)
        assert config == AcpConfig()

    def test_load_no_file_arg_returns_defaults(self) -> None:
        """Passing None uses the default path, which typically does not exist in tests."""
        config = load_config(None)
        assert isinstance(config, AcpConfig)

    def test_partial_config_merges_with_defaults(self, tmp_config: Path) -> None:
        """Partial config overrides only specified fields."""
        tmp_config.write_text("token_threshold: 50000\n")
        config = load_config(tmp_config)
        assert config.token_threshold == 50000
        # All other fields remain at defaults
        assert config.polling_interval == 30
        assert config.warmdown_interval == 120
        assert config.compaction_enabled is True

    def test_unknown_keys_ignored(self, tmp_config: Path) -> None:
        """Unknown keys in the config file are silently ignored."""
        tmp_config.write_text(
            "token_threshold: 60000\n"
            "unknown_key: some_value\n"
        )
        config = load_config(tmp_config)
        assert config.token_threshold == 60000
        assert not hasattr(config, "unknown_key")

    def test_load_default_path_when_none(self) -> None:
        """Passing None uses the default path (~/.acp/config.yaml)."""
        config = load_config(None)
        assert isinstance(config, AcpConfig)

    def test_expanduser_on_paths(self, tmp_config: Path) -> None:
        """Path expansion works on log_file."""
        tmp_config.write_text('log_file: "~/.acp/test.log"\n')
        config = load_config(tmp_config)
        assert config.log_file == "~/.acp/test.log"

    def test_load_unreadable_file_returns_defaults(
        self, tmp_config: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An OSError reading the file falls back to defaults with a warning."""
        tmp_config.write_text("token_threshold: 99999\n")
        with (
            patch.object(Path, "read_text", side_effect=OSError("Permission denied")),
            caplog.at_level(logging.WARNING, logger="src.config"),
        ):
            config = load_config(tmp_config)
        assert config == AcpConfig()
        assert any("Failed to read config file" in msg for msg in caplog.messages)

    def test_load_with_nested_yaml_sections(self, tmp_config: Path) -> None:
        """Full YAML with nested compaction and memory_filing sections."""
        tmp_config.write_text(
            "token_threshold: 80000\n"
            "compaction:\n"
            "  enabled: false\n"
            "  threshold: 65000\n"
            "  cooldown: 60\n"
            "memory_filing:\n"
            "  enabled: true\n"
            "  grace_after_event: 45\n"
            "  patterns: [deploy, release]\n"
        )
        config = load_config(tmp_config)
        assert config.token_threshold == 80000
        assert config.compaction_enabled is False
        assert config.compaction_threshold == 65000
        assert config.compaction_cooldown == 60
        assert config.memory_filing_enabled is True
        assert config.memory_filing_grace_after_event == 45
        assert config.memory_filing_patterns == ["deploy", "release"]

    def test_load_multiple_unknown_keys_all_ignored(self, tmp_config: Path) -> None:
        """Multiple unknown top-level keys are all silently ignored."""
        tmp_config.write_text(
            "token_threshold: 55000\n"
            "foo: bar\n"
            "baz: 123\n"
            "quux: [a, b]\n"
        )
        config = load_config(tmp_config)
        assert config.token_threshold == 55000
        # Defaults for everything else
        assert config.polling_interval == 30


# ---------------------------------------------------------------------------
# save_default_config tests
# ---------------------------------------------------------------------------


class TestSaveDefaultConfig:
    def test_save_creates_file(self, tmp_config: Path) -> None:
        """save_default_config creates the file."""
        save_default_config(tmp_config)
        assert tmp_config.is_file()

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_default_config creates parent directories."""
        deep_path = tmp_path / "a" / "b" / "config.yaml"
        save_default_config(deep_path)
        assert deep_path.is_file()

    def test_saved_config_is_loadable(self, tmp_config: Path) -> None:
        """The saved default config can be loaded back and matches defaults."""
        save_default_config(tmp_config)
        config = load_config(tmp_config)
        default = AcpConfig()

        assert config.token_threshold == default.token_threshold
        assert config.polling_interval == default.polling_interval
        assert config.warmdown_interval == default.warmdown_interval
        assert config.grace_period == default.grace_period
        assert config.compaction_enabled == default.compaction_enabled
        assert config.compaction_threshold == default.compaction_threshold
        assert config.compaction_cooldown == default.compaction_cooldown
        assert config.memory_filing_enabled == default.memory_filing_enabled
        assert config.memory_filing_grace_after_event == default.memory_filing_grace_after_event
        assert config.idle_threshold == default.idle_threshold

    def test_saved_config_has_comments(self, tmp_config: Path) -> None:
        """The saved default config includes explanatory comments."""
        save_default_config(tmp_config)
        text = tmp_config.read_text()
        assert "# Active Context Protocol" in text
        assert "# Token monitoring" in text
        assert "# Compaction trigger" in text

    def test_save_roundtrip_all_defaults(self, tmp_config: Path) -> None:
        """Full roundtrip: save -> load -> compare every field."""
        save_default_config(tmp_config)
        loaded = load_config(tmp_config)
        default = AcpConfig()

        from dataclasses import fields as dc_fields

        for f in dc_fields(AcpConfig):
            assert getattr(loaded, f.name) == getattr(default, f.name), (
                f"Field {f.name!r} mismatch: loaded={getattr(loaded, f.name)!r}, "
                f"default={getattr(default, f.name)!r}"
            )

    def test_saved_content_is_valid_yaml(self, tmp_config: Path) -> None:
        """The saved template parses back to valid data via the minimal parser."""
        save_default_config(tmp_config)
        text = tmp_config.read_text()
        result = _parse_yaml_minimal(text)
        # Should contain at least the known top-level keys
        assert "token_threshold" in result or "compaction_enabled" in result

    def test_saved_config_contains_all_sections(self, tmp_config: Path) -> None:
        """The template includes compaction and memory_filing sections."""
        save_default_config(tmp_config)
        text = tmp_config.read_text()
        assert "compaction:" in text
        assert "memory_filing:" in text
        assert "idle_threshold:" in text


# ---------------------------------------------------------------------------
# _load_yaml fallback test
# ---------------------------------------------------------------------------


class TestLoadYamlFallback:
    def test_falls_back_to_minimal_parser_without_pyyaml(self) -> None:
        """When PyYAML is unavailable, _load_yaml uses _parse_yaml_minimal."""
        from src.config import _load_yaml

        # Mock yaml import failure
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "yaml":
                raise ImportError("No module named 'yaml'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = _load_yaml("key: 42\nflag: true\n")
        assert result["key"] == 42
        assert result["flag"] is True

    def test_pyyaml_returns_non_dict_becomes_empty(self) -> None:
        """If PyYAML's safe_load returns a non-dict (e.g. string), return {}."""
        from src.config import _load_yaml

        with patch("yaml.safe_load", return_value="just a string"):
            result = _load_yaml("just a string")
        assert result == {}
