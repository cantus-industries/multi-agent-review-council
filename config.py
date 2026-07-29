"""Loading and validation of the council's persona configuration."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from models import ConfigError, Severity

__all__ = ["Rule", "Persona", "load_personas", "DEFAULT_CONFIG_PATH"]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "agents_config.json"

MAX_PERSONAS = 32
MAX_RULES_PER_PERSONA = 128
_VALID_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


@dataclass(frozen=True, slots=True)
class Rule:
    """A deterministic check one persona applies to a document."""

    code: str
    pattern: re.Pattern[str]
    severity: Severity
    message: str
    financial_cost: float = 0.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], persona: str) -> "Rule":
        if not isinstance(payload, Mapping):
            raise ConfigError(
                f"{persona}: each rule must be an object, got "
                f"{type(payload).__name__}"
            )
        missing = {"code", "pattern", "severity", "message"} - set(payload)
        if missing:
            raise ConfigError(f"{persona}: rule missing keys {sorted(missing)}")

        code = payload["code"]
        if not isinstance(code, str) or not code.strip():
            raise ConfigError(f"{persona}: rule code must be a non-empty string")

        raw_pattern = payload["pattern"]
        if not isinstance(raw_pattern, str) or not raw_pattern:
            raise ConfigError(
                f"{persona}/{code}: rule pattern must be a non-empty string"
            )
        try:
            pattern = re.compile(raw_pattern, re.IGNORECASE)
        except re.error as exc:
            raise ConfigError(
                f"{persona}/{code}: invalid regex {raw_pattern!r}: {exc}"
            ) from exc

        message = payload["message"]
        if not isinstance(message, str) or not message.strip():
            raise ConfigError(
                f"{persona}/{code}: rule message must be a non-empty string"
            )

        cost = payload.get("financial_cost", 0.0)
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            raise ConfigError(
                f"{persona}/{code}: financial_cost must be a number, got "
                f"{type(cost).__name__}"
            )
        if cost < 0:
            raise ConfigError(
                f"{persona}/{code}: financial_cost must not be negative"
            )

        return cls(
            code=code,
            pattern=pattern,
            severity=Severity.parse(payload["severity"]),
            message=message,
            financial_cost=float(cost),
        )


@dataclass(frozen=True, slots=True)
class Persona:
    """One council member: who they are and what they check for."""

    name: str
    title: str
    blocking: bool
    system_prompt: str
    rules: tuple[Rule, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Persona":
        if not isinstance(payload, Mapping):
            raise ConfigError(
                f"each persona must be an object, got {type(payload).__name__}"
            )
        missing = {"name", "title", "system_prompt"} - set(payload)
        if missing:
            raise ConfigError(f"persona missing keys {sorted(missing)}")

        name = payload["name"]
        if not isinstance(name, str) or not _VALID_NAME.match(name):
            raise ConfigError(
                f"persona name {name!r} must match {_VALID_NAME.pattern} "
                "(it is used as a dictionary key and in log output)"
            )

        for key in ("title", "system_prompt"):
            value = payload[key]
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{name}: {key} must be a non-empty string")

        blocking = payload.get("blocking", False)
        if not isinstance(blocking, bool):
            raise ConfigError(
                f"{name}: blocking must be a bool, got {type(blocking).__name__}"
            )

        raw_rules = payload.get("rules", [])
        if not isinstance(raw_rules, list):
            raise ConfigError(f"{name}: rules must be a list")
        if len(raw_rules) > MAX_RULES_PER_PERSONA:
            raise ConfigError(
                f"{name}: {len(raw_rules)} rules exceeds the "
                f"{MAX_RULES_PER_PERSONA} cap"
            )

        rules = tuple(Rule.from_dict(item, name) for item in raw_rules)
        codes = [r.code for r in rules]
        duplicates = {c for c in codes if codes.count(c) > 1}
        if duplicates:
            raise ConfigError(
                f"{name}: duplicate rule codes {sorted(duplicates)}"
            )

        return cls(
            name=name,
            title=payload["title"],
            blocking=blocking,
            system_prompt=payload["system_prompt"],
            rules=rules,
        )


def load_personas(
    path: str | os.PathLike[str] | None = None,
) -> tuple[Persona, ...]:
    """Read, parse, and validate the persona roster."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"no persona config at {config_path}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read {config_path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{config_path} is not valid UTF-8: {exc}") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{config_path} contains invalid JSON: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ConfigError(
            f"{config_path} must contain an object, got "
            f"{type(payload).__name__}"
        )

    raw_personas = payload.get("personas")
    if not isinstance(raw_personas, list):
        raise ConfigError(f"{config_path} must define a 'personas' list")
    if not raw_personas:
        raise ConfigError(
            f"{config_path} defines zero personas; a council needs members"
        )
    if len(raw_personas) > MAX_PERSONAS:
        raise ConfigError(
            f"{config_path} defines {len(raw_personas)} personas, exceeding "
            f"the {MAX_PERSONAS} cap"
        )

    personas = tuple(Persona.from_dict(item) for item in raw_personas)
    names = [p.name for p in personas]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ConfigError(f"duplicate persona names {sorted(duplicates)}")

    return personas


def load_governance_rules(path: str | os.PathLike[str] | None) -> str:
    """Read the shared standards text injected into every review."""
    if path is None:
        return ""
    rules_path = Path(path)
    try:
        return rules_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ConfigError(f"no governance rules file at {rules_path}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read {rules_path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{rules_path} is not valid UTF-8: {exc}") from exc
