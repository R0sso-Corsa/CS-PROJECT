"""Singleton configuration loader — YAML file + environment variable overrides."""

import os
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "config.yaml"
_config_singleton: Dict[str, Any] | None = None


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load and cache the config singleton. Environment variables override YAML values."""
    global _config_singleton
    if _config_singleton is not None:
        return _config_singleton

    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    # Env var override: ENT_ section replaces top-level keys
    for key, val in os.environ.items():
        if key.startswith("ENT_"):
            cfg_key = key[4:].lower()
            cfg[cfg_key] = val

    _config_singleton = cfg
    logger.info("Config loaded from %s", path)
    return cfg


def get(path: str, default: Any = None) -> Any:
    """Dot-path getter, e.g. get('models.lstm.hidden_size', 256)."""
    cfg = load_config()
    keys = path.split(".")
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def reload():
    """Force re-load from disk (useful in tests)."""
    global _config_singleton
    _config_singleton = None
    return load_config()


class Settings:
    """Attribute-style access to config keys."""

    def __getattr__(self, name: str) -> Any:
        val = get(name)
        if val is None:
            raise AttributeError(name)
        return val

    def to_dict(self) -> Dict[str, Any]:
        return load_config()


settings = Settings()
