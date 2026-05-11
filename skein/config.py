"""Skein configuration — loaded from ~/.config/skein/config.json at startup.

The config file is created by ``skein init`` and looked up in order:
  1. $SKEIN_CONFIG env var (path to the JSON file)
  2. ~/.config/skein/config.json

Any key can be overridden by an environment variable with the prefix SKEIN_
(uppercase, underscores).  E.g. SKEIN_PORT=9000 overrides config["port"].
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    "port": 8765,
    "host": "127.0.0.1",
    "db_path": str(Path.home() / ".config" / "skein" / "skein.db"),
    "embedding_provider": "hash",   # "hash" | "gemini" | "openai"
    "embedding_dimension": 768,
    "bearer_token": "",             # filled in by init
    "log_level": "info",
    "lease_cleanup_interval": 60,   # seconds between lease TTL cleanup
    "stale_mark_interval": 3600,    # seconds between stale-fragment scans
    "default_scope": "project:default",
}


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------

class SkeinConfig:
    """Holds runtime configuration for the Skein daemon."""

    def __init__(self, data: Dict[str, Any]) -> None:
        merged = {**DEFAULTS, **data}
        self.port: int = int(merged["port"])
        self.host: str = merged["host"]
        self.db_path: str = merged["db_path"]
        self.embedding_provider: str = merged["embedding_provider"]
        self.embedding_dimension: int = int(merged["embedding_dimension"])
        self.bearer_token: str = merged["bearer_token"]
        self.log_level: str = merged["log_level"]
        self.lease_cleanup_interval: int = int(merged["lease_cleanup_interval"])
        self.stale_mark_interval: int = int(merged["stale_mark_interval"])
        self.default_scope: str = merged["default_scope"]
        # keep raw dict for serialising back
        self._raw: Dict[str, Any] = merged

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._raw)

    def save(self, path: Optional[Path] = None) -> None:
        p = path or _default_config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self._raw, f, indent=2)
            f.write("\n")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _default_config_path() -> Path:
    env = os.environ.get("SKEIN_CONFIG")
    if env:
        return Path(env)
    return Path.home() / ".config" / "skein" / "config.json"


def load_config(path: Optional[Path] = None) -> SkeinConfig:
    """Load config from disk, then overlay SKEIN_* env vars."""
    p = path or _default_config_path()
    data: Dict[str, Any] = {}
    if p.exists():
        with open(p) as f:
            data = json.load(f)

    # Env-var overrides: SKEIN_PORT, SKEIN_HOST, SKEIN_DB_PATH, …
    for key in DEFAULTS:
        env_key = f"SKEIN_{key.upper()}"
        val = os.environ.get(env_key)
        if val is not None:
            # Attempt type coercion from the default
            default_val = DEFAULTS[key]
            if isinstance(default_val, int):
                data[key] = int(val)
            elif isinstance(default_val, bool):
                data[key] = val.lower() in ("1", "true", "yes")
            else:
                data[key] = val

    return SkeinConfig(data)


# ---------------------------------------------------------------------------
# Shared singleton (lazy-loaded per process)
# ---------------------------------------------------------------------------

_config: Optional[SkeinConfig] = None


def get_config() -> SkeinConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config(cfg: Optional[SkeinConfig] = None) -> None:
    """Replace the singleton — useful in tests."""
    global _config
    _config = cfg
