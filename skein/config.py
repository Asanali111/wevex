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
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "port": 8765,
    "host": "127.0.0.1",
    "db_path": str(Path.home() / ".config" / "skein" / "skein.db"),
    # ``bm25`` is the honest default — no fake vector embeddings, search
    # uses FTS5 keyword matching. Real semantic search lights up when the
    # user sets GEMINI_API_KEY (or OPENAI_API_KEY) and switches the provider.
    # ``hash`` remains available for tests but emits a doctor warning.
    "embedding_provider": "bm25",   # "bm25" | "gemini" | "openai" | "hash"
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

    def __init__(self, data: dict[str, Any]) -> None:
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
        self._raw: dict[str, Any] = merged

    def to_dict(self) -> dict[str, Any]:
        return dict(self._raw)

    def save(self, path: Path | None = None) -> None:
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


def _load_dotenv_file(path: Path) -> None:
    """Read a ``KEY=value`` .env file and seed ``os.environ`` with anything
    not already set. Iter 15: lets the user keep ``GEMINI_API_KEY`` /
    ``OPENAI_API_KEY`` in ``~/.config/skein/.env`` so the daemon picks them
    up regardless of which shell (launchd, terminal, hook subprocess) is
    starting it. Survives missing file silently.
    """
    if not path.is_file():
        return
    try:
        text = path.read_text()
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip optional surrounding quotes (KEY="value" / KEY='value')
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("\"", "'"):
            value = value[1:-1]
        # Don't clobber values already in the environment — the shell wins
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: Path | None = None) -> SkeinConfig:
    """Load config from disk, then overlay SKEIN_* env vars."""
    p = path or _default_config_path()
    # Iter 15: source ``~/.config/skein/.env`` (alongside the JSON config) so
    # API keys are available to whatever process is loading config — this is
    # the only safe place to put GEMINI_API_KEY when the daemon runs under
    # launchd, which doesn't inherit the user's shell environment.
    _load_dotenv_file(p.parent / ".env")
    data: dict[str, Any] = {}
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

_config: SkeinConfig | None = None


def get_config() -> SkeinConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config(cfg: SkeinConfig | None = None) -> None:
    """Replace the singleton — useful in tests."""
    global _config
    _config = cfg
