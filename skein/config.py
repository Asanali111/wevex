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
from typing import Any, Optional

from . import paths as _skein_paths

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "port": 8765,
    "host": "127.0.0.1",
    # Resolved via skein.paths so Windows lands in %APPDATA%\skein\ while
    # macOS / Linux stays at ~/.config/skein/ unchanged. See skein/paths.py.
    "db_path": str(_skein_paths.default_db_path()),
    # ``fastembed`` is the iter-23 default — local 384-dim BAAI/bge-small-en-v1.5
    # ships in the base install so `pip install skein && skein up` is
    # zero-config with real semantic search out of the box. No API key, no
    # cloud round-trip, ~130 MB one-time model download. Cloud providers
    # (``gemini``, ``openai``) are opt-in extras and take priority when their
    # respective env vars + packages are present (see
    # ``embeddings.best_available_provider_name``). ``bm25`` remains as the
    # FTS5-only fallback; ``hash`` is legacy for tests only.
    "embedding_provider": "fastembed",  # "fastembed" | "gemini" | "openai" | "bm25" | "hash"
    "embedding_dimension": 384,
    "bearer_token": "",             # filled in by init
    "log_level": "info",
    "lease_cleanup_interval": 60,   # seconds between lease TTL cleanup
    "stale_mark_interval": 3600,    # seconds between stale-fragment scans
    "default_scope": "project:default",
    # ADR-002 / iter 26: daemon-side replacements for deleted CLI commands.
    # Auto-sync regenerates each registered project's AGENTS.md when the
    # rendered output's hash changes (replaces `skein sync`). Auto-approve
    # drains the extraction-candidate inbox above the confidence threshold
    # and auto-rejects anything still pending past the max-age (replaces
    # `skein inbox approve / reject / auto-approve`).
    "agents_md_sync_interval": 60,          # seconds between sync sweeps
    "inbox_auto_approve_interval": 300,     # seconds between inbox sweeps
    "inbox_auto_approve_threshold": 0.85,   # confidence floor for auto-promote
    "inbox_auto_reject_days": 14,           # candidates older than this get rejected
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
        self.agents_md_sync_interval: int = int(merged["agents_md_sync_interval"])
        self.inbox_auto_approve_interval: int = int(merged["inbox_auto_approve_interval"])
        self.inbox_auto_approve_threshold: float = float(merged["inbox_auto_approve_threshold"])
        self.inbox_auto_reject_days: int = int(merged["inbox_auto_reject_days"])
        # keep raw dict for serialising back
        self._raw: dict[str, Any] = merged

    def to_dict(self) -> dict[str, Any]:
        return dict(self._raw)

    def save(self, path: Optional[Path] = None) -> None:
        p = path or _default_config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
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
    return _skein_paths.default_config_path()


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
        text = path.read_text(encoding="utf-8")
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


def load_config(path: Optional[Path] = None) -> SkeinConfig:
    """Load config from disk, then overlay SKEIN_* env vars."""
    p = path or _default_config_path()
    # Iter 15: source ``~/.config/skein/.env`` (alongside the JSON config) so
    # API keys are available to whatever process is loading config — this is
    # the only safe place to put GEMINI_API_KEY when the daemon runs under
    # launchd, which doesn't inherit the user's shell environment.
    _load_dotenv_file(p.parent / ".env")
    data: dict[str, Any] = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
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
