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
    # ``fastembed`` is the default: local BAAI/bge-small-en-v1.5 (384-dim),
    # no API key, ~130 MB one-time model download. ``openai`` is the opt-in
    # cloud option. ``bm25`` is the honest FTS5-only fallback. ``hash`` is
    # tests-only.
    # NOTE: ``"gemini"`` is accepted as a deprecated alias and silently
    # mapped to ``"fastembed"`` by load_config()/SkeinConfig.__init__. See
    # skein/embeddings.py — the Gemini *embedding* API was removed in
    # iter 27 because its rate limits wedged the daemon's event loop. The
    # **Gemini CLI** LLM client (skein/clients.py::GeminiCLIClient) is
    # unrelated and remains a fully-supported sync target.
    "embedding_provider": "fastembed",
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
    # Iter 28 boot-perf: code + docs scanner moved off the `skein up` hot
    # path into a daemon background sweep so warm boot hits <2 s.
    "passive_scan_interval": 300,           # seconds between scanner+docs sweeps
    # Iter 31 efficiency: how often the daemon checks whether the
    # FastembedProvider has been idle long enough to drop its ONNX
    # runtime (saves ~200 MB resident memory during inactive periods).
    # The idle window itself is the provider's _IDLE_UNLOAD_SECONDS,
    # overridable via SKEIN_FASTEMBED_IDLE_SECONDS — this knob is just
    # the polling cadence.
    "embedding_idle_check_interval": 60,
    # Iter 31 (Q-05 phase 3): how often the daemon nudges fragment.value
    # toward its recall-hits-derived target. 6h is slow enough that a
    # single noisy hour doesn't shift the corpus, fast enough that a week
    # of real usage materially re-ranks. Override via env var if you want
    # faster feedback during testing.
    "value_decay_interval": 21600,  # 6 hours
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
        # Normalise the deprecated "gemini" alias at load time so callers
        # never see it. Keeps the daemon alive on upgrade for users whose
        # on-disk config still names the removed Gemini embedding provider.
        ep = str(merged["embedding_provider"]).lower().strip()
        if ep == "gemini":
            ep = "fastembed"
        self.embedding_provider: str = ep
        self.bearer_token: str = merged["bearer_token"]
        self.log_level: str = merged["log_level"]
        self.lease_cleanup_interval: int = int(merged["lease_cleanup_interval"])
        self.stale_mark_interval: int = int(merged["stale_mark_interval"])
        self.default_scope: str = merged["default_scope"]
        self.agents_md_sync_interval: int = int(merged["agents_md_sync_interval"])
        self.inbox_auto_approve_interval: int = int(merged["inbox_auto_approve_interval"])
        self.inbox_auto_approve_threshold: float = float(merged["inbox_auto_approve_threshold"])
        self.inbox_auto_reject_days: int = int(merged["inbox_auto_reject_days"])
        self.passive_scan_interval: int = int(merged["passive_scan_interval"])
        self.embedding_idle_check_interval: int = int(merged["embedding_idle_check_interval"])
        self.value_decay_interval: int = int(merged["value_decay_interval"])
        # Drop legacy embedding_dimension if it crept in — dimension is
        # now read from the provider class so a stale 768 can't silently
        # zero out 384-dim fastembed vectors.
        merged.pop("embedding_dimension", None)
        # Normalise the persisted form too so save() emits "fastembed".
        merged["embedding_provider"] = ep
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
    """Load config from disk, then overlay SKEIN_* env vars.

    Performs a one-shot on-disk migration: if the config still names the
    removed ``gemini`` embedding provider, rewrite it to ``fastembed`` so
    the daemon doesn't crash on respawn and the user-visible config
    matches reality. The migration logs a warning, never raises.
    """
    p = path or _default_config_path()
    # Iter 15: source ``~/.config/skein/.env`` (alongside the JSON config) so
    # opt-in API keys (now: OPENAI_API_KEY) are available to whatever process
    # is loading config — this is the only safe place to put them when the
    # daemon runs under launchd, which doesn't inherit the user's shell env.
    _load_dotenv_file(p.parent / ".env")
    data: dict[str, Any] = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)

    # ---- on-disk migrations FIRST ----
    # Run before env-var overlay so a session-scoped SKEIN_EMBEDDING_PROVIDER
    # (e.g. tests forcing 'hash') can't mask the legacy value on disk and
    # prevent the rewrite from firing.
    needs_save = False
    legacy_on_disk = str(data.get("embedding_provider", "")).lower().strip()
    if legacy_on_disk == "gemini":
        import logging
        logging.getLogger("skein.config").warning(
            "Migrating embedding_provider 'gemini' -> 'fastembed' on disk "
            "(the Gemini embedding API was removed in iter 27). "
            "Existing 768-dim fragment vectors will be ignored by recall "
            "until you re-ingest — run `skein up` from each project root."
        )
        data["embedding_provider"] = "fastembed"
        needs_save = True
    if "embedding_dimension" in data:
        # Legacy key — dimension is now read from the provider class.
        data.pop("embedding_dimension", None)
        needs_save = True

    if needs_save and p.exists():
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
        except OSError:
            # A read-only or sandboxed config dir shouldn't crash the daemon;
            # the in-memory migration (via SkeinConfig.__init__) is enough
            # to keep the daemon alive on respawn.
            pass

    # Env-var overrides: SKEIN_PORT, SKEIN_HOST, SKEIN_DB_PATH, …
    # Applied *after* on-disk migration so legacy strings get rewritten
    # regardless of what env vars are set.
    for key in DEFAULTS:
        env_key = f"SKEIN_{key.upper()}"
        val = os.environ.get(env_key)
        if val is not None:
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
