"""Non-blocking version-check against PyPI with a 24-hour file cache.

Deliberately lightweight: no new deps, no threads, no event loop.  The
check is fire-and-forget — it never raises, never blocks the caller for
more than 2 s, and caches the result so repeated `skein status` calls
don't hammer PyPI.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Tuple

_PYPI_URL = "https://pypi.org/pypi/skn/json"
_CACHE_TTL = 86_400  # 24 h


def _cache_path() -> Path:
    from .config import _default_config_path
    return _default_config_path().parent / "update_check.json"


def _current_version() -> str:
    try:
        from importlib.metadata import version
        return version("skn")
    except Exception:
        pass
    try:
        from skein import __version__
        return __version__
    except Exception:
        return "0.0.0"


def _load_cache() -> Optional[dict]:
    try:
        raw = _cache_path().read_text()
        data = json.loads(raw)
        if time.time() - data.get("checked_at", 0) < _CACHE_TTL:
            return data
    except Exception:
        pass
    return None


def _save_cache(latest: str) -> None:
    try:
        p = _cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"checked_at": time.time(), "latest": latest}))
    except Exception:
        pass


def _fetch_latest() -> Optional[str]:
    try:
        import httpx
        resp = httpx.get(_PYPI_URL, timeout=2.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()["info"]["version"]
    except Exception:
        return None


def _version_tuple(v: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except Exception:
        return (0,)


def check_for_update() -> Optional[Tuple[str, str]]:
    """Return (current, latest) if an update is available, else None.

    Reads the 24 h cache first; hits PyPI only when stale.  Never raises.
    """
    try:
        current = _current_version()

        cached = _load_cache()
        if cached:
            latest = cached.get("latest", current)
        else:
            latest = _fetch_latest()
            if latest:
                _save_cache(latest)
            else:
                return None

        if _version_tuple(latest) > _version_tuple(current):
            return (current, latest)
    except Exception:
        pass
    return None


def update_banner() -> Optional[str]:
    """Return a one-line Rich markup string if an update is available, else None."""
    result = check_for_update()
    if result is None:
        return None
    current, latest = result
    return (
        f"[bold yellow]⬆ Update available:[/bold yellow] "
        f"[dim]{current}[/dim] → [bold]{latest}[/bold]  "
        f"[dim](run: skein update)[/dim]"
    )
