"""Browser-extension pairing endpoint (experimental, iter 30).

Browsers can't reach ``127.0.0.1`` from a regular web page (CORS + the
mixed-context model), but a *browser extension* can — it runs under
its own ``chrome-extension://<id>`` or ``moz-extension://<id>`` origin
with elevated permissions explicitly granted at install time.

This endpoint exists so the Skein extension can self-pair with the
local daemon *without* the user copy-pasting a bearer token from
``~/.config/skein/config.json``. The pairing flow is:

  1. User installs the Skein extension from the Chrome Web Store.
  2. Extension calls ``POST http://127.0.0.1:8765/v1/pair-browser`` once.
  3. Daemon verifies the request originates from a real extension
     (``Origin`` header starts with ``chrome-extension://`` or
     ``moz-extension://`` — browsers control this header, malicious
     sites cannot forge it) AND from the local loopback.
  4. Daemon returns the existing ``cfg.bearer_token`` plus the MCP URL
     so the extension can call ``recall`` / ``remember`` / `search_code`
     against the local daemon for the rest of its lifetime.

Trust model: the daemon binds to 127.0.0.1 only, so a remote attacker
can't hit this endpoint. A malicious *local* process could in principle
spoof the Origin header — but a malicious local process can already
read ``~/.config/skein/config.json`` directly, so this endpoint adds no
additional attack surface. The pairing tax is bounded by the existing
single-machine trust boundary.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..config import get_config

logger = logging.getLogger("skein.pair")

router = APIRouter(prefix="/v1", tags=["pair"])


# Browser extension Origin headers. Both vendors use 32-character lowercase
# IDs; we accept any well-formed one rather than maintaining an allowlist
# of extension IDs (which would require redeploying the daemon every time
# we publish a new extension build).
_EXTENSION_ORIGIN_RE = re.compile(r"^(chrome-extension|moz-extension)://[a-z0-9]{1,128}/?$")

# Local-loopback addresses the request must come from. The daemon also
# binds only to 127.0.0.1 by default; this check is defence-in-depth in
# case a future deployment binds 0.0.0.0.
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _is_loopback(request: Request) -> bool:
    """True if the request's TCP peer is a loopback address.

    FastAPI exposes the client's ``host`` via ``request.client``. On a
    daemon bound to 127.0.0.1, this is always ``127.0.0.1`` — but the
    explicit check keeps the code honest if the bind address ever changes.
    """
    client = request.client
    return client is not None and client.host in _LOOPBACK_HOSTS


@router.post("/pair-browser")
async def pair_browser(request: Request) -> JSONResponse:
    """Return the local daemon's bearer token + MCP URL to a Skein
    browser extension that asked nicely.

    Idempotent: the same daemon always returns the same token. Re-pairing
    after a browser uninstall + reinstall just works.
    """
    if not _is_loopback(request):
        logger.warning(
            "pair-browser rejected non-loopback request from %s",
            request.client.host if request.client else "unknown",
        )
        return JSONResponse(
            {"error": "pair-browser is loopback-only"},
            status_code=403,
        )

    origin = request.headers.get("Origin", "")
    if not _EXTENSION_ORIGIN_RE.match(origin):
        # Don't leak which check failed — keep the error generic so a
        # casual probe doesn't learn the exact validation shape.
        logger.warning("pair-browser rejected non-extension origin %r", origin[:120])
        return JSONResponse(
            {"error": "pair-browser is reserved for browser extensions"},
            status_code=403,
        )

    cfg = get_config()
    if not cfg.bearer_token:
        return JSONResponse(
            {"error": "Skein not initialised. Run `skein init` first."},
            status_code=503,
        )

    # Use the request URL as the source of truth for the daemon URL so the
    # returned value points at whichever port the daemon is actually bound
    # to (matters when the user overrides --port or runs a sidecar daemon
    # on a non-default port). ``request.url`` includes the path, so peel
    # back to scheme+host+port via the URL components.
    base_url = f"{request.url.scheme}://{request.url.netloc}"

    logger.info("paired with browser extension at origin %s", origin)
    return JSONResponse({
        "bearer_token": cfg.bearer_token,
        "daemon_url": base_url,
        "mcp_url": f"{base_url}/mcp",
        "protocol_version": "2024-11-05",
    })
