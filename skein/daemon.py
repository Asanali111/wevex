"""Cross-platform persistent-daemon manager.

The local Skein daemon should be a *background service* that:
  • starts on first activation (`skein up`)
  • survives terminal close
  • restarts automatically on reboot

Three backends, picked in this order:
  1. **launchd** (macOS): writes ~/Library/LaunchAgents/com.skein.daemon.plist
     and ``launchctl bootstrap``s the user agent.
  2. **systemd-user** (Linux): writes ~/.config/systemd/user/skein.service
     and ``systemctl --user enable --now``s it.
  3. **nohup** (anything else, or as a forced fallback): spawns a detached
     process and stores the PID at ``~/.config/skein/daemon.pid``. Survives
     terminal close but **not** reboot — there's no native auto-start.

The manager is idempotent. Calling ``ensure_running()`` when the daemon is
already up is a no-op.

We never assume root. Every file lives under ``$HOME``.
"""
from __future__ import annotations

import logging
import os
import platform
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("skein.daemon")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LAUNCHD_LABEL = "com.skein.daemon"
LAUNCHD_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"

SYSTEMD_UNIT_NAME = "skein.service"
SYSTEMD_UNIT_PATH = Path.home() / ".config" / "systemd" / "user" / SYSTEMD_UNIT_NAME

NOHUP_PID_FILE = Path.home() / ".config" / "skein" / "daemon.pid"
DAEMON_LOG_DIR = Path.home() / ".config" / "skein" / "logs"


# ---------------------------------------------------------------------------
# Status type
# ---------------------------------------------------------------------------

@dataclass
class DaemonStatus:
    running: bool
    method: str              # "launchd" | "systemd" | "nohup" | "external" | "off"
    pid: Optional[int]
    healthy: bool            # /health returns 200
    base_url: str

    def summary(self) -> str:
        if not self.running:
            return f"off (last managed by {self.method})" if self.method != "off" else "off"
        prefix = "running" if self.healthy else "starting"
        pid = f" pid={self.pid}" if self.pid else ""
        return f"{prefix} via {self.method}{pid} at {self.base_url}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_running(*, persist: bool = True, base_url: str = "",
                   skein_bin: Optional[str] = None) -> DaemonStatus:
    """Make sure the daemon is up. Idempotent.

    If a daemon is already responding to /health at base_url, do nothing.
    Otherwise install the right backend (launchd/systemd-user/nohup) and
    start the service. Use ``persist=False`` to force the nohup backend
    (useful for short-lived test runs).
    """
    status = current_status(base_url)
    if status.healthy:
        logger.debug("Daemon already running (%s)", status.method)
        return status

    skein_bin = skein_bin or _resolve_skein_bin()
    DAEMON_LOG_DIR.mkdir(parents=True, exist_ok=True)
    NOHUP_PID_FILE.parent.mkdir(parents=True, exist_ok=True)

    backend = _pick_backend(persist=persist)
    logger.info("Starting daemon via %s backend", backend)

    if backend == "launchd":
        _install_launchd(skein_bin)
    elif backend == "systemd":
        _install_systemd(skein_bin)
    else:
        _start_nohup(skein_bin)

    # Wait up to 30s for /health.  launchd applies a "minimum runtime" cooldown
    # after prior failures, so first-boot can take >10 s in pathological cases.
    deadline = time.time() + 30.0
    while time.time() < deadline:
        st = current_status(base_url)
        if st.healthy:
            _write_cached_backend(backend)
            return st
        time.sleep(0.25)
    return current_status(base_url)


def stop() -> DaemonStatus:
    """Stop and unregister the daemon (any backend)."""
    method = _detect_active_backend()
    if method == "launchd":
        _uninstall_launchd()
    elif method == "systemd":
        _uninstall_systemd()
    elif method == "nohup":
        _stop_nohup()
    else:
        # Try all three to be safe
        _uninstall_launchd(silent=True)
        _uninstall_systemd(silent=True)
        _stop_nohup(silent=True)
    # Invalidate the backend cache so a subsequent ensure_running takes the
    # slow path and probes fresh.
    try:
        _BACKEND_CACHE_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    return current_status()


def restart(*, persist: bool = True, base_url: str = "",
            skein_bin: Optional[str] = None) -> DaemonStatus:
    stop()
    time.sleep(0.3)
    return ensure_running(persist=persist, base_url=base_url, skein_bin=skein_bin)


def current_status(base_url: str = "") -> DaemonStatus:
    """Inspect what's running right now.

    Fast path (healthy daemon, common case): a single ~5 ms HTTP probe +
    a 1-line read from the cached-backend file. Avoids the expensive
    ``launchctl list`` / ``systemctl is-enabled`` probes (~800 ms each
    on macOS with a populated user domain) that the slow path uses.

    Cold path (daemon not healthy or no cached backend): fall back to the
    full on-disk probe.
    """
    from .config import get_config
    if not base_url:
        try:
            base_url = get_config().base_url
        except Exception:
            base_url = "http://127.0.0.1:8765"

    healthy = _check_health(base_url)

    if healthy:
        # Cheap path: skip launchctl/systemctl, trust the cached backend
        # label (written by ensure_running()) or fall back to a quick
        # platform default. Validates against on-disk plist/unit presence
        # without spawning a subprocess.
        method = _cached_backend() or _quick_backend_label()
        pid = _read_pid_for_backend(method)
        return DaemonStatus(
            running=True, method=method, pid=pid,
            healthy=True, base_url=base_url,
        )

    # Slow path: daemon isn't responding, we need to know if SOMETHING is
    # registered to launchd/systemd/nohup so we can either start it or
    # report the right diagnosis.
    method = _detect_active_backend()
    pid = _read_pid_for_backend(method)
    return DaemonStatus(
        running=pid is not None, method=method, pid=pid,
        healthy=False, base_url=base_url,
    )


# Cached backend label — written once on successful ensure_running(),
# read on every status check to avoid the launchctl probe.
_BACKEND_CACHE_FILE = Path.home() / ".config" / "skein" / "backend"


def _cached_backend() -> Optional[str]:
    try:
        if not _BACKEND_CACHE_FILE.exists():
            return None
        label = _BACKEND_CACHE_FILE.read_text().strip()
        # Validate the label by checking the on-disk unit file still exists.
        # Cheap stat — no subprocess.
        if label == "launchd":
            return label if LAUNCHD_PLIST.exists() else None
        if label == "systemd":
            return label if SYSTEMD_UNIT_PATH.exists() else None
        if label == "nohup":
            return label if NOHUP_PID_FILE.exists() else None
        if label == "external":
            return label
        return None
    except Exception:
        return None


def _write_cached_backend(label: str) -> None:
    try:
        _BACKEND_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _BACKEND_CACHE_FILE.write_text(label)
    except Exception:
        pass


def _quick_backend_label() -> str:
    """Cheap fallback when there's no cached label — purely file-stat based,
    no subprocess. Used only the very first time after install."""
    if platform.system() == "Darwin" and LAUNCHD_PLIST.exists():
        return "launchd"
    if SYSTEMD_UNIT_PATH.exists():
        return "systemd"
    if NOHUP_PID_FILE.exists():
        return "nohup"
    return "external"


# ---------------------------------------------------------------------------
# Backend pickers / detectors
# ---------------------------------------------------------------------------

def _pick_backend(*, persist: bool) -> str:
    if not persist:
        return "nohup"
    sys_name = platform.system()
    if sys_name == "Darwin":
        return "launchd"
    if sys_name == "Linux" and shutil.which("systemctl"):
        return "systemd"
    return "nohup"


def _detect_active_backend() -> str:
    """Look at on-disk state to figure out which backend (if any) owns the daemon."""
    if LAUNCHD_PLIST.exists() and platform.system() == "Darwin":
        # Verify launchd actually has it loaded
        out = subprocess.run(
            ["launchctl", "list"], capture_output=True, text=True,
        )
        if LAUNCHD_LABEL in out.stdout:
            return "launchd"
    if SYSTEMD_UNIT_PATH.exists() and shutil.which("systemctl"):
        out = subprocess.run(
            ["systemctl", "--user", "is-enabled", SYSTEMD_UNIT_NAME],
            capture_output=True, text=True,
        )
        if "enabled" in out.stdout:
            return "systemd"
    if NOHUP_PID_FILE.exists():
        return "nohup"
    # Maybe something is bound to the port that we didn't start
    if _check_health():
        return "external"
    return "off"


def is_tcc_protected_path(p: Path) -> bool:
    """True if ``p`` lives in a macOS TCC-restricted location.

    launchd-launched processes do not inherit the user's TCC consent, so
    binaries / libs under ``~/Documents``, ``~/Desktop``, ``~/Downloads``,
    iCloud Drive etc. fail with "Operation not permitted". This affects only
    macOS; on Linux/Windows we always return False.
    """
    if platform.system() != "Darwin":
        return False
    home = Path.home()
    protected = [
        home / "Documents",
        home / "Desktop",
        home / "Downloads",
        home / "Movies",
        home / "Music",
        home / "Pictures",
        home / "Library" / "Mobile Documents" / "com~apple~CloudDocs",
    ]
    try:
        resolved = p.resolve()
    except OSError:
        return False
    for prot in protected:
        try:
            resolved.relative_to(prot)
            return True
        except ValueError:
            continue
    return False


def relocate_venv_to_skein_home(*, package_source: Optional[Path] = None) -> Path:
    """Build a TCC-safe venv at ``~/.skein/venv`` and re-link the global ``skein``.

    Returns the path to the new ``skein`` binary inside the relocated venv.
    Used by ``skein up`` when the current install lives in a protected dir.
    """
    target_home = Path.home() / ".skein"
    target_home.mkdir(exist_ok=True)
    target_venv = target_home / "venv"

    # Locate the package source (where pyproject.toml lives)
    if package_source is None:
        import skein as _skein
        package_source = Path(_skein.__file__).parent.parent
    if not (package_source / "pyproject.toml").is_file():
        raise RuntimeError(
            f"Cannot relocate: source dir {package_source} has no pyproject.toml"
        )

    # The relocated venv must NOT be inside a TCC-protected dir; ~/.skein never is.
    # If the source itself is in Documents we copy it to ~/.skein/source first so
    # `pip install -e` resolves to a TCC-safe path too.
    if is_tcc_protected_path(package_source):
        new_source = target_home / "source"
        if new_source.exists():
            # Refresh in place
            shutil.rmtree(new_source)
        shutil.copytree(
            str(package_source), str(new_source),
            ignore=shutil.ignore_patterns(
                ".venv", ".git", "__pycache__", "*.egg-info",
                ".pytest_cache", "node_modules",
            ),
        )
        package_source = new_source

    # Create venv
    if not target_venv.is_dir():
        subprocess.run(
            [sys.executable, "-m", "venv", str(target_venv)],
            check=True,
        )
    pip = target_venv / "bin" / "pip"
    subprocess.run([str(pip), "install", "--quiet", "--upgrade", "pip"], check=True)
    subprocess.run(
        [str(pip), "install", "--quiet", "-e", str(package_source)],
        check=True,
    )

    new_bin = target_venv / "bin" / "skein"
    if not new_bin.is_file():
        raise RuntimeError(f"Relocation completed but {new_bin} missing")

    # Truncate stale launchd log so the user only sees post-relocation state
    for log in ("daemon.err", "daemon.out"):
        log_path = DAEMON_LOG_DIR / log
        if log_path.exists():
            try:
                log_path.write_text("")
            except OSError:
                pass

    # Update common PATH symlinks so `skein` everywhere points at the new venv
    for link in ("/usr/local/bin/skein", str(Path.home() / ".local" / "bin" / "skein")):
        path = Path(link)
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(new_bin)
        except (OSError, PermissionError):
            continue

    return new_bin


def _resolve_skein_bin() -> str:
    """Return the path to the `skein` executable to put in service files.

    Prefer the running interpreter's bin/skein (so launchd uses the same venv
    that ran `skein up`). Fall back to ``shutil.which("skein")``.
    """
    # If we're inside a venv, the venv's bin/skein is reliable
    if hasattr(sys, "prefix") and Path(sys.prefix).is_dir():
        candidate = Path(sys.prefix) / "bin" / "skein"
        if candidate.is_file():
            return str(candidate)
    # Otherwise PATH lookup
    found = shutil.which("skein")
    if found:
        return found
    raise RuntimeError(
        "Could not find the `skein` executable. "
        "Re-run from the venv where Skein is installed, "
        "or pass --skein-bin to skein up."
    )


def _check_health(base_url: str = "http://127.0.0.1:8765") -> bool:
    """Probe ``/health``. Uses stdlib ``urllib.request`` not httpx — httpx
    cold-import is ~700 ms plus another ~400 ms for first-call client setup
    (h11 + TLS context + connection pool), and that all lands on the
    critical path of every ``skein up``. For a single one-shot GET to a
    127.0.0.1 endpoint, stdlib is plenty and pre-imported."""
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=1.5) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ConnectionError, Exception):
        return False


def _read_pid_for_backend(method: str) -> Optional[int]:
    if method == "nohup" and NOHUP_PID_FILE.exists():
        try:
            pid = int(NOHUP_PID_FILE.read_text().strip())
            os.kill(pid, 0)  # raises if dead
            return pid
        except Exception:
            return None
    if method == "launchd":
        out = subprocess.run(
            ["launchctl", "list"], capture_output=True, text=True,
        )
        for line in out.stdout.splitlines():
            cols = line.split()
            if len(cols) >= 3 and cols[2] == LAUNCHD_LABEL:
                try:
                    return int(cols[0]) if cols[0] != "-" else None
                except ValueError:
                    return None
    if method == "systemd":
        out = subprocess.run(
            ["systemctl", "--user", "show", "-p", "MainPID", SYSTEMD_UNIT_NAME],
            capture_output=True, text=True,
        )
        for line in out.stdout.splitlines():
            if line.startswith("MainPID="):
                try:
                    pid = int(line.split("=", 1)[1])
                    return pid if pid > 0 else None
                except ValueError:
                    return None
    return None


# ---------------------------------------------------------------------------
# launchd backend (macOS)
# ---------------------------------------------------------------------------

_PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{skein_bin}</string>
        <string>serve</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>{stdout_path}</string>
    <key>StandardErrorPath</key><string>{stderr_path}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key><string>{path_env}</string>
    </dict>
    <key>WorkingDirectory</key><string>{home}</string>
    <key>ProcessType</key><string>Background</string>
</dict>
</plist>
"""


def _install_launchd(skein_bin: str) -> None:
    LAUNCHD_PLIST.parent.mkdir(parents=True, exist_ok=True)
    plist = _PLIST_TEMPLATE.format(
        label=LAUNCHD_LABEL,
        skein_bin=skein_bin,
        stdout_path=str(DAEMON_LOG_DIR / "daemon.out"),
        stderr_path=str(DAEMON_LOG_DIR / "daemon.err"),
        path_env=os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        home=str(Path.home()),
    )
    LAUNCHD_PLIST.write_text(plist)

    # Bootstrap the user agent (idempotent: bootout first, then bootstrap)
    uid = os.getuid()
    target = f"gui/{uid}"
    subprocess.run(
        ["launchctl", "bootout", target, str(LAUNCHD_PLIST)],
        capture_output=True,
    )
    proc = subprocess.run(
        ["launchctl", "bootstrap", target, str(LAUNCHD_PLIST)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        # Older macOS: fall back to legacy `load` syntax
        subprocess.run(["launchctl", "unload", str(LAUNCHD_PLIST)],
                       capture_output=True)
        subprocess.run(["launchctl", "load", "-w", str(LAUNCHD_PLIST)],
                       capture_output=True)
    subprocess.run(
        ["launchctl", "kickstart", "-k", f"{target}/{LAUNCHD_LABEL}"],
        capture_output=True,
    )


def _uninstall_launchd(silent: bool = False) -> None:
    if not LAUNCHD_PLIST.exists():
        return
    uid = os.getuid()
    target = f"gui/{uid}"
    subprocess.run(
        ["launchctl", "bootout", target, str(LAUNCHD_PLIST)],
        capture_output=True,
    )
    subprocess.run(
        ["launchctl", "unload", str(LAUNCHD_PLIST)],
        capture_output=True,
    )
    try:
        LAUNCHD_PLIST.unlink()
    except OSError:
        if not silent:
            raise


# ---------------------------------------------------------------------------
# systemd-user backend (Linux)
# ---------------------------------------------------------------------------

_SYSTEMD_UNIT_TEMPLATE = """[Unit]
Description=Skein local MCP context bus
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={skein_bin} serve
Restart=always
RestartSec=3
StandardOutput=append:{stdout_path}
StandardError=append:{stderr_path}
Environment=PATH={path_env}

[Install]
WantedBy=default.target
"""


def _install_systemd(skein_bin: str) -> None:
    SYSTEMD_UNIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    unit = _SYSTEMD_UNIT_TEMPLATE.format(
        skein_bin=skein_bin,
        stdout_path=str(DAEMON_LOG_DIR / "daemon.out"),
        stderr_path=str(DAEMON_LOG_DIR / "daemon.err"),
        path_env=os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    )
    SYSTEMD_UNIT_PATH.write_text(unit)

    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT_NAME],
        capture_output=True,
    )


def _uninstall_systemd(silent: bool = False) -> None:
    subprocess.run(
        ["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT_NAME],
        capture_output=True,
    )
    if SYSTEMD_UNIT_PATH.exists():
        try:
            SYSTEMD_UNIT_PATH.unlink()
        except OSError:
            if not silent:
                raise
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)


# ---------------------------------------------------------------------------
# nohup fallback
# ---------------------------------------------------------------------------

def _start_nohup(skein_bin: str) -> None:
    DAEMON_LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Open the log files, hand them to the child, then close the parent's
    # duplicates so we don't leak FDs (Popen dup's into the child).
    with open(DAEMON_LOG_DIR / "daemon.out", "ab") as log_out, \
         open(DAEMON_LOG_DIR / "daemon.err", "ab") as log_err:
        proc = subprocess.Popen(
            [skein_bin, "serve"],
            stdout=log_out, stderr=log_err, stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    NOHUP_PID_FILE.write_text(str(proc.pid))


def _stop_nohup(silent: bool = False) -> None:
    if not NOHUP_PID_FILE.exists():
        return
    try:
        pid = int(NOHUP_PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        # Wait briefly then SIGKILL if still alive
        for _ in range(20):
            try:
                os.kill(pid, 0)
                time.sleep(0.1)
            except ProcessLookupError:
                break
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    except (ValueError, ProcessLookupError):
        if not silent:
            raise
    finally:
        try:
            NOHUP_PID_FILE.unlink()
        except OSError:
            pass
