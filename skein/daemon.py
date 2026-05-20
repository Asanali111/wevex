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
  3. **nohup** (Windows + Linux-without-systemd + macOS with ``--no-persist``):
     spawns a detached process via :mod:`skein._proc` and stores the PID at
     the per-user state dir's ``daemon.pid``. Survives terminal close but
     **not** reboot — there's no native auto-start. On Windows, restart-on-
     login can be added later via a Scheduled Task pointing at ``skein up``.

The manager is idempotent. Calling ``ensure_running()`` when the daemon is
already up is a no-op.

We never assume root. Every file lives under ``$HOME``.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import _proc, paths as _skein_paths

logger = logging.getLogger("skein.daemon")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# launchd and systemd paths are intrinsically platform-specific (the
# launchd backend is only ever invoked when ``platform.system() == "Darwin"``;
# the systemd backend gated by ``shutil.which("systemctl")``). They stay at
# their native locations regardless of where Skein's data dir is.
LAUNCHD_LABEL = "com.skein.daemon"
LAUNCHD_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"

SYSTEMD_UNIT_NAME = "skein.service"
SYSTEMD_UNIT_PATH = Path.home() / ".config" / "systemd" / "user" / SYSTEMD_UNIT_NAME

# Windows Scheduled Task — iter 28 Windows port. Backslash creates a folder
# grouping under Task Scheduler ▸ Task Scheduler Library ▸ Skein. Task runs
# at user logon with restart-on-failure baked into the XML so the daemon
# stays alive across reboots without admin elevation.
SCHTASKS_TASK_NAME = r"Skein\Daemon"

# Nohup PID + log dir live under SKEIN_HOME — these *do* move on Windows
# (to %APPDATA%\skein\). See skein/paths.py.
NOHUP_PID_FILE = _skein_paths.daemon_pid_file()
DAEMON_LOG_DIR = _skein_paths.daemon_log_dir()


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
    elif backend == "schtasks":
        _install_schtasks(skein_bin)
    else:
        _start_nohup(skein_bin)

    # Poll /health directly every 0.5 s for up to 15 s. The previous loop
    # routed every iteration through current_status(), which on the unhealthy
    # path runs `launchctl list` and a PID lookup (~800 ms each). With the
    # FastAPI lifespan taking 5-10 s on a real boot, most of the supposed 30 s
    # budget evaporated in subprocess overhead — fast boots fit, slow boots
    # raced the loop and lost. Direct _check_health probes are sub-ms when the
    # port is closed and capped at 1.5 s when bound-but-slow, so 15 s of
    # efficient probes comfortably covers the >10 s "minimum runtime" cooldown
    # launchd applies after prior failures.
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if _check_health(base_url):
            _write_cached_backend(backend)
            return current_status(base_url)
        time.sleep(0.5)
    return current_status(base_url)


def stop() -> DaemonStatus:
    """Stop and unregister the daemon (any backend)."""
    method = _detect_active_backend()
    if method == "launchd":
        _uninstall_launchd()
    elif method == "systemd":
        _uninstall_systemd()
    elif method == "schtasks":
        _uninstall_schtasks()
    elif method == "nohup":
        _stop_nohup()
    else:
        # Try every backend to be safe — `silent=True` makes each a no-op
        # when the on-disk artefact for that backend is absent.
        _uninstall_launchd(silent=True)
        _uninstall_systemd(silent=True)
        _uninstall_schtasks(silent=True)
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
_BACKEND_CACHE_FILE = _skein_paths.backend_cache_file()


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
        if label == "schtasks":
            # No cheap on-disk artefact for a Windows Scheduled Task —
            # the registration lives inside the Task Scheduler database.
            # Trust the cached label on the warm path; the slow path's
            # `_detect_active_backend` verifies via `schtasks /Query` if
            # the daemon turns out to be unhealthy.
            return label if platform.system() == "Windows" else None
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
    sys_name = platform.system()
    if sys_name == "Darwin" and LAUNCHD_PLIST.exists():
        return "launchd"
    if SYSTEMD_UNIT_PATH.exists():
        return "systemd"
    if sys_name == "Windows":
        # No on-disk file for a Scheduled Task — the cached label is the
        # only cheap signal. Otherwise we'd need a subprocess. Default to
        # the persistent backend if we can see schtasks at all.
        if shutil.which("schtasks"):
            return "schtasks"
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
    # Iter 28: Windows gets schtasks (Scheduled Task) for reboot persistence —
    # the closest analog to launchd/systemd-user. Requires no admin elevation
    # (runs at user logon with /RL LIMITED). Falls back to nohup if the
    # schtasks.exe binary is somehow missing — should never happen on a
    # standard Windows install but keeps the daemon launchable on stripped
    # Windows containers / Server Core variants.
    if sys_name == "Windows" and shutil.which("schtasks"):
        return "schtasks"
    # Linux-without-systemd lands here (and the schtasks-less Windows
    # fallback above).
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
    # Iter 28: schtasks-registered task on Windows. `schtasks /Query /TN X`
    # exits 0 iff the task exists. Cheap (~100 ms) and gives us a definitive
    # answer where no on-disk file does.
    if platform.system() == "Windows" and shutil.which("schtasks"):
        out = subprocess.run(
            ["schtasks", "/Query", "/TN", SCHTASKS_TASK_NAME],
            capture_output=True, text=True,
        )
        if out.returncode == 0:
            return "schtasks"
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

    Prefer the running interpreter's venv (so launchd / systemd / schtasks
    use the same venv that ran `skein up`). Fall back to PATH lookup.

    Windows venvs lay the binary under ``Scripts\\skein.exe``; POSIX
    venvs use ``bin/skein``. Check both layouts so this works regardless of
    whether `skein up` was invoked from a Windows venv or a POSIX one.
    """
    if hasattr(sys, "prefix") and Path(sys.prefix).is_dir():
        prefix = Path(sys.prefix)
        candidates = [
            prefix / "Scripts" / "skein.exe",  # Windows venv
            prefix / "Scripts" / "skein",      # Windows console-script (rare)
            prefix / "bin" / "skein",          # POSIX venv
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
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
        except (OSError, ValueError):
            return None
        # _proc.pid_alive handles Windows (where os.kill(pid, 0) raises
        # OSError("Invalid argument") instead of doing the check).
        return pid if _proc.pid_alive(pid) else None
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
    if method == "schtasks":
        # `schtasks /Query /FO LIST /V /TN X` returns ~20 fields including a
        # "PID" / "Task PID" line while the task action is running. Field
        # labels are localised (German "Task-PID", French "PID de tâche"…)
        # so we look for any line that ends in a number after a colon and
        # has "PID" anywhere in the label. Returning None is OK — the
        # health probe is the source of truth for "is it running".
        out = subprocess.run(
            ["schtasks", "/Query", "/TN", SCHTASKS_TASK_NAME, "/FO", "LIST", "/V"],
            capture_output=True, text=True,
        )
        for line in out.stdout.splitlines():
            stripped = line.strip()
            if "PID" not in stripped.split(":", 1)[0].upper():
                continue
            _, _, value = stripped.partition(":")
            value = value.strip()
            if value.isdigit():
                pid = int(value)
                return pid if pid > 0 else None
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

    # Bootstrap the user agent (idempotent: bootout first, then bootstrap).
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
    # Iter 28: dropped the `launchctl kickstart -k` call that used to follow
    # bootstrap. RunAtLoad=true in the plist already starts the daemon
    # immediately and KeepAlive=true keeps it alive. Issuing kickstart -k
    # (the -k flag means "kill any running instance first") right after
    # bootstrap creates a race where launchd terminates the freshly-bound
    # daemon, the port lingers in TIME_WAIT, the respawn fails with
    # EADDRINUSE, and the daemon needs ~25 s to stabilise. Letting the
    # bootstrap-started instance keep running collapses cold `skein up`
    # from ~20–25 s to ~2 s.


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
    # systemctl only exists on Linux with a user-systemd session. Skipping
    # cleanly on macOS / Windows / containers without systemd is the right
    # behavior — `silent=True` callers (e.g. `stop()` on macOS where launchd
    # is what we actually want to manage) shouldn't crash here.
    if not shutil.which("systemctl"):
        if SYSTEMD_UNIT_PATH.exists():
            # Leftover unit file from a different OS or copied dotfile — drop
            # it but don't try to talk to systemctl.
            try:
                SYSTEMD_UNIT_PATH.unlink()
            except OSError:
                if not silent:
                    raise
        return
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
# Windows Scheduled Task backend
# ---------------------------------------------------------------------------

# Task definition XML — gives us launchd-equivalent behavior on Windows:
#   * triggers at user logon  →  survives reboots
#   * RestartOnFailure interval=1m count=3  →  KeepAlive analog
#   * LogonType=InteractiveToken            →  no stored password required
#   * RunLevel=LeastPrivilege               →  no UAC elevation
#   * MultipleInstancesPolicy=IgnoreNew     →  schtasks /Run while the task
#                                              is already running is a no-op,
#                                              not a duplicate spawn
#   * StopIfGoingOnBatteries / DisallowStartIfOnBatteries = false → keep
#     running on laptops not plugged in (developer machines)
#   * Hidden=true                           →  doesn't clutter the user's
#                                              foreground UI
#
# The schtasks.exe CLI's command-line form (/Create /SC ONLOGON …) doesn't
# expose RestartOnFailure or MultipleInstancesPolicy, so we go via /XML to
# get full parity with launchd's KeepAlive semantics.
_SCHTASKS_XML_TEMPLATE = r"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Local MCP context bus for coding LLMs (managed by `skein up`).</Description>
    <Author>{author}</Author>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{user}</UserId>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{user}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>true</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <DisallowStartOnRemoteAppSession>false</DisallowStartOnRemoteAppSession>
    <UseUnifiedSchedulingEngine>true</UseUnifiedSchedulingEngine>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{skein_bin}</Command>
      <Arguments>serve</Arguments>
      <WorkingDirectory>{cwd}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def _current_windows_user() -> str:
    """Return ``DOMAIN\\user`` (or ``COMPUTER\\user``) for the active logon.

    Used in the task XML's ``<UserId>`` field so schtasks creates a
    user-scoped Scheduled Task (no admin needed).
    """
    domain = os.environ.get("USERDOMAIN") or os.environ.get("COMPUTERNAME") or ""
    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    if domain and user:
        return f"{domain}\\{user}"
    return user or "USER"


def _install_schtasks(skein_bin: str) -> None:
    DAEMON_LOG_DIR.mkdir(parents=True, exist_ok=True)
    xml_path = _skein_paths.skein_home() / "schtasks.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    # Escape any `<`, `>`, `&` that might live in a user's path (`My & Files\`)
    # or domain name. schtasks parses the XML strictly and rejects unescaped
    # entities with the unhelpful "value is incorrectly formatted" error.
    from xml.sax.saxutils import escape as _xml_escape
    xml = _SCHTASKS_XML_TEMPLATE.format(
        author="skein",
        user=_xml_escape(_current_windows_user()),
        skein_bin=_xml_escape(skein_bin),
        cwd=_xml_escape(str(Path.home())),
    )
    # schtasks /Create /XML requires the file to be UTF-16 little-endian with
    # a BOM. Anything else (UTF-8, ASCII) silently produces "ERROR: The task
    # XML contains a value which is incorrectly formatted or out of range."
    xml_path.write_bytes(xml.encode("utf-16"))

    # /F forces overwrite if the task already exists — keeps `skein up`
    # idempotent across upgrades.
    subprocess.run(
        ["schtasks", "/Create", "/TN", SCHTASKS_TASK_NAME, "/XML",
         str(xml_path), "/F"],
        capture_output=True, text=True,
    )
    # Kick the task off immediately. /Run while the task is already running
    # is a no-op thanks to MultipleInstancesPolicy=IgnoreNew above.
    subprocess.run(
        ["schtasks", "/Run", "/TN", SCHTASKS_TASK_NAME],
        capture_output=True, text=True,
    )


def _uninstall_schtasks(silent: bool = False) -> None:
    if not shutil.which("schtasks"):
        return
    # End any running instance first so the next bootstrap doesn't race the
    # previous one's port binding.
    subprocess.run(
        ["schtasks", "/End", "/TN", SCHTASKS_TASK_NAME],
        capture_output=True, text=True,
    )
    proc = subprocess.run(
        ["schtasks", "/Delete", "/TN", SCHTASKS_TASK_NAME, "/F"],
        capture_output=True, text=True,
    )
    # /Delete returns non-zero when the task doesn't exist — fine in `silent`
    # mode (called from `stop()` to clean any leftover state from any
    # backend). Re-raise only when the caller wants strict behaviour.
    if proc.returncode != 0 and not silent:
        # Distinguish "task already gone" (benign) from a real failure. The
        # 0x80070002 ("file not found") HRESULT is what schtasks emits when
        # /Delete targets a missing task; the wording around it is
        # localised, so we key off the hex code only.
        stderr = proc.stderr or ""
        if "0x80070002" not in stderr and "ERROR" in stderr.upper():
            raise RuntimeError(
                f"schtasks /Delete failed: {stderr.strip()}"
            )
    # Best-effort cleanup of the XML scratch file.
    try:
        (_skein_paths.skein_home() / "schtasks.xml").unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# nohup fallback
# ---------------------------------------------------------------------------

def _start_nohup(skein_bin: str) -> None:
    DAEMON_LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Open log files, hand them to the child via spawn_detached, then let
    # the with-block close the parent's dup'd handles. The cross-platform
    # spawn is in skein/_proc.py — uses start_new_session on POSIX,
    # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS on Windows.
    with open(DAEMON_LOG_DIR / "daemon.out", "ab") as log_out, \
         open(DAEMON_LOG_DIR / "daemon.err", "ab") as log_err:
        pid = _proc.spawn_detached(
            [skein_bin, "serve"],
            stdout=log_out, stderr=log_err,
        )
    NOHUP_PID_FILE.write_text(str(pid))


def _stop_nohup(silent: bool = False) -> None:
    if not NOHUP_PID_FILE.exists():
        return
    pid: Optional[int] = None
    try:
        pid = int(NOHUP_PID_FILE.read_text().strip())
    except (OSError, ValueError):
        if not silent:
            raise
    # _proc.terminate_pid handles SIGTERM/SIGKILL on POSIX,
    # CTRL_BREAK_EVENT/TerminateProcess on Windows. Treats
    # "already-gone" as success.
    if pid is not None:
        _proc.terminate_pid(pid, timeout=2.0)
    try:
        NOHUP_PID_FILE.unlink()
    except OSError:
        pass
