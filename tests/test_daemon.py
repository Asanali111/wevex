"""Tests for the cross-platform daemon manager.

We test the nohup backend (the only one that runs identically in CI without
root or system services). launchd / systemd code paths are smoke-tested via
the "_pick_backend" logic and on real hardware in `skein up`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from skein import daemon as daemon_mod
from skein.daemon import current_status, ensure_running

# ---------------------------------------------------------------------------
# Shared isolation: redirect Path.home() so we don't touch the user's setup
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    # Re-bind the module-level path constants because they're computed at import
    monkeypatch.setattr(daemon_mod, "LAUNCHD_PLIST",
                        tmp_path / "Library/LaunchAgents/com.skein.daemon.plist")
    monkeypatch.setattr(daemon_mod, "SYSTEMD_UNIT_PATH",
                        tmp_path / ".config/systemd/user/skein.service")
    monkeypatch.setattr(daemon_mod, "NOHUP_PID_FILE",
                        tmp_path / ".config/skein/daemon.pid")
    monkeypatch.setattr(daemon_mod, "DAEMON_LOG_DIR",
                        tmp_path / ".config/skein/logs")
    return tmp_path


# ---------------------------------------------------------------------------
# Backend picker
# ---------------------------------------------------------------------------

class TestBackendPicker:
    def test_nohup_when_persist_false(self, monkeypatch):
        assert daemon_mod._pick_backend(persist=False) == "nohup"

    def test_launchd_on_darwin(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        assert daemon_mod._pick_backend(persist=True) == "launchd"

    def test_systemd_on_linux_when_systemctl_present(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/systemctl" if x == "systemctl" else None)
        assert daemon_mod._pick_backend(persist=True) == "systemd"

    def test_nohup_on_linux_without_systemctl(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        monkeypatch.setattr("shutil.which", lambda x: None)
        assert daemon_mod._pick_backend(persist=True) == "nohup"


# ---------------------------------------------------------------------------
# current_status when nothing is installed
# ---------------------------------------------------------------------------

class TestCurrentStatus:
    def test_off_when_nothing(self, isolated_home, monkeypatch):
        # Force health check to fail
        monkeypatch.setattr(daemon_mod, "_check_health", lambda *a, **kw: False)
        s = current_status()
        assert s.method == "off"
        assert not s.healthy
        assert not s.running
        assert s.pid is None

    def test_external_when_someone_else_serves(self, isolated_home, monkeypatch):
        monkeypatch.setattr(daemon_mod, "_check_health", lambda *a, **kw: True)
        s = current_status()
        assert s.method == "external"
        assert s.healthy is True


# ---------------------------------------------------------------------------
# nohup backend (the only one we can fully exercise in a unit test)
# ---------------------------------------------------------------------------

class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid


class TestNohupBackend:
    def test_start_writes_pid_file(self, isolated_home, monkeypatch):
        captured = {}
        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            return _FakeProcess(pid=12345)
        monkeypatch.setattr("subprocess.Popen", fake_popen)

        daemon_mod._start_nohup("/usr/local/bin/skein")
        assert daemon_mod.NOHUP_PID_FILE.exists()
        assert daemon_mod.NOHUP_PID_FILE.read_text().strip() == "12345"
        assert captured["cmd"] == ["/usr/local/bin/skein", "serve"]
        assert daemon_mod.DAEMON_LOG_DIR.exists()

    def test_stop_kills_pid(self, isolated_home, monkeypatch):
        # Plant a pid file
        daemon_mod.NOHUP_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        daemon_mod.NOHUP_PID_FILE.write_text("99999")
        killed = []
        def fake_kill(pid, sig):
            killed.append((pid, sig))
            # Pretend it's dead after first kill
            if len(killed) > 1:
                raise ProcessLookupError()
        monkeypatch.setattr("os.kill", fake_kill)
        daemon_mod._stop_nohup()
        # PID file should be gone, kill should have been called
        assert not daemon_mod.NOHUP_PID_FILE.exists()
        assert killed and killed[0][0] == 99999

    def test_stop_silent_when_no_pid_file(self, isolated_home):
        # Should not raise
        daemon_mod._stop_nohup()


# ---------------------------------------------------------------------------
# detect_active_backend
# ---------------------------------------------------------------------------

class TestDetectActiveBackend:
    def test_off_when_nothing(self, isolated_home, monkeypatch):
        monkeypatch.setattr(daemon_mod, "_check_health", lambda *a, **kw: False)
        assert daemon_mod._detect_active_backend() == "off"

    def test_nohup_when_pid_file_present(self, isolated_home, monkeypatch):
        daemon_mod.NOHUP_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        daemon_mod.NOHUP_PID_FILE.write_text("123")
        monkeypatch.setattr(daemon_mod, "_check_health", lambda *a, **kw: False)
        # launchd plist absent on tmpfs; this falls through to nohup
        assert daemon_mod._detect_active_backend() == "nohup"


# ---------------------------------------------------------------------------
# ensure_running short-circuits when daemon already healthy
# ---------------------------------------------------------------------------

class TestTCCDetection:
    def test_documents_path_protected_on_macos(self, tmp_path, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        bad_path = tmp_path / "Documents" / "myproject" / ".venv" / "bin" / "skein"
        bad_path.parent.mkdir(parents=True)
        bad_path.touch()
        assert daemon_mod.is_tcc_protected_path(bad_path) is True

    def test_skein_home_safe(self, tmp_path, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        good_path = tmp_path / ".skein" / "venv" / "bin" / "skein"
        good_path.parent.mkdir(parents=True)
        good_path.touch()
        assert daemon_mod.is_tcc_protected_path(good_path) is False

    def test_usr_local_safe(self, tmp_path, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        # /usr/local is outside home — never TCC-protected
        good_path = Path("/usr/local/bin/skein")
        if good_path.exists():
            assert daemon_mod.is_tcc_protected_path(good_path) is False

    def test_always_false_on_linux(self, tmp_path, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        bad_path = tmp_path / "Documents" / ".venv" / "bin" / "skein"
        bad_path.parent.mkdir(parents=True)
        bad_path.touch()
        assert daemon_mod.is_tcc_protected_path(bad_path) is False

    def test_resolves_symlinks(self, tmp_path, monkeypatch):
        """A symlink in /usr/local/bin pointing into Documents must be detected."""
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        real = tmp_path / "Documents" / "proj" / ".venv" / "bin" / "skein"
        real.parent.mkdir(parents=True)
        real.touch()
        link = tmp_path / "fake-usr-local" / "skein"
        link.parent.mkdir(parents=True)
        link.symlink_to(real)
        assert daemon_mod.is_tcc_protected_path(link) is True


class TestEnsureRunning:
    def test_no_op_when_already_healthy(self, isolated_home, monkeypatch):
        # Health is healthy → ensure_running just returns the status
        monkeypatch.setattr(daemon_mod, "_check_health", lambda *a, **kw: True)
        called = []
        monkeypatch.setattr(daemon_mod, "_install_launchd",
                            lambda *a, **k: called.append("launchd"))
        monkeypatch.setattr(daemon_mod, "_install_systemd",
                            lambda *a, **k: called.append("systemd"))
        monkeypatch.setattr(daemon_mod, "_start_nohup",
                            lambda *a, **k: called.append("nohup"))

        s = ensure_running(persist=False, base_url="http://127.0.0.1:8765")
        assert s.healthy is True
        assert called == []  # short-circuited


class TestRestartWaitsForSlowDaemon:
    """The FastAPI lifespan takes 5-10s on a real boot. The readiness poll in
    ensure_running() must stay patient enough to catch the daemon when it
    flips healthy mid-window, not declare failure after a few quick probes."""

    def test_restart_succeeds_when_health_flips_after_delay(
        self, isolated_home, monkeypatch
    ):
        # Simulate a daemon that takes ~5 s to come up: /health returns False
        # for the first 10 polls, then True. With the 0.5 s poll interval that
        # mirrors the production failure mode the user hit.
        probe_count = {"n": 0}
        flip_at = 10

        def fake_check_health(*args, **kwargs):
            probe_count["n"] += 1
            return probe_count["n"] > flip_at

        monkeypatch.setattr(daemon_mod, "_check_health", fake_check_health)
        monkeypatch.setattr(daemon_mod, "_resolve_skein_bin",
                            lambda: "/usr/local/bin/skein")
        monkeypatch.setattr(daemon_mod, "_install_launchd", lambda *a, **k: None)
        monkeypatch.setattr(daemon_mod, "_install_systemd", lambda *a, **k: None)
        monkeypatch.setattr(daemon_mod, "_start_nohup", lambda *a, **k: None)
        # nohup backend keeps the test cross-platform; persist=False forces it.
        s = daemon_mod.restart(persist=False, base_url="http://127.0.0.1:8765")

        assert s.healthy is True, (
            f"restart should have waited for the slow daemon, "
            f"but returned healthy=False after {probe_count['n']} probes"
        )
        # Proves the loop actually iterated rather than getting lucky on a
        # cached True — a regression that shortens the poll would trip here.
        assert probe_count["n"] > flip_at
