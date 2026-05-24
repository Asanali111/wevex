"""Tests for the Windows Scheduled Task backend (iter 28).

We can't actually run schtasks.exe on a macOS / Linux CI host, so every test
here intercepts the subprocess calls and asserts the right command lines /
XML payload would have been issued on a real Windows machine. Exercises:

  * `_install_schtasks` writes the XML in UTF-16 with the right shape and
    invokes `schtasks /Create /TN ... /XML ... /F` then `/Run`.
  * `_uninstall_schtasks` runs `/End` then `/Delete /F` and tolerates the
    "task already absent" return code.
  * `_detect_active_backend` returns "schtasks" when `/Query` succeeds.
  * `_read_pid_for_backend("schtasks")` parses the PID out of localised
    `/Query /FO LIST /V` output.

If a regression silently breaks one of these, Windows users will hit
permission errors or duplicate tasks; these tests are the antibody.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from skein import daemon as daemon_mod


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(daemon_mod, "DAEMON_LOG_DIR",
                        tmp_path / ".config/skein/logs")
    # Rebind module-level path constants so detection doesn't see the
    # user's real ~/.config/skein/* files.
    monkeypatch.setattr(daemon_mod, "LAUNCHD_PLIST",
                        tmp_path / "Library/LaunchAgents/com.skein.daemon.plist")
    monkeypatch.setattr(daemon_mod, "SYSTEMD_UNIT_PATH",
                        tmp_path / ".config/systemd/user/skein.service")
    monkeypatch.setattr(daemon_mod, "NOHUP_PID_FILE",
                        tmp_path / ".config/skein/daemon.pid")
    # Point skein_paths.skein_home() at the tmp_path so install/uninstall
    # writes its XML scratch file under the test tree.
    from skein import paths as _skein_paths
    monkeypatch.setattr(_skein_paths, "skein_home",
                        lambda: tmp_path / ".config/skein")
    return tmp_path


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestInstallSchtasks:
    def test_writes_utf16_xml_and_issues_create_then_run(
        self, isolated_home, monkeypatch
    ):
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _FakeProc(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)
        # Make USERDOMAIN/USERNAME deterministic regardless of host shell
        monkeypatch.setenv("USERDOMAIN", "TESTBOX")
        monkeypatch.setenv("USERNAME", "alice")

        daemon_mod._install_schtasks(r"C:\Skein\Scripts\skein.exe")

        # The scratch XML lands under skein_home().
        from skein import paths as _skein_paths
        xml_path = _skein_paths.skein_home() / "schtasks.xml"
        assert xml_path.exists(), "schtasks XML scratch file not written"
        raw = xml_path.read_bytes()
        # UTF-16 BOM check — schtasks /Create /XML rejects anything else.
        assert raw[:2] in (b"\xff\xfe", b"\xfe\xff"), "XML must be UTF-16 with BOM"

        # Decode + assert the XML mentions our user and binary.
        text = raw.decode("utf-16")
        assert "<UserId>TESTBOX\\alice</UserId>" in text
        assert r"<Command>C:\Skein\Scripts\skein.exe</Command>" in text
        assert "<Arguments>serve</Arguments>" in text
        # RestartOnFailure is the KeepAlive parity bit — must survive any
        # template refactor.
        assert "<RestartOnFailure>" in text
        # Logon trigger drives auto-start at reboot.
        assert "<LogonTrigger>" in text

        # Two subprocess calls: /Create then /Run, both targeting our task.
        assert len(calls) == 2
        create_cmd = calls[0]
        assert create_cmd[:3] == ["schtasks", "/Create", "/TN"]
        assert create_cmd[3] == daemon_mod.SCHTASKS_TASK_NAME
        assert "/XML" in create_cmd and "/F" in create_cmd
        run_cmd = calls[1]
        assert run_cmd[:4] == [
            "schtasks", "/Run", "/TN", daemon_mod.SCHTASKS_TASK_NAME,
        ]


class TestUninstallSchtasks:
    def test_end_then_delete_when_task_present(self, isolated_home, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _FakeProc(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("shutil.which",
                            lambda x: r"C:\Windows\System32\schtasks.exe"
                            if x == "schtasks" else None)

        daemon_mod._uninstall_schtasks()

        assert calls[0][:4] == [
            "schtasks", "/End", "/TN", daemon_mod.SCHTASKS_TASK_NAME,
        ]
        assert calls[1][:4] == [
            "schtasks", "/Delete", "/TN", daemon_mod.SCHTASKS_TASK_NAME,
        ]
        assert "/F" in calls[1]

    def test_silent_when_task_already_absent(self, isolated_home, monkeypatch):
        """A `/Delete` against a missing task returns 0x80070002 — must not raise."""
        def fake_run(cmd, **kwargs):
            if "/Delete" in cmd:
                return _FakeProc(
                    returncode=1,
                    stderr="ERROR: The system cannot find the file specified. (0x80070002)",
                )
            return _FakeProc(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("shutil.which", lambda x: "schtasks")

        # Should not raise.
        daemon_mod._uninstall_schtasks()

    def test_no_op_when_schtasks_missing(self, isolated_home, monkeypatch):
        """Bail out cleanly if `schtasks.exe` is not on PATH (Server Core)."""
        called = {"hit": False}

        def fake_run(cmd, **kwargs):
            called["hit"] = True
            return _FakeProc(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("shutil.which", lambda x: None)

        daemon_mod._uninstall_schtasks()
        assert called["hit"] is False, (
            "_uninstall_schtasks must not spawn subprocesses when schtasks "
            "is missing — would crash with FileNotFoundError on stripped Windows"
        )


class TestDetectActiveBackendWindows:
    def test_returns_schtasks_when_query_succeeds(self, isolated_home, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Windows")
        monkeypatch.setattr("shutil.which",
                            lambda x: "schtasks" if x == "schtasks" else None)

        def fake_run(cmd, **kwargs):
            # `/Query /TN Skein\Daemon` returns 0 → task exists.
            if cmd[:3] == ["schtasks", "/Query", "/TN"]:
                return _FakeProc(returncode=0, stdout="Task exists.")
            return _FakeProc(returncode=1)

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr(daemon_mod, "_check_health", lambda *a, **kw: False)

        assert daemon_mod._detect_active_backend() == "schtasks"

    def test_falls_through_when_no_task(self, isolated_home, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Windows")
        monkeypatch.setattr("shutil.which",
                            lambda x: "schtasks" if x == "schtasks" else None)
        monkeypatch.setattr("subprocess.run",
                            lambda *a, **kw: _FakeProc(returncode=1))
        monkeypatch.setattr(daemon_mod, "_check_health", lambda *a, **kw: False)

        # No task, no nohup pid file → "off"
        assert daemon_mod._detect_active_backend() == "off"


class TestReadPidForBackendSchtasks:
    def test_parses_pid_from_query_list_v(self, monkeypatch):
        # Realistic excerpt of `schtasks /Query /FO LIST /V` output. Field
        # labels are localised on non-English Windows installs — our parser
        # must key off "PID" appearing in the label, not the exact wording.
        stdout = (
            "Folder: \\Skein\n"
            "HostName: TESTBOX\n"
            "TaskName: \\Skein\\Daemon\n"
            "Status: Running\n"
            "Logon Mode: Interactive only\n"
            "Task To Run: skein.exe serve\n"
            "Task PID: 9176\n"
            "Run As User: TESTBOX\\alice\n"
        )
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: _FakeProc(returncode=0, stdout=stdout),
        )
        assert daemon_mod._read_pid_for_backend("schtasks") == 9176

    def test_returns_none_when_no_pid_line(self, monkeypatch):
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: _FakeProc(returncode=0, stdout="Status: Ready\n"),
        )
        # No "PID" field → consumers handle None (status displays it as
        # "via schtasks" with no pid suffix).
        assert daemon_mod._read_pid_for_backend("schtasks") is None


class TestCachedBackendSchtasks:
    def test_returns_schtasks_label_on_windows(self, isolated_home, monkeypatch):
        # Write the cached label.
        cache = isolated_home / ".config/skein/backend"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("schtasks")
        monkeypatch.setattr(daemon_mod, "_BACKEND_CACHE_FILE", cache)
        monkeypatch.setattr("platform.system", lambda: "Windows")
        assert daemon_mod._cached_backend() == "schtasks"

    def test_rejects_schtasks_label_off_windows(self, isolated_home, monkeypatch):
        # Stale cache from a different OS shouldn't poison detection.
        cache = isolated_home / ".config/skein/backend"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("schtasks")
        monkeypatch.setattr(daemon_mod, "_BACKEND_CACHE_FILE", cache)
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        assert daemon_mod._cached_backend() is None


class TestResolveSkeinBinWindowsLayout:
    def test_picks_up_scripts_skein_exe(self, tmp_path, monkeypatch):
        """A Windows venv lays the binary at ``Scripts\\skein.exe``."""
        prefix = tmp_path / "venv"
        scripts = prefix / "Scripts"
        scripts.mkdir(parents=True)
        bin_path = scripts / "skein.exe"
        bin_path.write_text("placeholder")
        monkeypatch.setattr(sys, "prefix", str(prefix))
        assert daemon_mod._resolve_skein_bin() == str(bin_path)
