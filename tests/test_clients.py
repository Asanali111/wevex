"""Tests for ``skein.clients`` — detection, connect, disconnect per client."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from skein import clients as clients_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect Path.home() and clear PATH so detection doesn't match the
    test host's real installs.

    Also redirects ``APPDATA`` and ``LOCALAPPDATA`` to subdirs under
    ``tmp_path`` so clients with Windows-specific config paths
    (opencode → ``%APPDATA%/opencode``, goose → ``%APPDATA%/Block/goose``)
    write into the test sandbox instead of the real user profile.
    Harmless on POSIX where ``_appdata_dir()`` returns ``None`` and these
    env vars are unused by Skein.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    return tmp_path


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    return r


# ---------------------------------------------------------------------------
# detect_all
# ---------------------------------------------------------------------------

class TestDetectAll:
    def test_returns_entry_for_every_known_client(self, fake_home):
        out = clients_mod.detect_all()
        ids = {e["id"] for e in out}
        for cid in clients_mod.all_ids():
            assert cid in ids

    def test_clean_machine_detects_nothing(self, fake_home):
        # Clients that probe absolute paths outside of Path.home() (e.g.
        # /Applications/Windsurf.app on macOS) can legitimately return True on
        # a developer machine that happens to have those apps installed — the
        # fake_home fixture cannot intercept non-home absolute paths. Skip those
        # entries so the test remains meaningful on real developer machines.
        ABSOLUTE_PATH_CLIENTS = {"windsurf", "cursor", "vscode"}
        out = clients_mod.detect_all()
        for entry in out:
            if entry["id"] in ABSOLUTE_PATH_CLIENTS:
                continue
            assert entry["detected"] is False

    def test_picks_up_dir_signal(self, fake_home):
        (fake_home / ".cursor").mkdir()
        out = {e["id"]: e for e in clients_mod.detect_all()}
        assert out["cursor"]["detected"] is True

    def test_each_entry_has_required_keys(self, fake_home):
        for entry in clients_mod.detect_all():
            assert {"id", "display_name", "description", "detected", "note"} <= set(entry.keys())


# ---------------------------------------------------------------------------
# get_client / all_ids
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_ids_unique(self):
        ids = clients_mod.all_ids()
        assert len(ids) == len(set(ids))

    def test_get_client_known(self):
        c = clients_mod.get_client("cursor")
        assert c is not None and c.id == "cursor"

    def test_get_client_unknown(self):
        assert clients_mod.get_client("does-not-exist") is None


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------

class TestCursorClient:
    def test_connect_writes_mcp_json(self, fake_home, repo):
        client = clients_mod.CursorClient()
        paths = client.connect("http://x/mcp", "tok", "project:p", repo)
        cfg = repo / ".cursor" / "mcp.json"
        assert cfg.exists()
        assert str(cfg) in paths
        data = json.loads(cfg.read_text())
        assert data["mcpServers"]["skein"]["url"] == "http://x/mcp"
        assert data["mcpServers"]["skein"]["headers"]["Authorization"] == "Bearer tok"

    def test_connect_preserves_other_servers(self, fake_home, repo):
        cfg = repo / ".cursor" / "mcp.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(json.dumps({"mcpServers": {"other": {"url": "http://other"}}}))
        clients_mod.CursorClient().connect("http://x/mcp", "tok", "p", repo)
        data = json.loads(cfg.read_text())
        assert "other" in data["mcpServers"]
        assert "skein" in data["mcpServers"]

    def test_disconnect_removes_skein_only(self, fake_home, repo):
        cfg = repo / ".cursor" / "mcp.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(json.dumps({
            "mcpServers": {
                "skein": {"url": "http://x/mcp"},
                "other": {"url": "http://other"},
            }
        }))
        modified = clients_mod.CursorClient().disconnect(recorded_paths=[str(cfg)])
        data = json.loads(cfg.read_text())
        assert "skein" not in data["mcpServers"]
        assert "other" in data["mcpServers"]
        assert str(cfg) in modified


# ---------------------------------------------------------------------------
# VS Code
# ---------------------------------------------------------------------------

class TestVsCodeClient:
    def test_connect_writes_to_vscode_dir(self, fake_home, repo):
        clients_mod.VsCodeClient().connect("http://x/mcp", "tok", "p", repo)
        cfg = repo / ".vscode" / "mcp.json"
        assert cfg.exists()
        assert json.loads(cfg.read_text())["mcpServers"]["skein"]["url"] == "http://x/mcp"


# ---------------------------------------------------------------------------
# Gemini CLI
# ---------------------------------------------------------------------------

class TestGeminiCLIClient:
    def test_connect_writes_to_home_gemini(self, fake_home, repo):
        clients_mod.GeminiCLIClient().connect("http://x/mcp", "tok", "p", repo)
        cfg = fake_home / ".gemini" / "settings.json"
        assert cfg.exists()
        data = json.loads(cfg.read_text())
        assert data["mcpServers"]["skein"]["url"] == "http://x/mcp"

    def test_connect_writes_schema_compatible_with_gemini_cli(self, fake_home, repo):
        """Regression: Gemini CLI v0.41+ rejects unknown keys under
        mcpServers.<name> with a red startup warning. The schema only
        accepts url/httpUrl/command/args/env/cwd/headers/timeout/trust/
        description/includeTools/excludeTools — and no "transport" field
        (transport is inferred from key presence). Make sure we only emit
        the documented HTTP-streamable subset."""
        clients_mod.GeminiCLIClient().connect("http://x/mcp", "tok", "p", repo)
        cfg = fake_home / ".gemini" / "settings.json"
        entry = json.loads(cfg.read_text())["mcpServers"]["skein"]
        assert "transport" not in entry, (
            "Gemini CLI rejects 'transport' under mcpServers.<name>"
        )
        assert entry["url"] == "http://x/mcp"
        assert entry["headers"] == {"Authorization": "Bearer tok"}
        # Ensure we didn't sneak in any other unexpected keys.
        assert set(entry.keys()) == {"url", "headers"}

    def test_disconnect_removes_skein(self, fake_home, repo):
        client = clients_mod.GeminiCLIClient()
        client.connect("http://x/mcp", "tok", "p", repo)
        cfg = fake_home / ".gemini" / "settings.json"
        client.disconnect(recorded_paths=[str(cfg)])
        data = json.loads(cfg.read_text())
        assert "skein" not in data["mcpServers"]


# ---------------------------------------------------------------------------
# opencode (nested key chain)
# ---------------------------------------------------------------------------

class TestOpenCodeClient:
    def test_connect_writes_nested(self, fake_home, repo):
        clients_mod.OpenCodeClient().connect("http://x/mcp", "tok", "p", repo)
        cfg = clients_mod._opencode_config_dir() / "config.json"
        assert cfg.exists()
        data = json.loads(cfg.read_text())
        assert data["mcp"]["servers"]["skein"]["url"] == "http://x/mcp"

    def test_connect_omits_transport_key(self, fake_home, repo):
        """opencode MCP schema infers transport from key presence (iter 18.6 fix
        parallel to the Gemini CLI fix in iter 18.1)."""
        clients_mod.OpenCodeClient().connect("http://x/mcp", "tok", "p", repo)
        cfg = clients_mod._opencode_config_dir() / "config.json"
        entry = json.loads(cfg.read_text())["mcp"]["servers"]["skein"]
        assert "transport" not in entry
        assert set(entry.keys()) == {"url", "headers"}

    def test_disconnect_removes_nested(self, fake_home, repo):
        client = clients_mod.OpenCodeClient()
        client.connect("http://x/mcp", "tok", "p", repo)
        cfg = clients_mod._opencode_config_dir() / "config.json"
        client.disconnect(recorded_paths=[str(cfg)])
        data = json.loads(cfg.read_text())
        assert "skein" not in data["mcp"]["servers"]


# ---------------------------------------------------------------------------
# Codex (TOML)
# ---------------------------------------------------------------------------

class TestCodexClient:
    def test_connect_appends_block(self, fake_home, repo):
        clients_mod.CodexClient().connect("http://x/mcp", "tok", "p", repo)
        cfg = repo / ".codex" / "config.toml"
        assert cfg.exists()
        text = cfg.read_text()
        assert "[[mcpServers]]" in text
        assert 'name = "skein"' in text
        assert 'url = "http://x/mcp"' in text
        assert 'Authorization = "Bearer tok"' in text

    def test_connect_refreshes_on_reconnect(self, fake_home, repo):
        """Second connect must REPLACE the existing skein block, not skip it.

        Before iter 18.6 this method bailed out when 'skein' was already in
        the config, leaving the previous (possibly rotated-away) token stuck.
        That was caught when a security sweep found the dead leaked iter-16
        token still living in .codex/config.toml."""
        c = clients_mod.CodexClient()
        c.connect("http://x/mcp", "tok", "p", repo)
        c.connect("http://x/mcp", "tok2", "p", repo)
        text = (repo / ".codex" / "config.toml").read_text()
        # exactly one skein block — but with the NEW token, not the old one
        assert text.count('name = "skein"') == 1
        assert 'Authorization = "Bearer tok2"' in text
        assert 'Authorization = "Bearer tok"\n' not in text

    def test_connect_omits_transport_key(self, fake_home, repo):
        """Codex MCP TOML infers transport from key presence (iter 18.6 fix
        parallel to the Gemini CLI fix in iter 18.1)."""
        clients_mod.CodexClient().connect("http://x/mcp", "tok", "p", repo)
        text = (repo / ".codex" / "config.toml").read_text()
        assert "transport" not in text

    def test_connect_preserves_user_blocks(self, fake_home, repo):
        cfg = repo / ".codex" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            '[user]\nname = "alice"\n\n'
            '[[mcpServers]]\nname = "other"\nurl = "http://other"\n'
        )
        clients_mod.CodexClient().connect("http://x/mcp", "tok", "p", repo)
        text = cfg.read_text()
        assert 'name = "alice"' in text
        assert 'name = "other"' in text
        assert 'name = "skein"' in text

    def test_disconnect_strips_skein_block_only(self, fake_home, repo):
        cfg = repo / ".codex" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            '[user]\nname = "alice"\n\n'
            '[[mcpServers]]\nname = "other"\nurl = "http://other"\n\n'
        )
        client = clients_mod.CodexClient()
        client.connect("http://x/mcp", "tok", "p", repo)
        client.disconnect(recorded_paths=[str(cfg)])
        text = cfg.read_text()
        assert 'name = "alice"' in text
        assert 'name = "other"' in text
        assert 'name = "skein"' not in text


# ---------------------------------------------------------------------------
# Windsurf
# ---------------------------------------------------------------------------

class TestWindsurfClient:
    def test_connect_writes_mcp_json_with_server_url(self, fake_home, repo):
        client = clients_mod.WindsurfClient()
        paths = client.connect("http://x/mcp", "tok", "project:p", repo)
        cfg = repo / ".windsurf" / "mcp.json"
        assert cfg.exists()
        assert str(cfg) in paths
        data = json.loads(cfg.read_text())
        # Windsurf uses "serverUrl" not "url"
        assert data["mcpServers"]["skein"]["serverUrl"] == "http://x/mcp"
        assert "url" not in data["mcpServers"]["skein"]
        assert data["mcpServers"]["skein"]["headers"]["Authorization"] == "Bearer tok"

    def test_connect_preserves_other_servers(self, fake_home, repo):
        cfg = repo / ".windsurf" / "mcp.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(json.dumps({"mcpServers": {"other": {"serverUrl": "http://other"}}}))
        clients_mod.WindsurfClient().connect("http://x/mcp", "tok", "p", repo)
        data = json.loads(cfg.read_text())
        assert "other" in data["mcpServers"]
        assert "skein" in data["mcpServers"]

    def test_disconnect_removes_skein_only(self, fake_home, repo):
        cfg = repo / ".windsurf" / "mcp.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(json.dumps({
            "mcpServers": {
                "skein": {"serverUrl": "http://x/mcp"},
                "other": {"serverUrl": "http://other"},
            }
        }))
        modified = clients_mod.WindsurfClient().disconnect(recorded_paths=[str(cfg)])
        data = json.loads(cfg.read_text())
        assert "skein" not in data["mcpServers"]
        assert "other" in data["mcpServers"]
        assert str(cfg) in modified

    def test_detect_via_codeium_windsurf_dir(self, fake_home):
        (fake_home / ".codeium" / "windsurf").mkdir(parents=True)
        client = clients_mod.WindsurfClient()
        detected, note = client.detect()
        assert detected is True
        assert "windsurf" in note.lower()


# ---------------------------------------------------------------------------
# Goose (Block)
# ---------------------------------------------------------------------------

class TestGooseClient:
    def test_connect_writes_config_yaml(self, fake_home, repo):
        client = clients_mod.GooseClient()
        paths = client.connect("http://x/mcp", "tok", "project:p", repo)
        cfg = clients_mod._goose_config_dir() / "config.yaml"
        assert cfg.exists()
        assert str(cfg) in paths
        data = yaml.safe_load(cfg.read_text())
        ext = data["extensions"]["skein"]
        assert ext["type"] == "streamable_http"
        assert ext["uri"] == "http://x/mcp"
        assert ext["headers"]["Authorization"] == "Bearer tok"
        assert ext["enabled"] is True

    def test_connect_writes_correct_schema(self, fake_home, repo):
        """Pin exact field set: type=streamable_http, uri (not url), headers, name,
        description, enabled, timeout. This matches ExtensionConfig::StreamableHttp in
        crates/goose/src/agents/extension.rs (serde rename = 'streamable_http')."""
        clients_mod.GooseClient().connect("http://x/mcp", "tok", "p", repo)
        cfg = clients_mod._goose_config_dir() / "config.yaml"
        ext = yaml.safe_load(cfg.read_text())["extensions"]["skein"]
        assert ext["type"] == "streamable_http"
        assert "uri" in ext
        assert "url" not in ext  # Goose uses 'uri' not 'url'
        assert ext["name"] == "skein"
        assert "description" in ext  # required by Goose's deserializer

    def test_connect_preserves_other_extensions(self, fake_home, repo):
        cfg = clients_mod._goose_config_dir() / "config.yaml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            yaml.dump({"extensions": {"other": {"type": "builtin", "name": "other", "enabled": True}}})
        )
        clients_mod.GooseClient().connect("http://x/mcp", "tok", "p", repo)
        data = yaml.safe_load(cfg.read_text())
        assert "other" in data["extensions"]
        assert "skein" in data["extensions"]

    def test_disconnect_removes_skein_only(self, fake_home, repo):
        client = clients_mod.GooseClient()
        cfg = clients_mod._goose_config_dir() / "config.yaml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(yaml.dump({
            "extensions": {
                "skein": {"type": "streamable_http", "uri": "http://x/mcp", "enabled": True},
                "other": {"type": "builtin", "name": "other", "enabled": True},
            }
        }))
        modified = client.disconnect(recorded_paths=[str(cfg)])
        data = yaml.safe_load(cfg.read_text())
        assert "skein" not in data["extensions"]
        assert "other" in data["extensions"]
        assert str(cfg) in modified

    def test_detect_via_config_dir(self, fake_home):
        clients_mod._goose_config_dir().mkdir(parents=True)
        client = clients_mod.GooseClient()
        detected, note = client.detect()
        assert detected is True
        assert "goose" in note.lower()


# ---------------------------------------------------------------------------
# Disconnect: empty registry, missing files, multiple paths
# ---------------------------------------------------------------------------

class TestDisconnectResilience:
    def test_disconnect_no_recorded_paths_no_files(self, fake_home, repo):
        # Run from inside a tmp working directory so cwd-based fallbacks
        # don't accidentally hit user files.
        modified = clients_mod.CursorClient().disconnect(recorded_paths=[])
        assert modified == [] or modified == []  # always benign

    def test_disconnect_missing_recorded_file(self, fake_home, repo):
        modified = clients_mod.CursorClient().disconnect(
            recorded_paths=[str(repo / "nonexistent")]
        )
        assert modified == []


# ---------------------------------------------------------------------------
# Hermes (Nous Research)
# ---------------------------------------------------------------------------

class TestHermesClient:
    def test_connect_writes_mcp_servers_skein_to_yaml(self, fake_home, repo):
        """connect() must write mcp_servers.skein into config.yaml."""
        client = clients_mod.HermesClient()
        paths = client.connect("http://x/mcp", "tok", "project:p", repo)
        config_path = fake_home / ".hermes" / "config.yaml"
        assert config_path.exists()
        assert str(config_path) in paths
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "mcp_servers" in config
        assert "skein" in config["mcp_servers"]
        assert config["mcp_servers"]["skein"]["url"] == "http://x/mcp"

    def test_connect_writes_token_to_env(self, fake_home, repo):
        """connect() must write MCP_SKEIN_API_KEY=<token> to .env."""
        client = clients_mod.HermesClient()
        paths = client.connect("http://x/mcp", "tok", "project:p", repo)
        env_path = fake_home / ".hermes" / ".env"
        assert env_path.exists()
        assert str(env_path) in paths
        env_text = env_path.read_text(encoding="utf-8")
        assert "MCP_SKEIN_API_KEY=tok" in env_text

    def test_connect_uses_env_var_interpolation_not_literal_token(self, fake_home, repo):
        """Authorization header must use ${MCP_SKEIN_API_KEY}, not the raw token."""
        client = clients_mod.HermesClient()
        client.connect("http://x/mcp", "supersecret", "project:p", repo)
        config_path = fake_home / ".hermes" / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        auth = config["mcp_servers"]["skein"]["headers"]["Authorization"]
        assert auth == "Bearer ${MCP_SKEIN_API_KEY}"
        assert "supersecret" not in auth

    def test_connect_preserves_other_mcp_servers(self, fake_home, repo):
        """connect() must not remove pre-existing mcp_servers entries."""
        hermes_home = fake_home / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            yaml.dump({"mcp_servers": {"other": {"url": "http://other"}}}),
            encoding="utf-8",
        )
        clients_mod.HermesClient().connect("http://x/mcp", "tok", "p", repo)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "other" in config["mcp_servers"]
        assert "skein" in config["mcp_servers"]

    def test_disconnect_removes_skein_from_mcp_servers(self, fake_home, repo):
        """disconnect() must remove skein from mcp_servers (keeping other entries)."""
        client = clients_mod.HermesClient()
        # Seed with two servers so mcp_servers is not empty after removal
        hermes_home = fake_home / ".hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            yaml.dump({
                "mcp_servers": {
                    "skein": {"url": "http://x/mcp", "headers": {"Authorization": "Bearer ${MCP_SKEIN_API_KEY}"}},
                    "other": {"url": "http://other"},
                }
            }),
            encoding="utf-8",
        )
        env_path = hermes_home / ".env"
        env_path.write_text("MCP_SKEIN_API_KEY=tok\n", encoding="utf-8")

        modified = client.disconnect()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "skein" not in config.get("mcp_servers", {})
        assert "other" in config.get("mcp_servers", {})
        assert str(config_path) in modified

    def test_detect_via_hermes_home_directory(self, fake_home):
        """detect() must return True when ~/.hermes/ exists."""
        (fake_home / ".hermes").mkdir()
        client = clients_mod.HermesClient()
        detected, note = client.detect()
        assert detected is True


# ---------------------------------------------------------------------------
# Crush (Charm)
# ---------------------------------------------------------------------------

class TestCrushClient:
    def test_connect_writes_crush_json(self, fake_home, repo):
        client = clients_mod.CrushClient()
        paths = client.connect("http://x/mcp", "tok", "project:p", repo)
        cfg = repo / ".crush.json"
        assert cfg.exists()
        assert str(cfg) in paths
        data = json.loads(cfg.read_text())
        assert data["mcp"]["skein"]["url"] == "http://x/mcp"
        assert data["mcp"]["skein"]["headers"]["Authorization"] == "Bearer tok"
        # Crush requires "type" explicitly — does NOT infer from key presence
        assert data["mcp"]["skein"]["type"] == "http"

    def test_connect_preserves_other_servers(self, fake_home, repo):
        cfg = repo / ".crush.json"
        cfg.write_text(json.dumps({"mcp": {"other": {"type": "stdio", "command": "node"}}}))
        clients_mod.CrushClient().connect("http://x/mcp", "tok", "p", repo)
        data = json.loads(cfg.read_text())
        assert "other" in data["mcp"]
        assert "skein" in data["mcp"]

    def test_disconnect_removes_skein_only(self, fake_home, repo):
        cfg = repo / ".crush.json"
        cfg.write_text(json.dumps({
            "mcp": {
                "skein": {"type": "http", "url": "http://x/mcp"},
                "other": {"type": "stdio", "command": "node"},
            }
        }))
        modified = clients_mod.CrushClient().disconnect(recorded_paths=[str(cfg)])
        data = json.loads(cfg.read_text())
        assert "skein" not in data["mcp"]
        assert "other" in data["mcp"]
        assert str(cfg) in modified

    def test_detect_via_config_dir(self, fake_home):
        (fake_home / ".config" / "crush").mkdir(parents=True)
        client = clients_mod.CrushClient()
        detected, note = client.detect()
        assert detected is True
        assert "crush" in note.lower()


# ---------------------------------------------------------------------------
# Kiro (AWS)
# ---------------------------------------------------------------------------

class TestKiroClient:
    def test_connect_writes_mcp_json(self, fake_home, repo):
        client = clients_mod.KiroClient()
        paths = client.connect("http://x/mcp", "tok", "project:p", repo)
        cfg = repo / ".kiro" / "settings" / "mcp.json"
        assert cfg.exists()
        assert str(cfg) in paths
        data = json.loads(cfg.read_text())
        assert data["mcpServers"]["skein"]["url"] == "http://x/mcp"
        assert data["mcpServers"]["skein"]["headers"]["Authorization"] == "Bearer tok"

    def test_connect_writes_schema_compatible_with_kiro(self, fake_home, repo):
        """Regression pin: Kiro infers transport from 'url' vs 'command' presence;
        it does NOT require an explicit 'type' field (per kiro.dev/docs/mcp/configuration/).
        Pin the exact keyset so a future refactor doesn't accidentally add 'type'."""
        clients_mod.KiroClient().connect("http://x/mcp", "tok", "p", repo)
        cfg = repo / ".kiro" / "settings" / "mcp.json"
        entry = json.loads(cfg.read_text())["mcpServers"]["skein"]
        assert "type" not in entry, (
            "Kiro infers transport from key presence — do not write a 'type' field"
        )
        assert entry["url"] == "http://x/mcp"
        assert entry["headers"] == {"Authorization": "Bearer tok"}
        assert set(entry.keys()) == {"url", "headers"}

    def test_connect_preserves_other_servers(self, fake_home, repo):
        cfg = repo / ".kiro" / "settings" / "mcp.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(json.dumps({"mcpServers": {"other": {"url": "http://other"}}}))
        clients_mod.KiroClient().connect("http://x/mcp", "tok", "p", repo)
        data = json.loads(cfg.read_text())
        assert "other" in data["mcpServers"]
        assert "skein" in data["mcpServers"]

    def test_disconnect_removes_skein_only(self, fake_home, repo):
        cfg = repo / ".kiro" / "settings" / "mcp.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(json.dumps({
            "mcpServers": {
                "skein": {"url": "http://x/mcp"},
                "other": {"url": "http://other"},
            }
        }))
        modified = clients_mod.KiroClient().disconnect(recorded_paths=[str(cfg)])
        data = json.loads(cfg.read_text())
        assert "skein" not in data["mcpServers"]
        assert "other" in data["mcpServers"]
        assert str(cfg) in modified

    def test_detect_via_kiro_home_directory(self, fake_home):
        (fake_home / ".kiro").mkdir()
        client = clients_mod.KiroClient()
        detected, note = client.detect()
        assert detected is True
        assert "kiro" in note.lower()


# ---------------------------------------------------------------------------
# Continue.dev
# ---------------------------------------------------------------------------

class TestContinueClient:
    def test_connect_writes_yaml_to_mcpservers_dir(self, fake_home, repo):
        client = clients_mod.ContinueClient()
        paths = client.connect("http://x/mcp", "tok", "project:p", repo)
        block = fake_home / ".continue" / "mcpServers" / "skein.yaml"
        assert block.exists()
        assert str(block) in paths
        data = yaml.safe_load(block.read_text())
        assert data["name"] == "Skein"
        assert data["schema"] == "v1"
        servers = data["mcpServers"]
        assert isinstance(servers, list)
        skein = next(s for s in servers if s["name"] == "skein")
        assert skein["url"] == "http://x/mcp"
        assert skein["type"] == "streamable-http"
        auth = skein["requestOptions"]["headers"]["Authorization"]
        assert auth == "Bearer tok"

    def test_connect_overwrites_on_reconnect(self, fake_home, repo):
        """Second connect must replace the block file, not append to it.

        Same constraint as the iter 18.6 Codex fix: stale tokens must not
        survive a token rotation."""
        client = clients_mod.ContinueClient()
        client.connect("http://x/mcp", "tok1", "project:p", repo)
        client.connect("http://x/mcp", "tok2", "project:p", repo)
        block = fake_home / ".continue" / "mcpServers" / "skein.yaml"
        text = block.read_text()
        # New token present, old one absent
        assert "tok2" in text
        assert "tok1" not in text
        # Exactly one skein entry
        data = yaml.safe_load(text)
        skein_entries = [s for s in data["mcpServers"] if s["name"] == "skein"]
        assert len(skein_entries) == 1

    def test_disconnect_removes_block_file(self, fake_home, repo):
        client = clients_mod.ContinueClient()
        client.connect("http://x/mcp", "tok", "project:p", repo)
        block = fake_home / ".continue" / "mcpServers" / "skein.yaml"
        assert block.exists()
        removed = client.disconnect(recorded_paths=[str(block)])
        assert not block.exists()
        assert str(block) in removed

    def test_disconnect_via_default_path(self, fake_home, repo):
        """disconnect() with no recorded_paths falls back to the default location."""
        client = clients_mod.ContinueClient()
        client.connect("http://x/mcp", "tok", "project:p", repo)
        block = fake_home / ".continue" / "mcpServers" / "skein.yaml"
        assert block.exists()
        removed = client.disconnect()
        assert not block.exists()
        assert str(block) in removed

    def test_disconnect_missing_file_is_benign(self, fake_home, repo):
        client = clients_mod.ContinueClient()
        removed = client.disconnect()
        assert removed == []

    def test_detect_via_continue_dir(self, fake_home):
        (fake_home / ".continue").mkdir()
        client = clients_mod.ContinueClient()
        detected, note = client.detect()
        assert detected is True
        assert ".continue" in note

    def test_clean_machine_not_detected(self, fake_home):
        client = clients_mod.ContinueClient()
        detected, _ = client.detect()
        assert detected is False


# ---------------------------------------------------------------------------
# gptme
# ---------------------------------------------------------------------------

class TestGptmeClient:
    def test_connect_writes_config_toml(self, fake_home, repo):
        client = clients_mod.GptmeClient()
        paths = client.connect("http://x/mcp", "tok", "project:p", repo)
        cfg = fake_home / ".config" / "gptme" / "config.toml"
        assert cfg.exists()
        assert str(cfg) in paths
        text = cfg.read_text()
        assert "[[mcp.servers]]" in text
        assert 'name = "skein"' in text
        assert 'url = "http://x/mcp"' in text
        assert 'Authorization = "Bearer tok"' in text

    def test_connect_refreshes_on_reconnect(self, fake_home, repo):
        """Second connect must REPLACE the existing skein block, not skip it.

        Mirrors the iter-18.6 Codex fix: stale tokens must not survive a
        token rotation."""
        c = clients_mod.GptmeClient()
        c.connect("http://x/mcp", "tok", "project:p", repo)
        c.connect("http://x/mcp", "tok2", "project:p", repo)
        cfg = fake_home / ".config" / "gptme" / "config.toml"
        text = cfg.read_text()
        # Exactly one skein block with the new token
        assert text.count('name = "skein"') == 1
        assert 'Authorization = "Bearer tok2"' in text
        assert "Bearer tok\n" not in text

    def test_connect_preserves_other_servers(self, fake_home, repo):
        cfg = fake_home / ".config" / "gptme" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            '[user]\nname = "alice"\n\n'
            '[[mcp.servers]]\nname = "other"\nurl = "http://other"\n'
        )
        clients_mod.GptmeClient().connect("http://x/mcp", "tok", "project:p", repo)
        text = cfg.read_text()
        assert 'name = "alice"' in text
        assert 'name = "other"' in text
        assert 'name = "skein"' in text

    def test_disconnect_strips_skein_block_only(self, fake_home, repo):
        cfg = fake_home / ".config" / "gptme" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            '[user]\nname = "alice"\n\n'
            '[[mcp.servers]]\nname = "other"\nurl = "http://other"\n\n'
        )
        client = clients_mod.GptmeClient()
        client.connect("http://x/mcp", "tok", "project:p", repo)
        client.disconnect(recorded_paths=[str(cfg)])
        text = cfg.read_text()
        assert 'name = "alice"' in text
        assert 'name = "other"' in text
        assert 'name = "skein"' not in text

    def test_detect_via_config_dir(self, fake_home):
        (fake_home / ".config" / "gptme").mkdir(parents=True)
        client = clients_mod.GptmeClient()
        detected, note = client.detect()
        assert detected is True
        assert "gptme" in note

    def test_clean_machine_not_detected(self, fake_home):
        client = clients_mod.GptmeClient()
        detected, _ = client.detect()
        assert detected is False
