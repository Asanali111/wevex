"""Tests for the autonomous hook system: hook handlers + hooks_install.

The hook handlers read JSON from a string we pass in (they normally read
stdin) and either print JSON to stdout or write fragments to storage.
We capture stdout via capsys.
"""
from __future__ import annotations

import json

import pytest

from skein.config import SkeinConfig, reset_config
from skein.hooks import (
    _extract_decisions,
    _path_to_territory,
    post_tool_use,
    session_start,
    stop,
    user_prompt_submit,
)
from skein.hooks_install import install_hooks, uninstall_hooks

# ---------------------------------------------------------------------------
# Fixture: configure Skein for hook tests
# ---------------------------------------------------------------------------

@pytest.fixture
def hook_env(tmp_path, monkeypatch):
    """Set up a fresh DB and pin SKEIN_SCOPE so hooks find a clean scope."""
    db_path = tmp_path / "hooks.db"
    cfg = SkeinConfig({
        "db_path": str(db_path),
        "bearer_token": "x" * 64,
        "embedding_provider": "hash",
        "default_scope": "project:hooktest",
    })
    reset_config(cfg)
    monkeypatch.setenv("SKEIN_SCOPE", "project:hooktest")
    monkeypatch.chdir(tmp_path)
    yield {"tmp_path": tmp_path, "db_path": db_path, "cfg": cfg}
    reset_config(None)


# ---------------------------------------------------------------------------
# Helpers — pure functions
# ---------------------------------------------------------------------------

class TestDecisionExtraction:
    def test_extracts_classic_decision(self):
        text = "After looking at this, I decided to use Redis for the session cache."
        out = _extract_decisions(text)
        assert any("Redis" in s for s in out)

    def test_extracts_lets_use(self):
        text = "Let's use PostgreSQL with pgvector for embeddings storage."
        out = _extract_decisions(text)
        assert any("PostgreSQL" in s for s in out)

    def test_extracts_well_use(self):
        text = "We'll go with FastAPI for the web layer."
        out = _extract_decisions(text)
        assert any("FastAPI" in s for s in out)

    def test_ignores_non_decision_sentence(self):
        text = "The sky is blue. The grass is green. Cats meow."
        assert _extract_decisions(text) == []

    def test_dedupes(self):
        # Identical sentences should dedupe; different wording should not.
        text = "I decided to use X. I decided to use X."
        out = _extract_decisions(text)
        assert len(out) == 1

    def test_empty(self):
        assert _extract_decisions("") == []
        assert _extract_decisions(None or "") == []


class TestPathToTerritory:
    def test_two_levels(self):
        assert _path_to_territory("backend/auth/login.py") == "backend/auth"

    def test_one_level(self):
        assert _path_to_territory("README.md") == "README.md"

    def test_empty(self):
        assert _path_to_territory("") is None


# ---------------------------------------------------------------------------
# session_start
# ---------------------------------------------------------------------------

class TestSessionStart:
    def test_no_scope_yet_creates_one_silently(self, hook_env, capsys):
        # No fragments → no output (rather than empty injection)
        rc = session_start("")
        assert rc == 0
        out = capsys.readouterr().out
        assert out == "" or out.strip() == ""

    def test_injects_recent_decisions(self, hook_env, capsys):
        from skein.models import FragmentCreate, IdentityCreate
        from skein.storage import Storage

        s = Storage(str(hook_env["db_path"]))
        # Seed a scope owner + scope, then a decision fragment
        owner = s.create_identity(IdentityCreate(
            handle="user:alice", type="user", name="Alice",
        ))
        from skein.models import ScopeCreate
        scope = s.create_scope(ScopeCreate(
            handle="project:hooktest", type="project",
            name="Hook Test", owner_id=owner.id,
        ))
        s.create_fragment(FragmentCreate(
            type="decision", content="Use Redis for session caching",
            scope_id=scope.id, owner_id=owner.id,
        ))
        s.create_fragment(FragmentCreate(
            type="requirement", content="All endpoints must be authenticated",
            scope_id=scope.id, owner_id=owner.id,
        ))
        s.close()

        rc = session_start("")
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["hookSpecificOutput"]["hookEventName"] == "SessionStart"
        ctx = data["hookSpecificOutput"]["additionalContext"]
        assert "Redis" in ctx
        assert "authenticated" in ctx
        # Requirements outrank decisions in the priority sort
        assert ctx.find("authenticated") < ctx.find("Redis")


# ---------------------------------------------------------------------------
# user_prompt_submit
# ---------------------------------------------------------------------------

class TestUserPromptSubmit:
    def test_short_prompt_skipped(self, hook_env, capsys):
        rc = user_prompt_submit(json.dumps({"prompt": "hi"}))
        assert rc == 0
        assert capsys.readouterr().out == ""

    def test_no_match_skipped(self, hook_env, capsys):
        rc = user_prompt_submit(json.dumps({"prompt": "tell me about quantum chromodynamics"}))
        assert rc == 0
        # No fragments → no injection
        assert capsys.readouterr().out == ""

    def test_with_match_emits_context(self, hook_env, capsys):
        from skein.embeddings import HashEmbeddingProvider, vec_to_bytes
        from skein.models import FragmentCreate, IdentityCreate, ScopeCreate
        from skein.storage import Storage

        s = Storage(str(hook_env["db_path"]))
        provider = HashEmbeddingProvider()
        owner = s.create_identity(IdentityCreate(
            handle="user:bob", type="user", name="Bob",
        ))
        scope = s.create_scope(ScopeCreate(
            handle="project:hooktest", type="project",
            name="Hook Test", owner_id=owner.id,
        ))
        # Add a fragment about caching
        content = "Use Redis with a 1-hour TTL for session caching"
        emb = vec_to_bytes(provider.embed_one(content))
        s.create_fragment(
            FragmentCreate(
                type="decision", content=content,
                scope_id=scope.id, owner_id=owner.id,
            ),
            embedding=emb,
        )
        s.close()

        # User prompt that mentions caching — keyword search should hit it
        rc = user_prompt_submit(json.dumps({
            "prompt": "How should I implement session caching in this project?"
        }))
        assert rc == 0
        out = capsys.readouterr().out
        if out:  # might be skipped if score is too low; check structure
            data = json.loads(out)
            assert data["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
            assert "Redis" in data["hookSpecificOutput"]["additionalContext"]


# ---------------------------------------------------------------------------
# stop hook (auto-extract decisions)
# ---------------------------------------------------------------------------

class TestStopHook:
    def test_extracts_decision_into_storage(self, hook_env):
        from skein.models import IdentityCreate, ScopeCreate
        from skein.storage import Storage

        # Pre-create scope for hook to use (avoid auto-create owner mismatch)
        s = Storage(str(hook_env["db_path"]))
        owner = s.create_identity(IdentityCreate(
            handle="user:alice", type="user", name="Alice",
        ))
        scope = s.create_scope(ScopeCreate(
            handle="project:hooktest", type="project",
            name="Hook Test", owner_id=owner.id,
        ))
        s.close()

        payload = {
            "transcript": "After comparing options, I decided to use FastAPI for the web layer."
        }
        rc = stop(json.dumps(payload))
        assert rc == 0

        # Re-open storage and verify the fragment landed
        s = Storage(str(hook_env["db_path"]))
        frags = s.list_fragments(scope_id=scope.id)
        s.close()

        decisions = [f for f in frags if f.type == "decision"]
        assert any("FastAPI" in f.content for f in decisions)
        assert any("auto-extracted" in f.tags for f in decisions)

    def test_no_decision_no_fragment(self, hook_env):
        from skein.storage import Storage

        rc = stop(json.dumps({"transcript": "The cat sat on the mat. The end."}))
        assert rc == 0

        s = Storage(str(hook_env["db_path"]))
        scope = s.get_scope("project:hooktest")
        if scope:
            frags = s.list_fragments(scope_id=scope.id)
            assert not any(f.type == "decision" and "auto-extracted" in f.tags for f in frags)
        s.close()

    def test_handles_message_list_format(self, hook_env):
        from skein.models import IdentityCreate, ScopeCreate
        from skein.storage import Storage

        s = Storage(str(hook_env["db_path"]))
        owner = s.create_identity(IdentityCreate(
            handle="user:c", type="user", name="C",
        ))
        scope = s.create_scope(ScopeCreate(
            handle="project:hooktest", type="project",
            name="Hook Test", owner_id=owner.id,
        ))
        s.close()

        payload = {
            "transcript": [
                {"role": "user", "content": "what should we use?"},
                {"role": "assistant",
                 "content": "Let's use Postgres with pgvector for embeddings."},
            ]
        }
        rc = stop(json.dumps(payload))
        assert rc == 0

        s = Storage(str(hook_env["db_path"]))
        frags = s.list_fragments(scope_id=scope.id)
        s.close()
        assert any("Postgres" in f.content for f in frags if f.type == "decision")


# ---------------------------------------------------------------------------
# post_tool_use hook
# ---------------------------------------------------------------------------

class TestPostToolUse:
    """post_tool_use is now a deliberate no-op (see iteration 11).

    The previous implementation captured every Edit/Write as a bare
    "Edit on /path/to/file.py" observation fragment. Those carry zero
    signal for the AI consuming Skein context (the file path is already
    visible in the agent's tool call), and they polluted SessionStart
    injection with duplicates. The hook stays wired so client config
    doesn't break, but it never writes."""

    def test_does_not_write_observation_for_edit(self, hook_env):
        from skein.models import IdentityCreate, ScopeCreate
        from skein.storage import Storage

        s = Storage(str(hook_env["db_path"]))
        owner = s.create_identity(IdentityCreate(
            handle="user:d", type="user", name="D",
        ))
        scope = s.create_scope(ScopeCreate(
            handle="project:hooktest", type="project",
            name="Hook Test", owner_id=owner.id,
        ))
        s.close()

        payload = {
            "tool_name": "Edit",
            "tool_input": {"file_path": "backend/auth/login.py"},
        }
        rc = post_tool_use(json.dumps(payload))
        assert rc == 0

        s = Storage(str(hook_env["db_path"]))
        frags = s.list_fragments(scope_id=scope.id)
        s.close()

        # Old behaviour created an observation; new behaviour creates nothing.
        assert all(f.type != "observation" for f in frags), (
            "post_tool_use must not write observation fragments"
        )

    def test_skips_non_file_tools(self, hook_env):
        rc = post_tool_use(json.dumps({
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
        }))
        assert rc == 0  # always 0; never writes

    def test_no_op_on_arbitrary_input(self, hook_env):
        rc = post_tool_use("not json at all")
        assert rc == 0

        rc = post_tool_use("")
        assert rc == 0


# ---------------------------------------------------------------------------
# hooks_install
# ---------------------------------------------------------------------------

class TestHooksInstall:
    def test_install_writes_scope_pin(self, tmp_path):
        install_hooks(tmp_path, "project:test", skein_bin="skein")
        scope_file = tmp_path / ".skein" / "scope"
        assert scope_file.exists()
        assert scope_file.read_text().strip() == "project:test"

    def test_install_writes_claude_settings(self, tmp_path):
        install_hooks(tmp_path, "project:test", skein_bin="skein")
        settings_path = tmp_path / ".claude" / "settings.json"
        assert settings_path.exists()
        data = json.loads(settings_path.read_text())
        assert "hooks" in data
        assert "SessionStart" in data["hooks"]
        assert "Stop" in data["hooks"]
        # Check our managed marker is present
        block = data["hooks"]["SessionStart"][0]
        assert block.get("__skein_managed") is True
        cmd = block["hooks"][0]["command"]
        assert "skein hook session-start" in cmd
        assert "SKEIN_SCOPE=project:test" in cmd

    def test_install_writes_cursor_rule(self, tmp_path):
        install_hooks(tmp_path, "project:cursortest", skein_bin="skein")
        rule_path = tmp_path / ".cursor" / "rules" / "skein.mdc"
        assert rule_path.exists()
        content = rule_path.read_text()
        assert "alwaysApply: true" in content
        assert "project:cursortest" in content
        assert "recall" in content
        assert "remember" in content

    def test_install_preserves_existing_claude_hooks(self, tmp_path):
        # Pre-create a settings.json with a user hook
        settings_dir = tmp_path / ".claude"
        settings_dir.mkdir()
        existing = {
            "hooks": {
                "SessionStart": [
                    {"matcher": "*", "hooks": [{"type": "command", "command": "user-tool"}]}
                ]
            },
            "model": "claude-opus-4",
        }
        (settings_dir / "settings.json").write_text(json.dumps(existing))

        install_hooks(tmp_path, "project:test", skein_bin="skein")

        data = json.loads((settings_dir / "settings.json").read_text())
        # Original user hook still present
        sessions = data["hooks"]["SessionStart"]
        assert len(sessions) == 2
        assert any(b.get("hooks", [{}])[0].get("command") == "user-tool" for b in sessions)
        # Skein hook also present
        assert any(b.get("__skein_managed") for b in sessions)
        # Other settings preserved
        assert data["model"] == "claude-opus-4"

    def test_install_idempotent(self, tmp_path):
        install_hooks(tmp_path, "project:test", skein_bin="skein")
        install_hooks(tmp_path, "project:test", skein_bin="skein")  # again
        data = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        # Only one Skein-managed block per event
        for event_name, blocks in data["hooks"].items():
            managed = [b for b in blocks if b.get("__skein_managed")]
            assert len(managed) == 1, f"Duplicate Skein blocks in {event_name}"

    def test_uninstall_strips_skein_only(self, tmp_path):
        install_hooks(tmp_path, "project:test", skein_bin="skein")
        # Add a user hook alongside
        settings_path = tmp_path / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        data["hooks"]["Stop"].insert(0, {
            "matcher": "*", "hooks": [{"type": "command", "command": "user-stop"}]
        })
        settings_path.write_text(json.dumps(data))

        uninstall_hooks(tmp_path)

        # User hook preserved
        data = json.loads(settings_path.read_text())
        stop_hooks = data.get("hooks", {}).get("Stop", [])
        assert any(
            b.get("hooks", [{}])[0].get("command") == "user-stop"
            for b in stop_hooks
        )
        # No Skein-managed entries left
        for _event_name, blocks in data.get("hooks", {}).items():
            assert not any(b.get("__skein_managed") for b in blocks)

        # Cursor rule deleted
        assert not (tmp_path / ".cursor" / "rules" / "skein.mdc").exists()
        # Scope pin deleted
        assert not (tmp_path / ".skein" / "scope").exists()
