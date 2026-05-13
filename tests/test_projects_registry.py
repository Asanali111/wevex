"""Tests for the active-projects registry (~/.config/skein/projects.json)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from skein import projects as projects_mod
from skein.projects import (
    ProjectEntry,
    get_project,
    list_projects,
    remove_project,
    touch_last_ingest,
    upsert_project,
)


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Redirect REGISTRY_PATH so tests don't touch the user's real file."""
    fake = tmp_path / "config" / "skein" / "projects.json"
    monkeypatch.setattr(projects_mod, "REGISTRY_PATH", fake)
    return fake


class TestRegistry:
    def test_list_empty_when_missing(self, isolated_registry):
        assert list_projects() == []

    def test_upsert_creates_file(self, isolated_registry):
        entry = ProjectEntry(
            scope="project:foo", root="/tmp/foo", source_root="foo",
        )
        upsert_project(entry)
        assert isolated_registry.exists()
        data = json.loads(isolated_registry.read_text())
        assert len(data["projects"]) == 1
        assert data["projects"][0]["scope"] == "project:foo"

    def test_upsert_dedupes_by_root(self, isolated_registry, tmp_path):
        root = str(tmp_path / "myproj")
        Path(root).mkdir()
        upsert_project(ProjectEntry(
            scope="project:foo", root=root, source_root="myproj",
        ))
        upsert_project(ProjectEntry(
            scope="project:foo-renamed", root=root, source_root="myproj",
        ))
        items = list_projects()
        assert len(items) == 1
        assert items[0].scope == "project:foo-renamed"

    def test_upsert_dedupes_by_scope(self, isolated_registry, tmp_path):
        a = str(tmp_path / "a")
        b = str(tmp_path / "b")
        Path(a).mkdir()
        Path(b).mkdir()
        upsert_project(ProjectEntry(scope="project:same", root=a, source_root="a"))
        upsert_project(ProjectEntry(scope="project:same", root=b, source_root="b"))
        items = list_projects()
        assert len(items) == 1
        # Second upsert wins
        assert items[0].source_root == "b"

    def test_upsert_preserves_added_at(self, isolated_registry, tmp_path):
        root = str(tmp_path / "preserve")
        Path(root).mkdir()
        upsert_project(ProjectEntry(
            scope="project:p", root=root, source_root="preserve",
            added_at="2026-01-01T00:00:00Z",
        ))
        second = upsert_project(ProjectEntry(
            scope="project:p", root=root, source_root="preserve",
            added_at="2026-12-31T23:59:59Z",
        ))
        # Original added_at survives
        assert second.added_at == "2026-01-01T00:00:00Z"

    def test_get_by_scope(self, isolated_registry, tmp_path):
        root = str(tmp_path / "x")
        Path(root).mkdir()
        upsert_project(ProjectEntry(scope="project:x", root=root, source_root="x"))
        assert get_project("project:x") is not None

    def test_get_by_root(self, isolated_registry, tmp_path):
        root = str((tmp_path / "y").resolve())
        Path(root).mkdir()
        upsert_project(ProjectEntry(scope="project:y", root=root, source_root="y"))
        assert get_project(root) is not None

    def test_get_returns_none_for_unknown(self, isolated_registry):
        assert get_project("project:does-not-exist") is None

    def test_remove(self, isolated_registry, tmp_path):
        root = str((tmp_path / "z").resolve())
        Path(root).mkdir()
        upsert_project(ProjectEntry(scope="project:z", root=root, source_root="z"))
        assert remove_project("project:z") is True
        assert list_projects() == []

    def test_remove_returns_false_when_absent(self, isolated_registry):
        assert remove_project("project:never-existed") is False

    def test_corrupt_json_resets(self, isolated_registry):
        isolated_registry.parent.mkdir(parents=True, exist_ok=True)
        isolated_registry.write_text("{this is not json")
        assert list_projects() == []

    def test_touch_last_ingest(self, isolated_registry, tmp_path):
        root = str((tmp_path / "t").resolve())
        Path(root).mkdir()
        upsert_project(ProjectEntry(scope="project:t", root=root, source_root="t"))
        touch_last_ingest(root)
        item = get_project("project:t")
        assert item is not None
        assert item.last_ingest is not None
