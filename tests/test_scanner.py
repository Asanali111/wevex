"""Tests for the passive code scanner (iter 14.1)."""
from __future__ import annotations

import json
from pathlib import Path

from skein.scanner import (
    AUTO_PROMOTE_THRESHOLD,
    DISCARD_THRESHOLD,
    ScannedFact,
    classify,
    scan_project,
)


def test_scan_empty_dir(tmp_path: Path) -> None:
    facts = scan_project(tmp_path)
    assert facts == []


def test_scan_missing_dir() -> None:
    facts = scan_project(Path("/path/does/not/exist"))
    assert facts == []


def test_scan_package_json(tmp_path: Path) -> None:
    pkg = {
        "name": "my-app",
        "engines": {"node": ">=20"},
        "dependencies": {"stripe": "^14.0", "react": "^18.0"},
        "scripts": {"test": "vitest", "build": "vite build"},
    }
    (tmp_path / "package.json").write_text(json.dumps(pkg))
    facts = scan_project(tmp_path)
    contents = [f.content for f in facts]
    assert any("my-app" in c for c in contents)
    assert any("Node.js" in c for c in contents)
    assert any("stripe" in c for c in contents)
    assert any("react" in c for c in contents)
    assert any("npm run test" in c for c in contents)


def test_scan_pyproject(tmp_path: Path) -> None:
    pyproject = """\
[project]
name = "myproj"
requires-python = ">=3.10"
dependencies = ["fastapi>=0.100", "pydantic>=2.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 88
"""
    (tmp_path / "pyproject.toml").write_text(pyproject)
    facts = scan_project(tmp_path)
    contents = [f.content for f in facts]
    assert any("myproj" in c for c in contents)
    assert any(">=3.10" in c for c in contents)
    assert any("fastapi" in c for c in contents)
    assert any("pytest" in c for c in contents)
    assert any("ruff" in c for c in contents)


def test_scan_dockerfile(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.11-slim\nEXPOSE 8000\nCMD [\"python\", \"app.py\"]\n"
    )
    facts = scan_project(tmp_path)
    contents = [f.content for f in facts]
    assert any("python:3.11-slim" in c for c in contents)
    assert any("8000" in c for c in contents)


def test_scan_gitignore_infers_stack(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("__pycache__\nnode_modules\n.env\n")
    facts = scan_project(tmp_path)
    contents = [f.content for f in facts]
    assert any("Python project" in c for c in contents)
    assert any("JavaScript" in c or "TypeScript" in c for c in contents)
    assert any(".env" in c for c in contents)


def test_scan_github_actions(tmp_path: Path) -> None:
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "ci.yml").write_text("name: ci\non: [push]\njobs: {}\n")
    facts = scan_project(tmp_path)
    assert any("GitHub Actions" in f.content for f in facts)


def test_scan_test_layout(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("")
    facts = scan_project(tmp_path)
    layout = [f for f in facts if "Tests live" in f.content]
    assert layout
    assert layout[0].type == "preference"
    assert layout[0].topic_key == "tests-layout"


def test_scanner_stamps_stable_topic_keys(tmp_path: Path) -> None:
    """Every auto-promoted scanner emission needs a topic_key so that the
    next scan with mutated content (e.g. file count, dep version) can
    supersede instead of duplicating."""
    pyproject = """\
[project]
name = "myproj"
requires-python = ">=3.10"
dependencies = ["fastapi>=0.100"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 88
"""
    (tmp_path / "pyproject.toml").write_text(pyproject)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("")
    (tmp_path / "README.md").write_text("# My Project\n\nA tagline.\n")
    facts = scan_project(tmp_path)
    # Every high-confidence (auto-promoted) fact should have a topic_key
    for f in facts:
        if f.confidence >= 0.90:
            assert f.topic_key, f"missing topic_key on: {f.content!r}"
    # Keys should be distinct across different facts in the same scan
    keys = [f.topic_key for f in facts if f.topic_key]
    assert len(keys) == len(set(keys)), f"duplicate topic_keys: {keys}"


def test_scan_readme_title(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# My Awesome Project\n\nA short tagline here.\n")
    facts = scan_project(tmp_path)
    assert any("My Awesome Project" in f.content for f in facts)


def test_classify_thresholds() -> None:
    high = ScannedFact(content="x", confidence=AUTO_PROMOTE_THRESHOLD + 0.01)
    mid = ScannedFact(content="x", confidence=AUTO_PROMOTE_THRESHOLD - 0.1)
    low = ScannedFact(content="x", confidence=DISCARD_THRESHOLD - 0.01)
    assert classify(high) == "auto"
    assert classify(mid) == "queue"
    assert classify(low) == "discard"


def test_invalid_package_json_doesnt_crash(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("not json {{{")
    # Should produce zero facts from package.json but not crash; other
    # scanners run independently.
    facts = scan_project(tmp_path)
    assert all("package.json" not in (f.source_file or "") for f in facts)
