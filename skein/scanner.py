"""Passive code scanner — extracts implicit facts from a project tree.

iter 14.1: when the LLM is silent, the codebase still tells you things. A
``package.json`` declares "we use Stripe API". A ``Dockerfile`` declares "we
ship Python 3.11 on Alpine". A ``tests/`` directory declares "tests live
here". This module reads those signals and produces ``ScannedFact``s that
``skein up`` (and the watcher) can promote into real fragments.

Design rules:
- **Heuristic-only.** No LLM calls, no network. Pure file reads + parsers.
- **Conservative.** Every fact has a confidence score; only the very
  unambiguous ones (deps in package.json, Python version in Dockerfile)
  auto-promote. Ambiguous ones go through ``skein inbox``.
- **Idempotent.** Same project, same scan output. Re-running on every
  ``skein up`` is safe because the dedup index in ``extraction_candidates``
  catches duplicates.
- **Cheap.** All scanners short-circuit when the source file is missing.
  A whole-project scan on company-brain takes ~50ms.

Each scanner contributes facts of types `fact` (high-confidence material
realities) or `preference` (style/convention choices).
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional

logger = logging.getLogger("skein.scanner")


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class ScannedFact:
    """One implicit fact extracted from the codebase."""
    content: str
    type: str = "fact"                  # fact | preference
    confidence: float = 1.0             # 0..1
    source_file: Optional[str] = None   # relative path inside project root
    tags: List[str] = field(default_factory=list)
    territory: Optional[str] = None


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def scan_project(root: Path) -> List[ScannedFact]:
    """Run every scanner against ``root`` and return the union of facts.

    Order doesn't matter — callers deduplicate. Empty list if the directory
    has nothing recognizable.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        return []
    facts: List[ScannedFact] = []
    for scanner in _SCANNERS:
        try:
            facts.extend(scanner(root))
        except Exception:
            logger.debug("scanner %s failed on %s", scanner.__name__, root,
                         exc_info=True)
    return facts


# ---------------------------------------------------------------------------
# Individual scanners
# ---------------------------------------------------------------------------


def _scan_package_json(root: Path) -> List[ScannedFact]:
    p = root / "package.json"
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text())
    except Exception:
        return []
    out: List[ScannedFact] = []
    # Project name → fact
    name = data.get("name")
    if name:
        out.append(ScannedFact(
            content=f"This project's npm package name is `{name}`.",
            confidence=0.98, source_file="package.json", tags=["npm"],
        ))
    # Runtime engines
    engines = data.get("engines", {})
    if engines.get("node"):
        out.append(ScannedFact(
            content=f"Node.js runtime: `{engines['node']}` (per package.json engines).",
            confidence=0.95, source_file="package.json", tags=["node", "runtime"],
        ))
    # Production deps → one fact per notable dep
    deps = data.get("dependencies") or {}
    notable = _filter_notable_deps(deps)
    for dep, ver in notable.items():
        out.append(ScannedFact(
            content=f"Uses npm package `{dep}` ({ver}).",
            confidence=0.92, source_file="package.json",
            tags=["npm", "dep", dep],
        ))
    # Scripts
    scripts = data.get("scripts") or {}
    for canonical in ("test", "build", "dev", "start", "lint"):
        if canonical in scripts:
            out.append(ScannedFact(
                content=f"`npm run {canonical}` runs: `{scripts[canonical]}`",
                type="procedure",
                confidence=0.88, source_file="package.json",
                tags=["npm", "script"],
            ))
    return out


def _scan_pyproject_toml(root: Path) -> List[ScannedFact]:
    p = root / "pyproject.toml"
    if not p.is_file():
        return []
    try:
        try:
            import tomllib  # py 3.11+
        except ImportError:
            import tomli as tomllib  # type: ignore
        data = tomllib.loads(p.read_text())
    except Exception:
        return []
    out: List[ScannedFact] = []
    proj = data.get("project", {}) or {}
    name = proj.get("name") or data.get("tool", {}).get("poetry", {}).get("name")
    if name:
        out.append(ScannedFact(
            content=f"This project's Python package name is `{name}`.",
            confidence=0.98, source_file="pyproject.toml", tags=["python"],
        ))
    py_req = proj.get("requires-python")
    if py_req:
        out.append(ScannedFact(
            content=f"Python version requirement: `{py_req}`.",
            confidence=0.97, source_file="pyproject.toml",
            tags=["python", "runtime"],
        ))
    deps = proj.get("dependencies") or []
    for d in deps[:60]:  # cap for sanity
        # Pull the package name out (before any version/extras)
        m = re.match(r"^([A-Za-z0-9_.\-]+)", d)
        if not m:
            continue
        pkg = m.group(1)
        out.append(ScannedFact(
            content=f"Uses Python package `{pkg}` (declared: `{d}`).",
            confidence=0.93, source_file="pyproject.toml",
            tags=["python", "dep", pkg.lower()],
        ))
    # Test runner inference
    tool = data.get("tool", {}) or {}
    if "pytest" in tool or (root / "pytest.ini").exists():
        out.append(ScannedFact(
            content="Test runner: pytest.",
            confidence=0.95, source_file="pyproject.toml",
            tags=["testing", "pytest"],
        ))
    if "ruff" in tool:
        out.append(ScannedFact(
            content="Linter: ruff (configured in pyproject.toml).",
            confidence=0.95, source_file="pyproject.toml", tags=["linting", "ruff"],
        ))
    return out


def _scan_requirements_txt(root: Path) -> List[ScannedFact]:
    p = root / "requirements.txt"
    if not p.is_file():
        return []
    out: List[ScannedFact] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)", line)
        if not m:
            continue
        pkg = m.group(1)
        out.append(ScannedFact(
            content=f"Uses Python package `{pkg}` (from requirements.txt: `{line}`).",
            confidence=0.93, source_file="requirements.txt",
            tags=["python", "dep", pkg.lower()],
        ))
    return out[:60]


def _scan_dockerfile(root: Path) -> List[ScannedFact]:
    p = root / "Dockerfile"
    if not p.is_file():
        return []
    out: List[ScannedFact] = []
    text = p.read_text()
    # Base image
    m = re.search(r"^\s*FROM\s+([^\s]+)", text, re.MULTILINE | re.IGNORECASE)
    if m:
        base = m.group(1)
        out.append(ScannedFact(
            content=f"Container base image: `{base}` (from Dockerfile).",
            confidence=0.97, source_file="Dockerfile", tags=["docker", "runtime"],
        ))
    # Exposed ports
    for port_m in re.finditer(r"^\s*EXPOSE\s+(\d+)", text, re.MULTILINE | re.IGNORECASE):
        out.append(ScannedFact(
            content=f"Service exposes port `{port_m.group(1)}` (Dockerfile EXPOSE).",
            confidence=0.95, source_file="Dockerfile", tags=["docker", "network"],
        ))
    # CMD/ENTRYPOINT
    cmd_m = re.search(r"^\s*(CMD|ENTRYPOINT)\s+(.+)$", text, re.MULTILINE | re.IGNORECASE)
    if cmd_m:
        cmd_text = cmd_m.group(2).strip()[:120]
        out.append(ScannedFact(
            content=f"Container startup command: `{cmd_text}` (Dockerfile {cmd_m.group(1)}).",
            type="procedure",
            confidence=0.90, source_file="Dockerfile", tags=["docker", "startup"],
        ))
    return out


def _scan_compose(root: Path) -> List[ScannedFact]:
    for name in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
        p = root / name
        if p.is_file():
            break
    else:
        return []
    out: List[ScannedFact] = []
    text = p.read_text()
    # Service-name extraction — looks like top-level `services:` block with
    # 2-space-indented keys. Conservative regex, won't catch all valid YAMLs.
    services_block = re.search(r"^services:\s*\n((?:[ \t]+.+\n?)+)", text, re.MULTILINE)
    if services_block:
        body = services_block.group(1)
        for svc_m in re.finditer(r"^[ \t]{2,4}([A-Za-z0-9_\-]+):\s*$", body, re.MULTILINE):
            svc = svc_m.group(1)
            out.append(ScannedFact(
                content=f"Docker Compose service: `{svc}` (defined in {p.name}).",
                confidence=0.88, source_file=p.name, tags=["docker-compose", svc],
            ))
    return out


def _scan_gitignore(root: Path) -> List[ScannedFact]:
    p = root / ".gitignore"
    if not p.is_file():
        return []
    text = p.read_text()
    out: List[ScannedFact] = []
    # Heuristic: presence of these patterns implies the stack
    indicators = {
        "__pycache__": "Python project (Python bytecode is gitignored).",
        "node_modules": "JavaScript/TypeScript project (node_modules is gitignored).",
        "target/": "Rust or Java project (target/ is gitignored).",
        "vendor/": "Go or PHP project (vendor/ is gitignored).",
        ".venv": "Python project using a virtual environment.",
        "venv/": "Python project using a virtual environment.",
        ".env": "Project uses `.env` files for secrets (and they are gitignored — good).",
    }
    for pattern, fact in indicators.items():
        if re.search(rf"^{re.escape(pattern)}/?\s*$", text, re.MULTILINE):
            out.append(ScannedFact(
                content=fact, confidence=0.82, source_file=".gitignore",
                tags=["stack-inference"],
            ))
    return out


def _scan_ci(root: Path) -> List[ScannedFact]:
    wf_dir = root / ".github" / "workflows"
    if not wf_dir.is_dir():
        return []
    out: List[ScannedFact] = []
    workflow_files = list(wf_dir.glob("*.yml")) + list(wf_dir.glob("*.yaml"))
    if workflow_files:
        out.append(ScannedFact(
            content=f"CI: GitHub Actions ({len(workflow_files)} workflow file(s) in .github/workflows/).",
            confidence=0.97, source_file=".github/workflows/",
            tags=["ci", "github-actions"],
        ))
    return out


def _scan_test_layout(root: Path) -> List[ScannedFact]:
    out: List[ScannedFact] = []
    for candidate in ("tests/", "test/", "__tests__/", "spec/"):
        d = root / candidate.rstrip("/")
        if d.is_dir():
            n_files = sum(1 for _ in d.rglob("*") if _.is_file())
            out.append(ScannedFact(
                content=f"Tests live in `{candidate}` ({n_files} files).",
                type="preference",
                confidence=0.92, source_file=candidate, tags=["testing", "layout"],
            ))
            break
    return out


def _scan_readme(root: Path) -> List[ScannedFact]:
    p = root / "README.md"
    if not p.is_file():
        return []
    out: List[ScannedFact] = []
    text = p.read_text()
    # First H1 → project tagline
    m = re.search(r"^#\s+(.+?)$", text, re.MULTILINE)
    if m:
        title = m.group(1).strip()
        if len(title) < 100:
            out.append(ScannedFact(
                content=f"Project title (per README.md): `{title}`.",
                confidence=0.95, source_file="README.md", tags=["meta"],
            ))
    # First non-heading paragraph after the title → tagline
    para_m = re.search(r"^>?\s*(.+?)\n\s*$", text, re.MULTILINE)
    if para_m and len(para_m.group(1)) < 250:
        out.append(ScannedFact(
            content=f"README tagline: \"{para_m.group(1).strip()}\".",
            confidence=0.80, source_file="README.md", tags=["meta"],
        ))
    return out


# ---------------------------------------------------------------------------
# Notable-dep filter
# ---------------------------------------------------------------------------

# Packages whose presence implies architectural decisions worth surfacing.
# Everything not in this set is still included but with slightly lower
# confidence so the noise can be down-ranked by callers.
_NOTABLE_NPM_DEPS = {
    "express", "fastify", "next", "react", "vue", "svelte", "nuxt",
    "stripe", "axios", "prisma", "drizzle-orm", "typeorm", "mongoose",
    "redis", "ioredis", "pg", "mysql", "mysql2", "supabase",
    "tailwindcss", "vite", "webpack", "rollup", "esbuild",
    "vitest", "jest", "playwright", "cypress",
    "zod", "yup", "trpc", "graphql", "apollo-server",
    "@anthropic-ai/sdk", "openai", "@modelcontextprotocol/sdk",
}


def _filter_notable_deps(deps: dict) -> dict:
    notable = {k: v for k, v in deps.items() if k.lower() in _NOTABLE_NPM_DEPS or k.startswith("@")}
    # Fall back to top N alphabetical if nothing notable matched
    if not notable:
        notable = dict(list(deps.items())[:15])
    return notable


# ---------------------------------------------------------------------------
# Scanner registry
# ---------------------------------------------------------------------------

_SCANNERS: List[Callable[[Path], List[ScannedFact]]] = [
    _scan_package_json,
    _scan_pyproject_toml,
    _scan_requirements_txt,
    _scan_dockerfile,
    _scan_compose,
    _scan_gitignore,
    _scan_ci,
    _scan_test_layout,
    _scan_readme,
]


# ---------------------------------------------------------------------------
# Promotion helpers — called by `skein up` / the daemon
# ---------------------------------------------------------------------------

# Threshold above which a scanned fact is auto-promoted into the fragments
# table; everything else lands in the ``extraction_candidates`` review queue.
AUTO_PROMOTE_THRESHOLD = 0.90
DISCARD_THRESHOLD = 0.50


def classify(fact: ScannedFact) -> str:
    """Return one of {"auto", "queue", "discard"} based on confidence."""
    if fact.confidence >= AUTO_PROMOTE_THRESHOLD:
        return "auto"
    if fact.confidence < DISCARD_THRESHOLD:
        return "discard"
    return "queue"
