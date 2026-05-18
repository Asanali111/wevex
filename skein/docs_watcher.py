"""Passive docs scanner — indexes the project's own markdown documentation.

iter 19: scanners in ``skein/scanner.py`` extract implicit facts from
machine-readable manifests (``package.json``, ``pyproject.toml``, ``Dockerfile``).
That answers *"what stack is this?"* but not *"what does this project DO?"*.
The README, CHANGELOG, ADRs, and ``docs/`` tree carry that information — and
because the LLM-facing bus only knew scanner facts, ``recall "project state"``
was returning *"Uses Python package httpx"* instead of the README intro.

This module reads the project's own markdown documentation and emits
``ScannedFact`` records that flow through the same ``promote_scanned_facts``
pipeline as the code scanner. Topic keys are stable across runs so re-running
``skein up`` supersedes old fragments instead of stacking copies.

Design rules:
- **Heuristic-only.** No LLM calls, no network. Pure file reads + regex.
- **Deterministic.** Discovered paths are sorted before processing, so the same
  input tree always yields the same fact ordering.
- **Cheap.** Single pass, ≤100 KB per file. UTF-8 with ``errors='replace'``.
- **Respects gitignore.** Minimal manual parser so we don't pull in a new dep.
- **No Obsidian dependency.** Project-local docs only.

Fragment shapes:
- README/CONTRIBUTING/short docs (≤2000 chars): one ``state`` fragment per file.
- Longer docs: split by top-level ``^# `` / ``^## `` headings, each becomes a
  fragment whose body is truncated at ~800 chars.
- ADR files (filename matches ``ADR-\\d+`` or path contains ``/adr/`` /
  ``/decisions/``): typed ``decision`` instead of ``state``, confidence 0.92.
- CHANGELOG: parsed per ``^## `` entry, cap 10 most recent.
- LICENSE: single ``fact`` fragment with tag ``license``.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from .scanner import ScannedFact

logger = logging.getLogger("skein.docs_watcher")


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

# Skip files larger than this (cap on per-doc read).
MAX_FILE_BYTES = 100 * 1024  # 100 KB

# Files at or below this size emit a single fragment with the full body.
SHORT_FILE_CHARS = 2000

# Per-section body truncation when splitting longer files by heading.
SECTION_BODY_TRUNCATE = 800

# Maximum total length of a section fragment (heading + body + truncate marker).
SECTION_FRAGMENT_MAX = 1000

# CHANGELOG entries cap — keep only the most recent N versions to avoid bloat.
CHANGELOG_ENTRY_CAP = 10

# Confidence levels.
DOCS_CONFIDENCE = 0.95
ADR_CONFIDENCE = 0.92


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def scan_docs(repo: Path) -> list[ScannedFact]:
    """Scan a project's markdown documentation and return ScannedFact list.

    Discovery patterns (all relative to ``repo``):
      - ``README*.md``, ``README*`` (no extension)
      - ``CHANGELOG*.md``
      - ``CONTRIBUTING*.md``
      - ``LICENSE*`` (short ``fact`` fragment)
      - ``docs/**/*.md``, ``doc/**/*.md``
      - ``adr/**/*.md``, ``ADR/**/*.md``, ``decisions/**/*.md``,
        ``architecture/**/*.md``
      - ``.skein/docs/**/*.md`` (project-specific overrides)

    Returns ``[]`` when the directory has no recognizable docs (or doesn't
    exist).
    """
    repo = Path(repo).resolve()
    if not repo.is_dir():
        return []

    ignored = _load_gitignore_patterns(repo)
    paths = _discover_doc_paths(repo, ignored)
    if not paths:
        return []

    facts: list[ScannedFact] = []
    for path in paths:
        try:
            facts.extend(_process_file(repo, path))
        except Exception:
            logger.debug("docs_watcher failed on %s", path, exc_info=True)
    return facts


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


# Patterns are processed in declaration order; results are sorted at the end
# so the union is deterministic regardless of glob walk order.
_GLOB_PATTERNS: tuple[str, ...] = (
    "README*",
    "CHANGELOG*",
    "CONTRIBUTING*",
    "LICENSE*",
    "docs/**/*.md",
    "doc/**/*.md",
    "adr/**/*.md",
    "ADR/**/*.md",
    "decisions/**/*.md",
    "architecture/**/*.md",
    ".skein/docs/**/*.md",
)


def _discover_doc_paths(repo: Path, ignored: set[str]) -> list[Path]:
    """Return a sorted, deduped list of doc paths under ``repo``."""
    seen: set[Path] = set()
    for pattern in _GLOB_PATTERNS:
        for hit in repo.glob(pattern):
            if not hit.is_file():
                continue
            if _is_ignored(repo, hit, ignored):
                continue
            if not _is_text_doc(hit):
                continue
            try:
                if hit.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            seen.add(hit.resolve())
    return sorted(seen, key=lambda p: str(p.relative_to(repo)))


def _is_text_doc(p: Path) -> bool:
    """Filter to plausible text docs.

    README/LICENSE files often have no extension; the rest should be ``.md``.
    Anything else (``.pdf``, ``.png``, ``.docx``) is skipped.
    """
    name = p.name
    if name.lower().startswith(("readme", "license", "contributing", "changelog")):
        # No-extension or .md/.txt/.rst — accept text-ish docs.
        ext = p.suffix.lower()
        if ext in ("", ".md", ".markdown", ".txt", ".rst"):
            return True
        return False
    # All other matches must end in .md/.markdown.
    return p.suffix.lower() in (".md", ".markdown")


# ---------------------------------------------------------------------------
# .gitignore (minimal)
# ---------------------------------------------------------------------------

# Always skip these directory prefixes even if .gitignore is missing. Hidden
# dirs are blanket-skipped (see _is_ignored) with the explicit ``.skein/docs/``
# carve-out so project-local overrides still get picked up.
_BUILTIN_SKIP_DIRS: tuple[str, ...] = (
    "node_modules/",
    ".venv/",
    "venv/",
    "_archive_v2/",
    "__pycache__/",
    "dist/",
    "build/",
)


def _load_gitignore_patterns(repo: Path) -> set[str]:
    """Read the project's ``.gitignore`` and return a set of normalized
    path-prefix patterns. Empty set if no .gitignore exists.

    We intentionally do NOT implement full gitignore semantics — no negation,
    no glob expansion. Just simple prefix matches against the repo-relative
    path. Good enough for the common case of `docs/private/` style entries.
    """
    out: set[str] = set()
    gi = repo / ".gitignore"
    if not gi.is_file():
        return out
    try:
        text = gi.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        # Strip a leading slash (gitignore treats it as anchored-to-repo;
        # our matcher already does prefix matching from repo root).
        if line.startswith("/"):
            line = line[1:]
        out.add(line)
    return out


def _is_ignored(repo: Path, path: Path, gitignore: set[str]) -> bool:
    """Return True if ``path`` should be skipped (gitignore or builtin)."""
    try:
        rel = path.relative_to(repo)
    except ValueError:
        return True
    rel_str = str(rel).replace(os.sep, "/")

    # Builtin hidden-dir skip, with .skein/docs carve-out.
    parts = rel.parts
    for part in parts[:-1]:  # parents only — the filename itself can start with "."
        if part.startswith("."):
            # Allow project-local override directory ".skein/docs/".
            if part == ".skein" and len(parts) >= 3 and parts[1] == "docs":
                continue
            return True

    for prefix in _BUILTIN_SKIP_DIRS:
        if rel_str.startswith(prefix):
            return True

    for pat in gitignore:
        if not pat:
            continue
        # Strip trailing slash so "docs/private/" matches "docs/private".
        normalized = pat.rstrip("/")
        if not normalized:
            continue
        if rel_str == normalized or rel_str.startswith(normalized + "/"):
            return True
    return False


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def _process_file(repo: Path, path: Path) -> list[ScannedFact]:
    """Read one doc file and emit zero or more ScannedFacts.

    Classification order (overlapping rules in the spec resolved here):
      1. LICENSE → single ``fact`` fragment, tag ``license``.
      2. CHANGELOG → split by ``^## `` entries, cap at most recent N.
      3. ADR (filename ``ADR-\\d+`` or path contains ``/adr/`` or
         ``/decisions/``) → typed ``decision``.
      4. Other docs → short → single ``state``; long → split by headings.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    text = text.strip()
    if not text:
        return []

    rel = path.relative_to(repo)
    rel_str = str(rel).replace(os.sep, "/")
    stem = path.stem if path.suffix else path.name
    name_lower = path.name.lower()

    # 1) LICENSE
    if name_lower.startswith("license"):
        return [_make_license_fact(rel_str, text, stem)]

    # 2) CHANGELOG
    if name_lower.startswith("changelog"):
        return _split_changelog(rel_str, text, stem)

    # 3) ADR
    if _is_adr(rel_str, path.name):
        return _split_adr(rel_str, text, stem)

    # 4) Other docs
    if len(text) <= SHORT_FILE_CHARS:
        return [_make_short_doc_fact(rel_str, text, stem)]
    return _split_by_heading(rel_str, text, stem)


# ---------------------------------------------------------------------------
# Short-doc, long-doc, license helpers
# ---------------------------------------------------------------------------


def _make_short_doc_fact(rel_str: str, text: str, stem: str) -> ScannedFact:
    tags = ["docs", stem]
    territory = "docs"
    name_lower = stem.lower()
    if name_lower.startswith("readme"):
        tags.append("readme")
        territory = "readme"
    elif name_lower.startswith("contributing"):
        tags.append("contributing")
        territory = "contributing"
    return ScannedFact(
        content=text,
        type="state",
        confidence=DOCS_CONFIDENCE,
        source_file=rel_str,
        tags=tags,
        territory=territory,
        topic_key=f"docs:{stem}",
    )


def _make_license_fact(rel_str: str, text: str, stem: str) -> ScannedFact:
    # License bodies can be long; truncate so we never blow past the
    # SECTION_FRAGMENT_MAX budget when this lands in AGENTS.md.
    body = text
    if len(body) > SECTION_FRAGMENT_MAX:
        body = body[:SECTION_BODY_TRUNCATE].rstrip() + "\n…(truncated)"
    return ScannedFact(
        content=body,
        type="fact",
        confidence=DOCS_CONFIDENCE,
        source_file=rel_str,
        tags=["docs", "license", stem],
        territory="license",
        topic_key=f"docs:{stem}",
    )


_HEADING_RE = re.compile(r"^(#{1,2})\s+(.+?)\s*$", re.MULTILINE)


def _split_by_heading(rel_str: str, text: str, stem: str) -> list[ScannedFact]:
    """Split ``text`` into one fragment per top-level (H1/H2) heading.

    Headings deeper than H2 are kept inside their parent section. Content
    before the first heading is dropped (it's almost always a license blurb
    or boilerplate).
    """
    sections = _extract_sections(text)
    if not sections:
        # No headings found — treat the whole file as a single fragment,
        # truncated.
        return [
            ScannedFact(
                content=_truncate_body(text),
                type="state",
                confidence=DOCS_CONFIDENCE,
                source_file=rel_str,
                tags=["docs", stem],
                territory="docs",
                topic_key=f"docs:{stem}",
            )
        ]

    base_tags = ["docs", stem]
    territory = "docs"
    name_lower = stem.lower()
    if name_lower.startswith("readme"):
        base_tags.append("readme")
        territory = "readme"
    elif name_lower.startswith("contributing"):
        base_tags.append("contributing")
        territory = "contributing"

    out: list[ScannedFact] = []
    for heading, body in sections:
        slug = _slugify(heading)
        fragment_body = body.strip()
        if len(fragment_body) > SECTION_BODY_TRUNCATE:
            fragment_body = fragment_body[:SECTION_BODY_TRUNCATE].rstrip() + "\n…(truncated)"
        full = f"# {heading}\n\n{fragment_body}".strip()
        if len(full) > SECTION_FRAGMENT_MAX:
            full = full[:SECTION_FRAGMENT_MAX - len("\n…(truncated)")].rstrip() + "\n…(truncated)"
        tags = list(base_tags)
        if slug:
            tags.append(slug)
        out.append(ScannedFact(
            content=full,
            type="state",
            confidence=DOCS_CONFIDENCE,
            source_file=rel_str,
            tags=tags,
            territory=territory,
            topic_key=f"docs:{stem}:{slug or 'section'}",
        ))
    return out


def _truncate_body(s: str) -> str:
    if len(s) <= SECTION_FRAGMENT_MAX:
        return s.strip()
    return s[:SECTION_FRAGMENT_MAX - len("\n…(truncated)")].rstrip() + "\n…(truncated)"


def _extract_sections(text: str) -> list[tuple[str, str]]:
    """Return [(heading_title, body), …] split by ``^#`` or ``^##`` lines.

    The body of each section runs up to the next H1/H2 heading. H3+ headings
    are kept inline as part of the body.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return []
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((title, body))
    return sections


# ---------------------------------------------------------------------------
# ADR detection
# ---------------------------------------------------------------------------

_ADR_FILENAME_RE = re.compile(r"^ADR[-_]?\d+", re.IGNORECASE)


def _is_adr(rel_str: str, filename: str) -> bool:
    """An ADR is a markdown file under ``/adr/`` / ``/decisions/`` /
    ``/architecture/`` OR whose filename starts with ``ADR-<number>``.
    """
    if _ADR_FILENAME_RE.match(filename):
        return True
    lowered = rel_str.lower()
    return any(seg in lowered for seg in ("/adr/", "/decisions/", "/architecture/"))


def _split_adr(rel_str: str, text: str, stem: str) -> list[ScannedFact]:
    """Emit one or more ``decision`` fragments for an ADR.

    Short ADRs become a single fragment. Longer ones are split by H1/H2 so
    "Context", "Decision", "Consequences" become individually queryable.
    """
    adr_number = _adr_number(stem)
    base_tags: list[str] = ["docs", "adr", stem]
    if adr_number:
        base_tags.append(adr_number)

    territory = f"adr/{adr_number}" if adr_number else "adr"

    if len(text) <= SHORT_FILE_CHARS:
        return [ScannedFact(
            content=text,
            type="decision",
            confidence=ADR_CONFIDENCE,
            source_file=rel_str,
            tags=base_tags,
            territory=territory,
            topic_key=f"docs:{stem}",
        )]

    sections = _extract_sections(text)
    if not sections:
        return [ScannedFact(
            content=_truncate_body(text),
            type="decision",
            confidence=ADR_CONFIDENCE,
            source_file=rel_str,
            tags=base_tags,
            territory=territory,
            topic_key=f"docs:{stem}",
        )]

    out: list[ScannedFact] = []
    for heading, body in sections:
        slug = _slugify(heading)
        body = body.strip()
        if len(body) > SECTION_BODY_TRUNCATE:
            body = body[:SECTION_BODY_TRUNCATE].rstrip() + "\n…(truncated)"
        full = f"# {heading}\n\n{body}".strip()
        if len(full) > SECTION_FRAGMENT_MAX:
            full = full[:SECTION_FRAGMENT_MAX - len("\n…(truncated)")].rstrip() + "\n…(truncated)"
        tags = list(base_tags)
        if slug:
            tags.append(slug)
        out.append(ScannedFact(
            content=full,
            type="decision",
            confidence=ADR_CONFIDENCE,
            source_file=rel_str,
            tags=tags,
            territory=territory,
            topic_key=f"docs:{stem}:{slug or 'section'}",
        ))
    return out


def _adr_number(stem: str) -> Optional[str]:
    m = re.match(r"^ADR[-_]?(\d+)", stem, re.IGNORECASE)
    if m:
        return f"ADR-{m.group(1)}"
    return None


# ---------------------------------------------------------------------------
# Changelog parsing
# ---------------------------------------------------------------------------

_CHANGELOG_ENTRY_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _split_changelog(rel_str: str, text: str, stem: str) -> list[ScannedFact]:
    """Parse a CHANGELOG into one fragment per ``^## `` entry.

    Conventions vary; we just pull each H2 block as-is, slug the header for
    the topic key, and cap at CHANGELOG_ENTRY_CAP most recent entries
    (in file order — top-of-file is newest by the usual convention).
    """
    matches = list(_CHANGELOG_ENTRY_RE.finditer(text))
    if not matches:
        # Fall back to treating the whole file as a single state fragment.
        return [_make_short_doc_fact(rel_str, text[:SHORT_FILE_CHARS], stem)]

    out: list[ScannedFact] = []
    for i, m in enumerate(matches):
        if i >= CHANGELOG_ENTRY_CAP:
            break
        version = m.group(1).strip()
        slug = _slugify(version) or f"entry-{i}"
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if len(body) > SECTION_BODY_TRUNCATE:
            body = body[:SECTION_BODY_TRUNCATE].rstrip() + "\n…(truncated)"
        full = f"## {version}\n\n{body}".strip()
        if len(full) > SECTION_FRAGMENT_MAX:
            full = full[:SECTION_FRAGMENT_MAX - len("\n…(truncated)")].rstrip() + "\n…(truncated)"
        out.append(ScannedFact(
            content=full,
            type="state",
            confidence=DOCS_CONFIDENCE,
            source_file=rel_str,
            tags=["docs", "changelog", stem, slug],
            territory="changelog",
            topic_key=f"changelog:{slug}",
        ))
    return out


# ---------------------------------------------------------------------------
# Slug helper
# ---------------------------------------------------------------------------

_SLUG_NONALNUM = re.compile(r"[^a-z0-9]+")


def _slugify(s: str) -> str:
    """Kebab-case slug — lowercased, alnum only, joined by ``-``.

    Used for heading-derived tags so ``recall "installation"`` matches the
    README's Install section even when the heading was "Installation Steps".
    """
    s = s.lower().strip()
    s = _SLUG_NONALNUM.sub("-", s).strip("-")
    return s
