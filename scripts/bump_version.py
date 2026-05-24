#!/usr/bin/env python3
"""Bump the Skein version in pyproject.toml and skein/__init__.py.

Usage:
    python scripts/bump_version.py           # bump patch (default)
    python scripts/bump_version.py patch     # same
    python scripts/bump_version.py minor     # 0.1.x → 0.2.0
    python scripts/bump_version.py major     # 0.x.y → 1.0.0
    python scripts/bump_version.py 0.3.1     # set exact version
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _read_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise ValueError("version not found in pyproject.toml")
    return m.group(1)


def _bump(version: str, part: str) -> str:
    parts = [int(x) for x in version.split(".")]
    while len(parts) < 3:
        parts.append(0)
    if part == "major":
        parts[0] += 1; parts[1] = 0; parts[2] = 0
    elif part == "minor":
        parts[1] += 1; parts[2] = 0
    else:
        parts[2] += 1
    return ".".join(str(p) for p in parts)


def _replace_version(path: Path, old: str, new: str, pattern: str) -> None:
    text = path.read_text()
    # re.MULTILINE so ^ matches start-of-line, not just start-of-string —
    # otherwise we couldn't anchor `^version =` against a multi-line TOML.
    updated = re.sub(pattern, lambda m: m.group(0).replace(old, new), text, flags=re.MULTILINE)
    if updated == text:
        raise ValueError(f"Version string not found in {path}")
    path.write_text(updated)


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "patch"

    old = _read_version()

    if re.match(r"^\d+\.\d+", arg):
        new = arg
    elif arg in ("major", "minor", "patch"):
        new = _bump(old, arg)
    else:
        print(f"Unknown argument: {arg!r}. Use major/minor/patch or an explicit version.", file=sys.stderr)
        sys.exit(1)

    _replace_version(
        ROOT / "pyproject.toml", old, new,
        r'^version\s*=\s*"[^"]+"',
    )
    _replace_version(
        ROOT / "skein" / "__init__.py", old, new,
        r'__version__\s*=\s*"[^"]+"',
    )

    print(f"Bumped {old} → {new}")
    print()
    print("Next steps:")
    print(f"  git add pyproject.toml skein/__init__.py")
    print(f"  git commit -m 'chore: bump version to {new}'")
    print(f"  git push origin main")
    print()
    print("GitHub Actions will build and publish to PyPI automatically.")


if __name__ == "__main__":
    main()
