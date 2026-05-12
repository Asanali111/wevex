"""Labeled benchmark corpora. Stable IDs — do not renumber existing entries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).parent


def load(name: str) -> List[Dict[str, Any]]:
    return json.loads((_HERE / name).read_text())


def fragments() -> List[Dict[str, Any]]:
    return load("fragments.json")


def labeled_queries() -> List[Dict[str, Any]]:
    return load("labeled_queries.json")


def commits() -> List[Dict[str, Any]]:
    return load("commits.json")


def code_files() -> Dict[str, str]:
    """Synthetic code files for ingest. ``{relative_path: contents}``."""
    return json.loads((_HERE / "code_files.json").read_text())
