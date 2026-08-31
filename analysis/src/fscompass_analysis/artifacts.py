"""Locating the repository-root shared artifacts.

SPEC.md §37.1 requires both platforms and the analysis tooling to consume the *same* files
rather than copies, because independently translated constants are a common source of false
parity.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT_ENVIRONMENT_VARIABLE = "FSC_REPO_ROOT"

_MARKER = Path("config") / "precision-profile-v1.json"


def repository_root() -> Path:
    """Return the repository root, or raise rather than fall back to a copy."""
    override = os.environ.get(REPO_ROOT_ENVIRONMENT_VARIABLE)
    if override:
        root = Path(override).resolve()
        if not (root / _MARKER).is_file():
            raise FileNotFoundError(
                f"{REPO_ROOT_ENVIRONMENT_VARIABLE}={root} does not contain {_MARKER}"
            )
        return root

    for candidate in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (candidate / _MARKER).is_file():
            return candidate
    raise FileNotFoundError(
        f"could not locate the repository root from {__file__}; a runtime that cannot find "
        "the shared artifacts must fail rather than fall back to a copy"
    )


def artifact_path(relative: str) -> Path:
    path = repository_root() / relative
    if not path.is_file():
        raise FileNotFoundError(f"required shared artifact is missing: {relative}")
    return path


def load_json(relative: str) -> Any:
    return json.loads(artifact_path(relative).read_text(encoding="utf-8"))


def sha256_of(relative: str) -> str:
    """``sha256:<hex>``, the form §22 and §24 use for every hashed artifact."""
    digest = hashlib.sha256(artifact_path(relative).read_bytes()).hexdigest()
    return f"sha256:{digest}"


PRECISION_PROFILE = "config/precision-profile-v1.json"
FENG_SHUI_RULES = "config/feng-shui-rules-v1.json"
GRADE_REACHABILITY_CLAIMS = "testdata/grade-reachability-claims-v1.json"
EXAMPLE_ENGINE_OUTPUT_EVENT = "testdata/telemetry-event-engine-output-v1.example.json"
EXAMPLE_SESSION_MANIFEST = "testdata/session-manifest-v1.example.json"
