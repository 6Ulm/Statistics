"""The frozen ``fixtures-v1`` shared contract (SPEC.md §37.1).

Every fixture below is read from the repository root, never copied into a runtime. §37.1:
"give both agents the same files and tolerances, with neither editing shared fixtures
unilaterally; a contract change requires one reviewed change set, regenerated fixtures, both
test suites, and a new fixture version".
"""

from __future__ import annotations

from typing import Any

from .artifacts import load_json, sha256_of

FIXTURE_VERSION = "fixtures-v1"

CIRCULAR_MATH = "testdata/angles/circular-math-v1.json"
ESTIMATORS = "testdata/angles/estimators-v1.json"
CIRCULAR_AGGREGATE = "testdata/angles/circular-aggregate-v1.json"
ATTITUDE_GOLDEN = "testdata/quaternions/attitude-golden-v1.json"
FENGSHUI_CLASSIFICATION = "testdata/fengshui/classification-v1.json"
FENGSHUI_REFERENCE_TRANSFORM = "testdata/fengshui/reference-transform-v1.json"
UNCERTAINTY_COMPOSITION = "testdata/uncertainty/composition-v1.json"
MAGNETIC_CLASSIFICATION = "testdata/magnetic/classification-v1.json"
REFERENCE_RESOLUTION = "testdata/reference/resolution-v1.json"
STATE_REDUCER = "testdata/state/reducer-v1.json"
CERTIFICATION_KEY = "testdata/certification/key-v1.json"
TELEMETRY_CODEC = "testdata/telemetry/codec-v1.json"

MANIFEST = "testdata/fixtures-v1.manifest.json"

#: Everything the freeze covers: the config both platforms read, the schemas that validate it,
#: and every generated golden fixture.
FROZEN_ARTIFACTS: tuple[str, ...] = (
    "third_party/noaa-wmm/sha256.txt",
    "config/feng-shui-rules-v1.json",
    "config/precision-profile-v1.json",
    "schemas/feng-shui-rules-v1.schema.json",
    "schemas/precision-profile-v1.schema.json",
    "schemas/session-manifest-v1.schema.json",
    "schemas/telemetry-event-v1.schema.json",
    "testdata/grade-reachability-claims-v1.json",
    "testdata/session-manifest-v1.example.json",
    "testdata/telemetry-event-engine-output-v1.example.json",
    CIRCULAR_MATH,
    ESTIMATORS,
    CIRCULAR_AGGREGATE,
    ATTITUDE_GOLDEN,
    FENGSHUI_CLASSIFICATION,
    FENGSHUI_REFERENCE_TRANSFORM,
    UNCERTAINTY_COMPOSITION,
    MAGNETIC_CLASSIFICATION,
    REFERENCE_RESOLUTION,
    STATE_REDUCER,
    CERTIFICATION_KEY,
    TELEMETRY_CODEC,
)


def load(relative: str) -> Any:
    return load_json(relative)


def build_manifest() -> dict[str, Any]:
    """The ``fixtures-v1`` hash manifest: what is frozen, and at which bytes."""
    return {
        "fixtureVersion": FIXTURE_VERSION,
        "specSections": ["§37.1"],
        "note": (
            "Frozen before platform work diverges. Neither platform agent edits these "
            "unilaterally; a contract change requires one reviewed change set, regenerated "
            "fixtures, both test suites, and a new fixture version."
        ),
        "artifacts": {relative: sha256_of(relative) for relative in FROZEN_ARTIFACTS},
    }
