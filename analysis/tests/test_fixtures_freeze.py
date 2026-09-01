"""SPEC.md §37.1 — the shared contract is frozen as ``fixtures-v1`` before platforms diverge.

§37.1's ordering is mandatory: "one integration owner completes Phases 0–1 and freezes the
contract before platform work diverges: commit schemas, config, canonical types,
angle/quaternion vectors, WMM vectors, replay fixtures, rule boundaries, and vendored NOAA
hashes; tag that baseline ``fixtures-v1``; give both agents the same files and tolerances, with
neither editing shared fixtures unilaterally".

These tests make the freeze checkable rather than asserted: the manifest must cover every
shared artifact, every hash must match the bytes on disk, and the generator that produced the
derived fixtures must reproduce them exactly.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from fscompass_analysis import artifacts, fixtures


@pytest.fixture(scope="module")
def manifest():
    return fixtures.load(fixtures.MANIFEST)


def test_the_manifest_declares_the_frozen_version(manifest):
    assert manifest["fixtureVersion"] == fixtures.FIXTURE_VERSION == "fixtures-v1"
    assert manifest["declaredAngleToleranceDeg"] == 1e-6


def test_every_frozen_artifact_hash_matches_the_bytes_on_disk(manifest):
    for relative, recorded in manifest["artifacts"].items():
        assert recorded == artifacts.sha256_of(relative), relative


def test_the_manifest_covers_every_shared_artifact(manifest):
    """A fixture outside the manifest is one a platform agent could edit unnoticed."""
    covered = set(manifest["artifacts"])
    for relative in fixtures.FROZEN_ARTIFACTS:
        assert relative in covered, relative


def test_the_manifest_freezes_the_declared_not_vendored_wmm_state(manifest):
    """D-2: freezing the *declared* absence makes a later vendoring a visible contract change.

    §37.1 lists "vendored NOAA hashes" among the artifacts to freeze. There are none, so what
    is frozen is the manifest that says so.
    """
    assert "third_party/noaa-wmm/sha256.txt" in manifest["artifacts"]
    text = artifacts.artifact_path("third_party/noaa-wmm/sha256.txt").read_text(encoding="utf-8")
    entries = [
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    assert entries == [], (
        "NOAA artifacts appear to have been vendored. That is a contract change: regenerate "
        "the fixtures, run both suites, and bump the fixture version (§37.1)."
    )


def test_every_shared_fixture_declares_its_version_and_sections():
    for relative in (
        fixtures.CIRCULAR_MATH,
        fixtures.ESTIMATORS,
        fixtures.CIRCULAR_AGGREGATE,
        fixtures.ATTITUDE_GOLDEN,
        fixtures.FENGSHUI_CLASSIFICATION,
        fixtures.FENGSHUI_REFERENCE_TRANSFORM,
        fixtures.UNCERTAINTY_COMPOSITION,
        fixtures.MAGNETIC_CLASSIFICATION,
        fixtures.REFERENCE_RESOLUTION,
        fixtures.STATE_REDUCER,
        fixtures.CERTIFICATION_KEY,
        fixtures.TELEMETRY_CODEC,
    ):
        document = fixtures.load(relative)
        assert document["fixtureVersion"] == fixtures.FIXTURE_VERSION, relative
        assert document["specSections"], relative


def test_the_generator_reproduces_the_committed_fixtures(repo_root):
    """§37.2: rerunning the generator must reproduce the committed files byte for byte.

    Without this the generator is documentation rather than provenance, and a hand-edited
    fixture would survive review.
    """
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "generate-shared-fixtures.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "committed fixtures are stale relative to the generator:\n"
        f"{result.stdout}\n{result.stderr}"
    )


def test_no_fixture_contains_a_nonfinite_literal():
    """§22.2 applies to fixtures too: "No exceptions, including fixtures".

    Checked by parsing rather than by grepping. ``testdata/telemetry/codec-v1.json`` carries
    the token ``NaN`` inside a *quoted string* on purpose — those are the malformed lines the
    decoder must reject — and a text search would flag that legitimate content while missing a
    genuinely nonfinite *value* written some other way.
    """
    import json

    def reject(token: str) -> None:
        raise AssertionError(f"nonstandard JSON literal {token!r}")

    for relative in fixtures.FROZEN_ARTIFACTS:
        if not relative.endswith(".json"):
            continue
        text = artifacts.artifact_path(relative).read_text(encoding="utf-8")
        json.loads(text, parse_constant=reject)


def test_the_codec_fixture_still_carries_its_rejected_documents():
    """The corollary: the malformed lines the decoder must reject are still present.

    Without this, "no fixture parses a nonfinite literal" could be satisfied by quietly
    deleting the negative cases.
    """
    document = fixtures.load(fixtures.TELEMETRY_CODEC)
    lines = [case["line"] for case in document["rejectedDocuments"]]
    assert any("NaN" in line for line in lines)
    assert any("Infinity" in line for line in lines)
