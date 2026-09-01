"""SPEC.md §33.1, R67/R68 — one allowlisted signed-difference implementation per runtime.

§9 owns one normative contract; each executable runtime (Android core, iOS core, analysis
tooling) has exactly **one** allowlisted implementation of ``shortestSignedDifferenceDeg``, and
every other call site uses that runtime's shared utility. ``shortestTargetDeltaDeg`` and
``absoluteCircularDifferenceDeg`` are thin delegating wrappers, and no local ``deltaDeg`` alias
is permitted.

§33.1 also fixes how the audit must work: it scans **source** ``atan2`` call sites outside the
allowlisted implementation files, and it MUST NOT reject explanatory prose — "Documentation and
tests may quote a prohibited formula as text, so a blind repository-wide grep is not sufficient".
This audit therefore reads production source sets only, and the ``docs/`` prose plus the
discrimination test in ``test_circular.py`` that deliberately computes the prohibited formula
both stay legal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class Runtime:
    """One executable runtime and the files allowed to contain low-level angle math."""

    name: str
    #: Production source globs. Tests and docs are deliberately out of scope.
    source_globs: tuple[str, ...]
    #: The single file allowed to implement the signed circular difference.
    signed_difference_implementation: str
    #: Files additionally allowed to call ``atan2``, for the separately allowlisted
    #: bearing-projection and circular-mean implementations (§33.1).
    additional_atan2_allowlist: tuple[str, ...]


RUNTIMES: tuple[Runtime, ...] = (
    Runtime(
        name="analysis",
        source_globs=("analysis/src/fscompass_analysis/*.py",),
        signed_difference_implementation="analysis/src/fscompass_analysis/circular.py",
        additional_atan2_allowlist=("analysis/src/fscompass_analysis/frames.py",),
    ),
    Runtime(
        name="android",
        source_globs=(
            "android/heading-core/src/main/kotlin/**/*.kt",
            "android/fengshui-core/src/main/kotlin/**/*.kt",
        ),
        signed_difference_implementation=(
            "android/heading-core/src/main/kotlin/com/fengshuicompass/headingcore/math/"
            "CircularMath.kt"
        ),
        additional_atan2_allowlist=(
            "android/heading-core/src/main/kotlin/com/fengshuicompass/headingcore/frames/"
            "Frames.kt",
        ),
    ),
    Runtime(
        name="ios",
        source_globs=(
            "ios/HeadingCore/Sources/HeadingCore/*.swift",
            "ios/FengShuiCore/Sources/FengShuiCore/*.swift",
        ),
        signed_difference_implementation="ios/HeadingCore/Sources/HeadingCore/CircularMath.swift",
        additional_atan2_allowlist=("ios/HeadingCore/Sources/HeadingCore/Frames.swift",),
    ),
)

_ATAN2 = re.compile(r"\batan2\b")
#: The signed-difference shape: an ``atan2`` whose arguments are a sine and a cosine.
_SIGNED_DIFFERENCE_FORMULA = re.compile(r"atan2\s*\(\s*[^)]*\bsin\b", re.IGNORECASE)
#: R68: no local alias for the one normative contract.
_FORBIDDEN_ALIAS = re.compile(r"\b(fun|def|func|let|val)\s+deltaDeg\b")


def _sources(repo_root: Path, runtime: Runtime) -> list[Path]:
    files: list[Path] = []
    for pattern in runtime.source_globs:
        files.extend(sorted(repo_root.glob(pattern)))
    return files


def _relative(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


@pytest.mark.parametrize("runtime", RUNTIMES, ids=lambda runtime: runtime.name)
def test_each_runtime_has_exactly_one_signed_difference_implementation(repo_root, runtime):
    """R68: "each runtime has one allowlisted implementation".

    A runtime whose sources are not present yet reports that fact rather than passing
    vacuously — an audit that silently finds nothing to audit is indistinguishable from one
    that passed.
    """
    sources = _sources(repo_root, runtime)
    if not sources:
        pytest.fail(
            f"runtime {runtime.name!r} has no source files matching {runtime.source_globs}; "
            "the audit cannot report a pass over an empty set"
        )
    implementations = [
        _relative(repo_root, path)
        for path in sources
        if _SIGNED_DIFFERENCE_FORMULA.search(path.read_text(encoding="utf-8"))
    ]
    assert implementations == [runtime.signed_difference_implementation], (
        f"{runtime.name}: expected exactly one signed-difference implementation at "
        f"{runtime.signed_difference_implementation}, found {implementations}"
    )


@pytest.mark.parametrize("runtime", RUNTIMES, ids=lambda runtime: runtime.name)
def test_atan2_call_sites_are_confined_to_allowlisted_files(repo_root, runtime):
    """§33.1: "CI MUST audit source-code ``atan2`` call sites outside those allowlisted
    implementation files and the separately allowlisted bearing-projection/circular-mean
    implementations"."""
    allowed = {runtime.signed_difference_implementation, *runtime.additional_atan2_allowlist}
    offenders = [
        _relative(repo_root, path)
        for path in _sources(repo_root, runtime)
        if _ATAN2.search(path.read_text(encoding="utf-8"))
        and _relative(repo_root, path) not in allowed
    ]
    assert not offenders, (
        f"{runtime.name}: atan2 outside the allowlist. A new signed-difference formula or "
        f"helper outside the approved implementation sites fails (§33.1). Offenders: {offenders}"
    )


@pytest.mark.parametrize("runtime", RUNTIMES, ids=lambda runtime: runtime.name)
def test_no_local_delta_deg_alias_exists(repo_root, runtime):
    """R68: "no local ``deltaDeg`` alias is permitted"."""
    offenders = [
        _relative(repo_root, path)
        for path in _sources(repo_root, runtime)
        if _FORBIDDEN_ALIAS.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"{runtime.name}: local deltaDeg alias found in {offenders}"


def test_the_audit_detects_a_second_implementation(tmp_path):
    """A gate that cannot fail is not a gate.

    The detector is pointed at a synthetic tree containing a duplicate implementation. No
    shipped file is touched (§37 rule 12).
    """
    (tmp_path / "src").mkdir()
    good = tmp_path / "src" / "CircularMath.py"
    good.write_text("d = degrees(atan2(sin(r), cos(r)))\n", encoding="utf-8")
    sneaky = tmp_path / "src" / "Kpi.py"
    sneaky.write_text(
        "# a second definition, exactly the R67 defect\n"
        "e = degrees(atan2(sin(radians(m - g)), cos(radians(m - g))))\n",
        encoding="utf-8",
    )
    found = [
        path.name
        for path in sorted((tmp_path / "src").glob("*.py"))
        if _SIGNED_DIFFERENCE_FORMULA.search(path.read_text(encoding="utf-8"))
    ]
    assert found == ["CircularMath.py", "Kpi.py"]


def test_the_audit_does_not_reject_explanatory_prose(repo_root):
    """§33.1: documentation and tests may quote a prohibited formula as text.

    ``SPEC.md`` itself quotes ``atan2(sin(a-b), cos(a-b))`` while explaining why it is
    insufficient, and ``test_circular.py`` computes it deliberately to prove the antipode
    normalization is load-bearing. A blind repository-wide grep would reject both.
    """
    spec = (repo_root / "SPEC.md").read_text(encoding="utf-8")
    assert _SIGNED_DIFFERENCE_FORMULA.search(spec), (
        "SPEC.md is expected to quote the prohibited formula in prose; if it no longer does, "
        "this test's premise needs revisiting rather than deleting"
    )
    discrimination_test = (repo_root / "analysis" / "tests" / "test_circular.py").read_text(
        encoding="utf-8"
    )
    assert _SIGNED_DIFFERENCE_FORMULA.search(discrimination_test)
    # Neither path is in any runtime's audited source set.
    for runtime in RUNTIMES:
        audited = {_relative(repo_root, path) for path in _sources(repo_root, runtime)}
        assert "SPEC.md" not in audited
        assert "analysis/tests/test_circular.py" not in audited


def test_the_delegating_wrappers_contain_no_angle_math_of_their_own(repo_root):
    """§9/R68: ``shortestTargetDeltaDeg`` and ``absoluteCircularDifferenceDeg`` are exact
    delegating wrappers, so each body is a single call to the one implementation."""
    source = (repo_root / "analysis" / "src" / "fscompass_analysis" / "circular.py").read_text(
        encoding="utf-8"
    )
    for wrapper, expected_call in (
        ("def shortest_target_delta_deg", "return shortest_signed_difference_deg(target, current)"),
        (
            "def absolute_circular_difference_deg",
            "return abs(shortest_signed_difference_deg(a, b))",
        ),
    ):
        start = source.index(wrapper)
        # Bound the body at the next top-level definition of any kind: a plain "\ndef "
        # search would run past a following decorator or class and swallow its docstring.
        ends = [
            index
            for index in (
                source.find("\ndef ", start + 1),
                source.find("\nclass ", start + 1),
                source.find("\n@", start + 1),
            )
            if index != -1
        ]
        body = source[start : min(ends)]
        assert expected_call in body, wrapper
        assert "atan2" not in body, wrapper
        assert "radians" not in body, wrapper
