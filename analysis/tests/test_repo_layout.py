"""SPEC.md §4.1 repository layout.

"Names may follow local convention; boundaries MUST NOT." This test asserts the layout the
Phase 0 task creates, so a later refactor that dissolves a module boundary fails here rather
than being discovered when a diagnostic module has quietly become the production estimator.
"""

from __future__ import annotations

import pytest

REQUIRED_FILES = [
    "SPEC.md",
    "README.md",
    "docs/BENCHMARK.md",
    "docs/RISKS.md",
    "docs/IMPLEMENTATION_NOTES.md",
    "docs/TESTING.md",
    "docs/PRIVACY.md",
    "config/precision-profile-v1.json",
    "config/feng-shui-rules-v1.json",
    "schemas/precision-profile-v1.schema.json",
    "schemas/feng-shui-rules-v1.schema.json",
    "schemas/telemetry-event-v1.schema.json",
    "schemas/session-manifest-v1.schema.json",
    "third_party/noaa-wmm/UPSTREAM.md",
    "third_party/noaa-wmm/sha256.txt",
    "android/settings.gradle.kts",
    "android/gradle/libs.versions.toml",
    "ios/Package.swift",
    "analysis/pyproject.toml",
    "scripts/validate-fixtures.sh",
    "scripts/verify-artifacts.sh",
    "scripts/generate-scorecard.sh",
]

REQUIRED_DIRECTORIES = [
    "testdata/angles",
    "testdata/quaternions",
    "testdata/wmm",
    "testdata/fengshui",
    "testdata/replay",
    "third_party/noaa-wmm/LICENSES",
    "third_party/noaa-wmm/src",
    "third_party/noaa-wmm/coefficients",
    "third_party/noaa-wmm/error-model",
    "android/app",
    "android/heading-core",
    "android/heading-google",
    "android/heading-diagnostics",
    "android/benchmark-mode",
    "android/fengshui-core",
    "ios/FengShuiCompass",
    "ios/HeadingCore",
    "ios/HeadingApple",
    "ios/HeadingDiagnostics",
    "ios/BenchmarkMode",
    "ios/FengShuiCore",
    "analysis/src",
    "analysis/tests",
    "scripts",
]


@pytest.mark.parametrize("relative", REQUIRED_FILES)
def test_required_file_exists(repo_root, relative):
    assert (repo_root / relative).is_file(), f"§4.1 requires {relative}"


@pytest.mark.parametrize("relative", REQUIRED_DIRECTORIES)
def test_required_directory_exists(repo_root, relative):
    assert (repo_root / relative).is_dir(), f"§4.1 requires {relative}/"


def test_pure_cores_do_not_depend_on_platform_frameworks(repo_root):
    """§4.1: 'heading-core/HeadingCore are pure with no UI or framework singleton.'"""
    android_core = (repo_root / "android/heading-core/build.gradle.kts").read_text()
    assert "com.android" not in android_core, (
        "heading-core must not apply an Android plugin; it is a pure Kotlin/JVM module"
    )

    for swift_file in (repo_root / "ios/HeadingCore/Sources").rglob("*.swift"):
        text = swift_file.read_text()
        for framework in ("import UIKit", "import SwiftUI", "import CoreLocation", "import CoreMotion"):
            assert framework not in text, f"{swift_file.name} imports {framework}; HeadingCore is pure"


def test_benchmark_module_is_not_in_the_release_dependency_set(repo_root):
    """§23 / §29.7: benchmark and replay code is internal-build only."""
    app_build = (repo_root / "android/app/build.gradle.kts").read_text()
    assert 'debugImplementation(project(":benchmark-mode"))' in app_build
    assert 'implementation(project(":benchmark-mode"))' not in app_build
