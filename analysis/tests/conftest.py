"""Shared fixtures. Everything is read from the repository root, never from a copy (§37.1)."""

from __future__ import annotations

import pytest

from fscompass_analysis import artifacts


@pytest.fixture(scope="session")
def repo_root():
    return artifacts.repository_root()


@pytest.fixture(scope="session")
def profile():
    return artifacts.load_json(artifacts.PRECISION_PROFILE)


@pytest.fixture(scope="session")
def rules():
    return artifacts.load_json(artifacts.FENG_SHUI_RULES)


@pytest.fixture(scope="session")
def claims():
    return artifacts.load_json(artifacts.GRADE_REACHABILITY_CLAIMS)


@pytest.fixture(scope="session")
def example_event():
    return artifacts.load_json(artifacts.EXAMPLE_ENGINE_OUTPUT_EVENT)
