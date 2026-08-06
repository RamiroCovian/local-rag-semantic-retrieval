"""Fixtures compartidos de testing."""

from pathlib import Path

import pytest


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"
