"""Trivial smoke test — confirms the package imports and pytest is wired up."""

import cbb_sim


def test_package_imports():
    assert cbb_sim.__version__


def test_arithmetic_sanity():
    assert 1 + 1 == 2
