"""Shared test setup."""
import pytest


@pytest.fixture(autouse=True)
def _english_messages(monkeypatch):
    """Tests compare printed text in English whatever language the test PC runs (#108); tests/test_i18n.py changes it."""
    monkeypatch.setenv("LOCALLLM_LANG", "en")
