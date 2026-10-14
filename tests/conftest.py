"""Shared test setup."""
import pytest


@pytest.fixture(autouse=True)
def _no_update_check(monkeypatch):
    """Tests never ask GitHub for update notices (#120); tests/test_updates.py turns it back on with a fake opener."""
    monkeypatch.setenv("LOCALLLM_NO_UPDATE_CHECK", "1")
