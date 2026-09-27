import pytest

from nodeforge.core.config import NodeForgeConfig
from nodeforge.core.errors import PipelineError


def test_defaults_valid():
    cfg = NodeForgeConfig()
    assert cfg.seed == 42 and cfg.hidden == 32


def test_invalid_dropout():
    with pytest.raises(PipelineError):
        NodeForgeConfig(dropout=1.5)


def test_invalid_tau():
    with pytest.raises(PipelineError):
        NodeForgeConfig(tau=0.0)


def test_invalid_knn_k():
    with pytest.raises(PipelineError):
        NodeForgeConfig(knn_k=1)


def test_env_override(monkeypatch):
    monkeypatch.setenv("ENV_NODEFORGE_SEED", "7")
    monkeypatch.setenv("ENV_NODEFORGE_HIDDEN", "16")
    cfg = NodeForgeConfig.from_env()
    assert cfg.seed == 7 and cfg.hidden == 16


def test_env_bad_key(monkeypatch):
    monkeypatch.setenv("ENV_NODEFORGE_NOPE", "1")
    with pytest.raises(PipelineError):
        NodeForgeConfig.from_env()


def test_env_bad_value(monkeypatch):
    monkeypatch.setenv("ENV_NODEFORGE_SEED", "abc")
    with pytest.raises(PipelineError):
        NodeForgeConfig.from_env()
