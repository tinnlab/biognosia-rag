"""Tests for MCP_LLM_MODE — sampling (default) vs standalone.

These run with MCP_TEST_MODE=1 (see conftest.py) and import neither torch nor
any LLM client library.
"""
import json

import pytest

from src_rag_web import mcp_route, mcp_server


# ── mode parsing ─────────────────────────────────────────────────────────────

def test_llm_mode_defaults_to_sampling(monkeypatch):
    monkeypatch.delenv("MCP_LLM_MODE", raising=False)
    assert mcp_server._llm_mode() == "sampling"
    assert mcp_route._standalone_llm() is False


def test_empty_llm_mode_is_sampling(monkeypatch):
    monkeypatch.setenv("MCP_LLM_MODE", "")
    assert mcp_server._llm_mode() == "sampling"
    assert mcp_route._standalone_llm() is False


@pytest.mark.parametrize("value", ["standalone", "  STANDALONE  ", "Standalone"])
def test_standalone_is_recognised_case_and_space_insensitively(monkeypatch, value):
    monkeypatch.setenv("MCP_LLM_MODE", value)
    assert mcp_server._llm_mode() == "standalone"
    assert mcp_route._standalone_llm() is True


def test_unknown_llm_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("MCP_LLM_MODE", "direct")
    with pytest.raises(RuntimeError, match="MCP_LLM_MODE"):
        mcp_server._llm_mode()


# ── startup validation is reachable from the server module ───────────────────

def test_server_validate_rejects_missing_model():
    with pytest.raises(RuntimeError, match="BIOGNOSIA_LLM_MODEL"):
        mcp_server._validate_llm_config({"provider": "openai", "base_url": "http://x/v1"})


def test_server_require_sdk_reports_missing_package(monkeypatch):
    """A provider whose client library is absent must fail with an install hint."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("no module named anthropic")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="pip install anthropic"):
        mcp_server._require_llm_sdk("anthropic")


def test_server_require_sdk_ignores_providers_with_no_sdk():
    mcp_server._require_llm_sdk("ollama")  # aiohttp-based, nothing to import


# ── capability advertisement ─────────────────────────────────────────────────
#
# _handle_initialize is called directly rather than through TestClient: with
# MCP_TEST_MODE=0 the real lifespan would run and import torch.

def _initialize_capabilities(monkeypatch, *, test_mode: bool):
    monkeypatch.setattr(mcp_route, "_test_mode", lambda: test_mode)
    response = mcp_route._handle_initialize({"jsonrpc": "2.0", "id": 1})
    return json.loads(response.body)["result"]["capabilities"]


def test_initialize_advertises_sampling_by_default(monkeypatch):
    monkeypatch.delenv("MCP_LLM_MODE", raising=False)
    caps = _initialize_capabilities(monkeypatch, test_mode=False)
    assert "tools" in caps
    assert "sampling" in caps


def test_initialize_omits_sampling_in_standalone_mode(monkeypatch):
    """Standalone never sends sampling/createMessage, so it must not claim to."""
    monkeypatch.setenv("MCP_LLM_MODE", "standalone")
    caps = _initialize_capabilities(monkeypatch, test_mode=False)
    assert "tools" in caps
    assert "sampling" not in caps


def test_test_mode_still_advertises_sampling(monkeypatch):
    """The canned test pipeline does a real round-trip, so the capability holds."""
    monkeypatch.setenv("MCP_LLM_MODE", "standalone")
    caps = _initialize_capabilities(monkeypatch, test_mode=True)
    assert "sampling" in caps
