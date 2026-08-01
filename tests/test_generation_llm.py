"""
Unit tests for the generation-LLM provider switch (build_generation_llm).

The local Ollama path must stay argument-identical to the evaluated
setup — the _final_v1 series was produced with exactly this
construction, and the thesis describes it. The hosted branches exist
only for the hosted-generation ablation row.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from src.cti_rag.rag.chain import build_generation_llm


_BASE_CONFIG = {
    "model_name": "llama3.1:8b-instruct-q5_K_M",
    "base_url": "http://localhost:11434",
    "temperature": 0.1,
    "top_p": 0.9,
    "max_tokens": 2048,
    "request_timeout": 120,
}


def test_default_provider_builds_ollama_with_evaluated_arguments():
    captured = {}

    class _Dummy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch("src.cti_rag.rag.chain.ChatOllama", _Dummy):
        label, llm = build_generation_llm(dict(_BASE_CONFIG))

    assert label == "ollama:llama3.1:8b-instruct-q5_K_M"
    assert isinstance(llm, _Dummy)
    # Argument-identical to the _final_v1 construction: same keys, same
    # values, no extra kwargs like timeout sneaking into the local path.
    assert captured == {
        "model": "llama3.1:8b-instruct-q5_K_M",
        "base_url": "http://localhost:11434",
        "temperature": 0.1,
        "top_p": 0.9,
        "num_predict": 2048,
    }


def test_missing_provider_key_defaults_to_ollama():
    with patch("src.cti_rag.rag.chain.ChatOllama") as dummy:
        label, _ = build_generation_llm(dict(_BASE_CONFIG))
    assert label.startswith("ollama:")
    assert dummy.called


def test_openrouter_provider_builds_chatopenai_with_generation_params():
    config = dict(_BASE_CONFIG)
    config["provider"] = "openrouter"
    config["model_name_openrouter"] = "anthropic/claude-haiku-4.5"

    captured = {}

    class _Dummy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), patch(
        "langchain_openai.ChatOpenAI", _Dummy
    ):
        label, llm = build_generation_llm(config)

    assert label == "openrouter:anthropic/claude-haiku-4.5"
    assert isinstance(llm, _Dummy)
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["model"] == "anthropic/claude-haiku-4.5"
    # Generation parameters carry over (not the judge's temperature=0.0).
    assert captured["temperature"] == 0.1
    assert captured["top_p"] == 0.9
    assert captured["max_tokens"] == 2048


def test_openrouter_provider_without_key_raises():
    config = dict(_BASE_CONFIG)
    config["provider"] = "openrouter"

    env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        try:
            build_generation_llm(config)
        except RuntimeError as exc:
            assert "OPENROUTER_API_KEY" in str(exc)
        else:
            raise AssertionError("expected RuntimeError without API key")
