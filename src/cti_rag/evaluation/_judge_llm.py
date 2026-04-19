"""
Shared judge-LLM factory for evaluation modules.

Both RAGAS and the Structured Rubric LLM-as-Judge evaluator need to
instantiate the same provider-agnostic "judge" model configuration. This
module centralizes the selection logic so a single config switch
(``evaluation.ragas.eval_llm_provider``) controls both.

Providers:
  - ``ollama``      — local Ollama (default)
  - ``openai``      — OpenAI directly via ``OPENAI_API_KEY``
  - ``openrouter``  — OpenRouter-hosted OpenAI models via ``OPENROUTER_API_KEY``

The factory returns the raw LangChain chat model (not a RAGAS wrapper)
so each caller can wrap it as needed — RAGAS wraps with
``LangchainLLMWrapper``, the rubric evaluator invokes it directly.
"""

from __future__ import annotations

import logging
import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def build_judge_llm(
    provider: str,
    eval_config: dict,
    ollama_base_url: str,
    timeout: int = 120,
) -> tuple[str, object]:
    """Instantiate the configured judge LLM.

    Parameters
    ----------
    provider:
        One of ``"ollama"``, ``"openai"``, ``"openrouter"``. Case-insensitive.
        Unknown values fall back to ``"ollama"``.
    eval_config:
        The ``evaluation.ragas`` sub-tree of settings.yaml. Expected keys:
        ``eval_llm``, ``eval_llm_openai``, ``eval_llm_openrouter``.
    ollama_base_url:
        Base URL for local Ollama (used only for the ``ollama`` provider).
    timeout:
        Per-call timeout in seconds. Ollama typically needs a much longer
        timeout than hosted APIs; callers should pass ~600 for RAGAS and
        keep the default ~120 for single-shot rubric judgments.

    Returns
    -------
    ``(model_label, chat_model)``
        ``model_label`` is a human-readable identifier such as
        ``"openrouter:openai/gpt-4o-mini"`` for logging and artifact
        metadata. ``chat_model`` is the raw LangChain chat model; RAGAS
        callers should wrap it in ``LangchainLLMWrapper`` themselves.
    """
    provider_norm = (provider or "ollama").lower()

    if provider_norm == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "eval_llm_provider=openai requires the OPENAI_API_KEY environment variable."
            )
        model_name = eval_config.get("eval_llm_openai", "gpt-4o-mini")
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.0,
            timeout=timeout,
            max_retries=4,
        )
        return f"openai:{model_name}", llm

    if provider_norm == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "eval_llm_provider=openrouter requires the OPENROUTER_API_KEY environment variable."
            )
        model_name = eval_config.get("eval_llm_openrouter", "openai/gpt-4o-mini")
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
            timeout=timeout,
            max_retries=4,
        )
        return f"openrouter:{model_name}", llm

    # Default: local Ollama
    model_name = eval_config.get("eval_llm", "qwen2.5:7b-instruct")
    llm = ChatOllama(
        model=model_name,
        base_url=ollama_base_url,
        temperature=0.0,
        timeout=timeout,
    )
    return f"ollama:{model_name}", llm
