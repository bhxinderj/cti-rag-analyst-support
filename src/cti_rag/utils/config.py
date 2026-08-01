"""
Centralized configuration loader.
Loads settings.yaml and provides typed access to all parameters.
"""

import copy
import logging
import os
from pathlib import Path
from functools import lru_cache

import yaml


# PROJECT_ROOT = repo root (cti-rag-analyst-support/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "settings.yaml"


@lru_cache(maxsize=1)
def _load_base_config() -> dict:
    """Load and cache the raw YAML configuration."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_config() -> dict:
    """
    Load configuration and apply the active experiment setup, if configured.

    Supported setups:
    - A: NVD + CISA KEV + MISP
    - B: NVD + CISA KEV + MISP + CISA Advisories
    """
    config = copy.deepcopy(_load_base_config())

    # Generation-provider override for one-off runs (e.g. the hosted
    # ablation row) without touching the frozen settings.yaml default.
    llm_provider = os.environ.get("CTI_RAG_LLM_PROVIDER", "").strip().lower()
    if llm_provider:
        if llm_provider not in {"ollama", "openai", "openrouter"}:
            raise ValueError(f"Unsupported CTI_RAG_LLM_PROVIDER: {llm_provider}")
        config["llm"]["provider"] = llm_provider

    active_setup = os.environ.get("CTI_RAG_SETUP", "").strip().lower()

    if not active_setup:
        return config

    if active_setup not in {"a", "b"}:
        raise ValueError(f"Unsupported CTI_RAG_SETUP: {active_setup}")

    setup_suffix = f"setup_{active_setup}"
    include_cisa_advisories = active_setup == "b"

    config["data"]["active_setup"] = active_setup
    config["data"]["include_cisa_advisories"] = include_cisa_advisories
    config["data"]["processed_path"] = f"data/processed/{setup_suffix}/all_documents.json"
    config["retrieval"]["bm25"]["index_path"] = f"data/indexes/{setup_suffix}/bm25_index.json"
    config["retrieval"]["vector"]["collection_name"] = f"{config['retrieval']['vector']['collection_name']}_{setup_suffix}"
    config["chromadb"]["persist_directory"] = f"data/indexes/{setup_suffix}/chromadb"
    config["evaluation"]["results_dir"] = f"data/evaluation_results/{setup_suffix}"

    return config


def get_project_root() -> Path:
    return PROJECT_ROOT


def suppress_noisy_third_party_logs() -> None:
    """Silence known low-signal library warnings in local prototype runs."""
    for logger_name in (
        "chromadb.telemetry.product.posthog",
        "posthog",
    ):
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.propagate = False


def resolve_local_hf_snapshot(model_name: str) -> Path | None:
    """Return the newest local Hugging Face snapshot for the model, if present."""
    model_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    ref_path = model_dir / "refs" / "main"
    if ref_path.exists():
        snapshot_id = ref_path.read_text().strip()
        if snapshot_id:
            snapshot_path = snapshots_dir / snapshot_id
            if snapshot_path.exists():
                return snapshot_path

    snapshots = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()))
    return snapshots[-1] if snapshots else None


def require_local_hf_snapshot(model_name: str, *, artifact_label: str) -> Path:
    """Fail fast when a runtime model artifact is not available locally."""
    snapshot_path = resolve_local_hf_snapshot(model_name)
    if snapshot_path is None:
        raise FileNotFoundError(
            f"{artifact_label} '{model_name}' is not available in the local Hugging Face cache."
        )
    return snapshot_path
