"""
Centralized configuration loader.
Loads settings.yaml and provides typed access to all parameters.
"""

import copy
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
    config["retrieval"]["bm25"]["index_path"] = f"data/indexes/{setup_suffix}/bm25_index.pkl"
    config["retrieval"]["vector"]["collection_name"] = f"{config['retrieval']['vector']['collection_name']}_{setup_suffix}"
    config["chromadb"]["persist_directory"] = f"data/indexes/{setup_suffix}/chromadb"
    config["evaluation"]["results_dir"] = f"data/evaluation_results/{setup_suffix}"

    return config


def get_project_root() -> Path:
    return PROJECT_ROOT
