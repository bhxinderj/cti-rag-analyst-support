"""
Centralized configuration loader.
Loads settings.yaml and provides typed access to all parameters.
"""

from pathlib import Path
from functools import lru_cache

import yaml


# PROJECT_ROOT = repo root (cti-rag-analyst-support/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "settings.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Load and cache the YAML configuration."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    return PROJECT_ROOT
