# ========================
# file: cube_bench/core/base.py
# ========================
from __future__ import annotations
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import logging

from ..config import Config
from ..io import load_prompts
from ..prompts.prompt_factory import PromptFactory

logger = logging.getLogger(__name__)


class BaseTest(ABC):
    """Abstract base for all tests."""
    def __init__(self, assistant, config: Config, n_moves: int, verbose: bool):
        self.assistant = assistant
        self.config = config
        self.n_moves = n_moves
        self.verbose = verbose
        # Lazily read prompts.json once
        try:
            self.prompts = PromptFactory()
        except Exception:
            self.prompts = {}

    def load_dataset(self, difficulty: str = "easy", path: Optional[Path] = None) -> List[Dict[str, Any]]:
        data_path = path or self.config.dataset_path
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if difficulty == "all":
            return data
        return [item for item in data if item.get("difficulty") == difficulty]

    @abstractmethod
    def run(self, num_samples: int):
        ...