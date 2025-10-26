# ==================
# file: cube_bench/io.py
# ==================
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def save_results(file_path: Path, results: Dict[str, Any]):
    """Append or create a JSON list file with results."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    existing_data = []
    if file_path.exists():
        try:
            existing_data = json.loads(file_path.read_text(encoding="utf-8"))
            if not isinstance(existing_data, list):
                logger.warning("Results file existed but was not a list. Overwriting.")
                existing_data = []
        except json.JSONDecodeError:
            logger.warning("Could not decode existing JSON, overwriting file.")
            existing_data = []
    existing_data.append(results)
    file_path.write_text(json.dumps(existing_data, indent=4), encoding="utf-8")
    logger.info("Saved results → %s", file_path)


def load_prompts(prompts_path: Path) -> Dict[str, Any]:
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found at {prompts_path}")
    return json.loads(prompts_path.read_text(encoding="utf-8"))
