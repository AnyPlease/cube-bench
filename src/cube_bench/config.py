# =====================
# file: cube_bench/config.py
# =====================
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    dataset_path: Path
    prompts_path: Path
    results_dir: Path
    max_scramble_len: int = 10
    batch_size: int = 25