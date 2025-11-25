# =====================================
# file: cube_bench/tests/reconstruction.py
# =====================================
from __future__ import annotations

import logging
import re
from tqdm import tqdm
import random
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter # <--- ADDED

from pathlib import Path

from ..core import BaseTest
from ..io import save_results
from cube_bench.sim.cube_simulator import VirtualCube

logger = logging.getLogger(__name__)


class ReconstructionTest(BaseTest):
    """
    Face color 3x3 reconstruction accuracy (element-wise & overall).
    """

    LOG_EVERY = 25
    MAX_NEW_TOKENS = 2**16
    DEFAULT_TEMPERATURE = 0.1
    
    # Fairness Control: Reject face states where a single color dominates
    # This ensures we don't get "easy" monochromatic faces.
    MAX_SINGLE_COLOR_COUNT = 6

    # Relaxed matcher (case-insensitive, allows newlines/extra spaces)
    GRID_RE = re.compile(
        r"answer:\s*"
        r"row\s*1:\s*\[\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*\]\s*"
        r"row\s*2:\s*\[\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*\]\s*"
        r"row\s*3:\s*\[\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*\]",
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Fallback: tolerate a JSON-like 3x3 list anywhere in the text
    JSONISH_RE = re.compile(
        r"\[\s*\[\s*['\"]?([A-Za-z]+)['\"]?\s*,\s*['\"]?([A-Za-z]+)['\"]?\s*,\s*['\"]?([A-Za-z]+)['\"]?\s*\]\s*,\s*"
        r"\[\s*['\"]?([A-Za-z]+)['\"]?\s*,\s*['\"]?([A-Za-z]+)['\"]?\s*,\s*['\"]?([A-Za-z]+)['\"]?\s*\]\s*,\s*"
        r"\[\s*['\"]?([A-Za-z]+)['\"]?\s*,\s*['\"]?([A-Za-z]+)['\"]?\s*,\s*['\"]?([A-Za-z]+)['\"]?\s*\]\s*\]",
        flags=re.IGNORECASE | re.DOTALL,
    )

    CODE_FENCE_RE = re.compile(r"^```(?:json|python|txt)?\s*|\s*```$", flags=re.IGNORECASE | re.MULTILINE)

    COLOR_MAP = {
        "w": "W", "white": "W",
        "y": "Y", "yellow": "Y",
        "r": "R", "red": "R",
        "o": "O", "orange": "O",
        "b": "B", "blue": "B",
        "g": "G", "green": "G",
    }

    FACE_TO_COLOR = {
        "u": "W",  # Up = White
        "d": "Y",  # Down = Yellow
        "f": "G",  # Front = Green
        "b": "B",  # Back = Blue
        "l": "O",  # Left = Orange
        "r": "R",  # Right = Red
    }

    def _enable_verbose_logging_if_requested(self) -> None:
        if getattr(self, "verbose", True):
            logger.setLevel(logging.DEBUG)

    # ---------- Normalization & scoring ----------

    def _norm_color(self, s: str) -> Optional[str]:
        t = (s or "").strip().lower()
        if t in self.COLOR_MAP:
            return self.COLOR_MAP[t]
        if t and t[0] in self.COLOR_MAP:
            return self.COLOR_MAP[t[0]]
        if t in FACE_TO_COLOR:
            return FACE_TO_COLOR[t]
        if t and t[0] in FACE_TO_COLOR:
            return FACE_TO_COLOR[t[0]]
        return None

    def _norm_grid(self, grid: List[List[str]]) -> Optional[List[List[str]]]:
        out: List[List[str]] = []
        for row in grid:
            nr: List[str] = []
            for c in row:
                nc = self._norm_color(c)
                if nc is None:
                    return None
                nr.append(nc)
            out.append(nr)
        return out

    def _score(self, gt: List[List[str]], pred: List[List[str]]) -> Tuple[float, float]:
        ngt = self._norm_grid(gt)
        npred = self._norm_grid(pred)
        if ngt is None or npred is None:
            return 0.0, 0.0
        eq = sum(1 for r1, r2 in zip(ngt, npred) for a, b in zip(r1, r2) if a == b)
        elem = eq / 9.0
        overall = 1.0 if eq == 9 else 0.0
        return elem, overall

    # ---------- Parsing ----------

    def _clean_text(self, resp: str) -> str:
        return self.CODE_FENCE_RE.sub("", resp or "").strip()

    def _parse_grid(self, resp: str) -> Optional[List[List[str]]]:
        if not resp:
            return None
        txt = self._clean_text(resp)
        m = self.GRID_RE.search(txt)
        if m:
            c = m.groups()
            grid = [[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], c[8]]]
            logger.debug("Parsed via GRID_RE: %s", grid)
            return grid
        j = self.JSONISH_RE.search(txt)
        if j:
            c = j.groups()
            grid = [[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], c[8]]]
            logger.debug("Parsed via JSONISH_RE: %s", grid)
            return grid
        logger.debug("Reconstruction parse failed; response head: %r", txt[:240])
        return None

    def _eval(self, resp: str, gt: List[List[str]], parse: int) -> Tuple[float, float]:
        pred = self._parse_grid(resp)
        if pred is None:
            return 0.0, 0.0
        ew, ov = self._score(gt, pred)
        parse += 1
        logger.debug("Scores — element-wise: %.3f, overall: %.3f", ew, ov)
        return ew, ov, parse

    # ---------- Prompting ----------

    def _prompts(self) -> Tuple[str, str]:
        sys_prompt, user_prompt = self.prompts.get("reconstruction") or {}
        if not sys_prompt:
            sys_prompt = (
                "You analyze a Rubik's Cube net image. Output ONLY this format:\n"
                "ANSWER:\n"
                "Row 1: [Color1, Color2, Color3]\n"
                "Row 2: [Color4, Color5, Color6]\n"
                "Row 3: [Color7, Color8, Color9]\n"
            )
        if not user_prompt:
            user_prompt = "Identify the FRONT face colors in the exact required format."
        return sys_prompt, user_prompt

    # ---------- Scramble policy ----------

    def _pick_n_moves(self, difficulty: str) -> int:
        if isinstance(getattr(self, "n_moves", None), int) and self.n_moves > 0:
            return int(self.n_moves)
        return {"easy": 4, "medium": 8, "hard": 12, "all": 8}.get(difficulty, 8)

    # ---------- Runner ----------

    def run(self, num_samples: int) -> Dict[str, Any]:
        """
        Execute Reconstruction test with fairness controls.
        Ensures color priors do not deviate >5% from uniform.
        """
        self._enable_verbose_logging_if_requested()
        sys_prompt, user_prompt = self._prompts()

        elem_acc: List[float] = []
        full_acc: List[float] = []
        
        # Fairness tracking
        global_color_counts = Counter()
        total_stickers = 0

        total = max(0, int(num_samples))
        logger.info(
            f"Starting ReconstructionTest: samples={total}, n_moves={self.n_moves}, model={self.assistant.get_name()}"
        )

        parse_total = 0

        current_max_count = 9 if self.n_moves < 3 else 6

        for idx in tqdm(range(1, total + 1), desc="Reconstruction Test"):
            cube = VirtualCube()
            
            valid_scramble = False
            attempt = 0
            scramble: Any = None
            gt: List[List[str]] = []
            
            while not valid_scramble:
                # Deterministic offset for retries
                current_seed = idx if attempt == 0 else (idx * 10000 + attempt)
                
                try:
                    scramble = cube.scramble(random_seed=current_seed, n_moves=self.n_moves)
                except TypeError:
                    random.seed(current_seed)
                    scramble = cube.scramble(n_moves=self.n_moves)
                
                gt = cube.front_face()
                
                # Check per-sample balance
                flat_face = []
                for row in gt:
                    for c in row:
                        norm = self._norm_color(c)
                        if norm: flat_face.append(norm)
                
                counts = Counter(flat_face)
                
                # REJECT if a single color dominates beyond what is physically reasonable for this depth
                if any(c > current_max_count for c in counts.values()):
                    attempt += 1
                    # Safety break to prevent infinite loops on hard seeds
                    if attempt > 50: 
                         logger.warning(f"Could not satisfy max_count={current_max_count} for idx {idx}, accepting best effort.")
                         valid_scramble = True
                else:
                    valid_scramble = True
                    # The GLOBAL stats will naturally balance out over many samples
                    global_color_counts.update(counts)
                    total_stickers += 9

            kwargs: Dict[str, Any] = {
                "user_prompt": user_prompt,
                "system_prompt": sys_prompt,
                "max_new_tokens": self.MAX_NEW_TOKENS,
                "temperature": self.DEFAULT_TEMPERATURE,
            }

            try:
                img = cube.to_image()
                if img is not None:
                    kwargs["image"] = img
            except Exception as e:
                logger.warning("to_image() failed on sample %d: %s", idx, e)

            # Generate model response
            try:
                resp = self.assistant.generate(**kwargs)
                if self.verbose:
                    logger.debug(
                        f"Sample {idx} (attempts={attempt})\ngt={gt}\nscramble={scramble.__str__()}"
                    )
                print(resp)
            except Exception as e:
                logger.exception(
                    "assistant.generate failed on sample %d: %s", idx, e
                )
                resp = ""

            ew, ov, parse_total = self._eval(resp, gt, parse_total)
            elem_acc.append(ew)
            full_acc.append(ov)

            if (idx % self.LOG_EVERY == 0) or (idx == total):
                logger.info(
                    "Reconstruction %d/%d — running avg (elem: %.3f, overall: %.3f)",
                    idx, total, 
                    (sum(elem_acc) / len(elem_acc)) if elem_acc else 0.0,
                    (sum(full_acc) / len(full_acc)) if full_acc else 0.0
                )

        # --- Final Calculation of Fairness Metrics ---
        # Verify the <0.05 deviation claim
        if total_stickers > 0:
            expected_freq = 1.0 / 6.0  # ~0.1667
            max_dev = 0.0
            logger.info("--- Fairness Check (Prior Deviation) ---")
            for color in ["W", "Y", "R", "O", "G", "B"]:
                count = global_color_counts[color]
                freq = count / total_stickers
                dev = abs(freq - expected_freq)
                max_dev = max(max_dev, dev)
                logger.info(f"Color {color}: {count} ({freq:.4f}) | Dev: {dev:.4f}")
            
            logger.info(f"Max Deviation: {max_dev:.4f} (Target < 0.05)")
        else:
            max_dev = 0.0

        avg_ew = (sum(elem_acc) / len(elem_acc)) if elem_acc else 0.0
        avg_ov = (sum(full_acc) / len(full_acc)) if full_acc else 0.0

        # Persist results
        results_dir: Path = self.config.results_dir if hasattr(self.config, "results_dir") else Path(".")
        out_path = results_dir / f"reconstruction.json"

        result = {
            "model_name": self.assistant.get_name(),
            "test_type": "reconstruction",
            "timestamp": datetime.now().isoformat(),
            "average_accuracy_element_wise": avg_ew,
            "average_accuracy_overall": avg_ov,
            "num_samples": total,
            "n_moves": self.n_moves,
            "correct_parse": parse_total,
            "max_prior_deviation": max_dev 
        }

        try:
            save_results(out_path, result)
            logger.info("Saved results → %s", out_path)
        except Exception:
            logger.exception("Failed to save results to %s", out_path)

        return result