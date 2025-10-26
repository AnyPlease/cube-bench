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

from pathlib import Path

from ..core import BaseTest
from ..io import save_results
from cube_bench.sim.cube_simulator import VirtualCube

logger = logging.getLogger(__name__)


class ReconstructionTest(BaseTest):
    """
    Face color 3x3 reconstruction accuracy (element-wise & overall).

    Expected model format (case/spacing tolerant):
    ANSWER:
    Row 1: [Color1, Color2, Color3]
    Row 2: [Color4, Color5, Color6]
    Row 3: [Color7, Color8, Color9]

    Production notes:
    - Uses deterministic scrambles per-sample.
    - Logs progress every `LOG_EVERY` samples (INFO); deep per-sample details at DEBUG when `self.verbose` is True.
    - Parsing tolerates code fences, extra prose, and JSON-like lists.
    """

    LOG_EVERY = 25
    MAX_NEW_TOKENS = 2**16
    DEFAULT_TEMPERATURE = 0.1

    # Relaxed matcher (case-insensitive, allows newlines/extra spaces)
    GRID_RE = re.compile(
        r"answer:\s*"
        r"row\s*1:\s*\[\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*\]\s*"
        r"row\s*2:\s*\[\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*\]\s*"
        r"row\s*3:\s*\[\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*\]",
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Fallback: tolerate a JSON-like 3x3 list anywhere in the text (captures full words, not just initials)
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
        # Promote module logger to DEBUG if test is run with verbose=True
        if getattr(self, "verbose", True):
            logger.setLevel(logging.DEBUG)

    # ---------- Normalization & scoring ----------

    def _norm_color(self, s: str) -> Optional[str]:
        t = (s or "").strip().lower()
        # exact color words/initials
        if t in self.COLOR_MAP:
            return self.COLOR_MAP[t]
        if t and t[0] in self.COLOR_MAP:  # e.g., "white" → 'w'
            return self.COLOR_MAP[t[0]]
        # face tokens (full word or initial), e.g., "U", "up", "front"
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
        # Remove fenced blocks and surrounding noise
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
        # Prefer explicit attribute if provided (e.g., via config or CLI)
        if isinstance(getattr(self, "n_moves", None), int) and self.n_moves > 0:
            return int(self.n_moves)
        return {"easy": 4, "medium": 8, "hard": 12, "all": 8}.get(difficulty, 8)

    # ---------- Runner ----------

    def run(self, num_samples: int) -> Dict[str, Any]:
        """
        Execute Reconstruction test.

        Args:
            num_samples: number of independent scrambles/images to evaluate.

        Returns:
            Dict with aggregated metrics and run metadata.
        """
        self._enable_verbose_logging_if_requested()
        sys_prompt, user_prompt = self._prompts()

        elem_acc: List[float] = []
        full_acc: List[float] = []

        total = max(0, int(num_samples))
        logger.info(
            f"Starting ReconstructionTest: samples={total}, n_moves={self.n_moves}, model={self.assistant.get_name()}"
        )

        parse_total = 0

        for idx in tqdm(range(1, total + 1), desc="Reconstruction Test"):
            cube = VirtualCube()

            # Deterministic scramble per sample index
            scramble: Any = None
            try:
                scramble = cube.scramble(random_seed=idx, n_moves=self.n_moves)
            except TypeError:
                # Fallback if signature differs
                random.seed(idx)
                scramble = cube.scramble(n_moves=self.n_moves)

            gt = cube.front_face()  # 3x3 ground-truth colors

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
                        f"Sample {idx}\ngt={gt}\nscramble={scramble.__str__()}"
                    )
                
                print(resp)
            except Exception as e:
                logger.exception(
                    "assistant.generate failed on sample %d (n_moves=%d, scramble=%s): %s",
                    idx, self.n_moves, scramble, e
                )
                resp = ""

            ew, ov, parse_total = self._eval(resp, gt, parse_total)
            elem_acc.append(ew)
            full_acc.append(ov)

            if (idx % self.LOG_EVERY == 0) or (idx == total):
                running_elem = (sum(elem_acc) / len(elem_acc)) if elem_acc else 0.0
                running_overall = (sum(full_acc) / len(full_acc)) if full_acc else 0.0
                logger.info(
                    "Reconstruction %d/%d — running avg (elem: %.3f, overall: %.3f)",
                    idx, total, running_elem, running_overall
                )

        avg_ew = (sum(elem_acc) / len(elem_acc)) if elem_acc else 0.0
        avg_ov = (sum(full_acc) / len(full_acc)) if full_acc else 0.0

        logger.info("Reconstruction — element-wise: %.3f, overall: %.3f", avg_ew, avg_ov)

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
            "correct_parse": parse_total
        }

        try:
            save_results(out_path, result)
            logger.info("Saved results → %s", out_path)
        except Exception:
            logger.exception("Failed to save results to %s", out_path)

        return result
