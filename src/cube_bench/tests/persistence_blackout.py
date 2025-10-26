# =========================================
# file: cube_bench/tests/persistence_blackout.py
# =========================================
from __future__ import annotations
import logging
import random
import copy
import re
from tqdm import tqdm
import time
from pathlib import Path
from datetime import datetime
from functools import lru_cache
from typing import Dict, Tuple, Optional, List

import kociemba

from ..core import BaseTest
from ..io import save_results
from cube_bench.sim.cube_simulator import VirtualCube
from cube_bench.oracle.optimal_oracle import OptimalOracle, StickerCountHeuristic
from cube_bench.oracle.pattern_db import exact_depth_leq10, ensure_frontier

logger = logging.getLogger(__name__)


class PersistenceBlackoutTest(BaseTest):
    """
    World-model persistence under missing observation.

    blackout_mode:
      - "none"  : no blackout (normal SBS with image+text)
      - "image" : hide the image at blackout steps, keep text
      - "all"   : hide both image and current text; reconstruct from initial text + move history
    """
    OPTIONS_RE = re.compile(r"<ANSWER>\s*([ABCD])\s*</ANSWER>", re.IGNORECASE)

    def __init__(
        self,
        assistant,
        config,
        n_moves: int = 2,
        blackout_mode: str = "all",
        blackout_interval: int = 3,
        blackout_phase: int = 1,
        max_steps: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__(assistant, config)
        self.test_type = "persistence_blackout"
        self.n_moves = int(n_moves)
        self.blackout_mode = blackout_mode.lower().strip()
        if self.blackout_mode not in {"none", "image", "all"}:
            raise ValueError(f"Invalid blackout_mode: {blackout_mode!r}. Use one of: none | image | all.")
        self.blackout_interval = max(1, int(blackout_interval))
        self.blackout_phase = max(0, int(blackout_phase))
        self.max_steps = max_steps if (max_steps is None) else int(max_steps)
        self.verbose = verbose

        # Build/load solved-side frontier once (depth 5 → exact up to 10)
        # cache_base = getattr(config, "cache_dir", None) or (self.config.results_dir / ".cache")
        self._frontier_path = "/home/dana0001/wj84_scratch/dana0001/rubiks-cube-thesis/scripts/cube_bench/oracle/frontier5.pkl.gz"
        self._solved_frontier = ensure_frontier(self._frontier_path, depth_limit=5, progress=self.verbose)


        self.oracle = OptimalOracle(
            vc_factory=VirtualCube,
            heuristic=StickerCountHeuristic(),
            time_cap_ms=2000,
            use_kociemba_fallback=True,
            enable_axis_prune=True,
            enable_inverse_prune=True,
        )

    # ---------- distance helpers (EXACT <=10 with fallback) ----------
    @lru_cache(maxsize=200_000)
    def _exact10(self, s54: str) -> Optional[int]:
        # exact HTM depth if <=10 else None
        return exact_depth_leq10(s54, self._solved_frontier, limit_each_side=5, progress=False)

    @lru_cache(maxsize=100_000)
    def _kociemba_len(self, s54: str) -> int:
        seq = kociemba.solve(s54)
        return len(seq.split()) if seq else 0

    def _distance(self, vc: VirtualCube) -> int:
        """HTM depth using exact meet-in-the-middle if possible; otherwise Kociemba length."""
        s54 = vc.to_kociemba()
        d = self._exact10(s54)
        if d is not None:
            return d
        # fallback for >10-depth positions
        return self._kociemba_len(s54)

    def _progress(self, vc: VirtualCube, move: str) -> Tuple[bool, int, int]:
        d0 = self._distance(vc)
        c = vc.clone()
        c.apply(move)
        if c.is_solved():
            return True, d0, 0
        d1 = self._distance(c)
        return (d1 < d0), d0, d1

    def _optimal_first(self, vc: VirtualCube) -> set[str]:
        """Moves that reduce exact HTM depth (<=10), with fallback for deeper states."""
        if vc.is_solved():
            return set()
        d0 = self._distance(vc)
        good = set()
        for m in VirtualCube.AVAILABLE_MOVES:
            c = vc.clone()
            c.apply(m)
            if c.is_solved() or self._distance(c) == d0 - 1:
                good.add(m)
        return good
    
    def _oracle_optimal_first_moves(self, vc: VirtualCube) -> set[str]:
        """Oracle-based decrease (fast)."""
        if vc.is_solved():
            return set()
        good = set()
        for m in VirtualCube.AVAILABLE_MOVES:
            c = vc.clone()
            c.apply(m)
            if c.is_solved() or self.oracle.classify_neighbor(vc, m) == "DECREASE":
                good.add(m)
        return good

    # ---------- blackout helpers ----------
    def _blackout_steps(self, n_steps: int) -> set[int]:
        if self.blackout_mode == "none":
            return set()
        return {t for t in range(n_steps) if (t % self.blackout_interval) == self.blackout_phase}

    # ---------- parsing ----------
    def _parse_answer(self, text: str) -> Optional[str]:
        m = self.OPTIONS_RE.search(text or "")
        return m.group(1).upper() if m else None

    # ---------- prompts ----------
    def _prompt_obs(self, state_text: str, options: Dict[str, str]):
        sys_prompt = (
            "You are an expert Rubik's Cube solver. The textual cube state is the single source of truth.\n"
            "Output EXACTLY one of A, B, C, or D wrapped in <ANSWER> tags (no extra text)."
        )
        user = (
            f"Textual Cube State (ground truth):\n{state_text}\n\n"
            f"Candidate Moves:\nA: {options['A']}\nB: {options['B']}\nC: {options['C']}\nD: {options['D']}\n\n"
            "Reply with exactly one line:\n<ANSWER> A/B/C/D </ANSWER>"
        )
        return sys_prompt, user

    def _prompt_blackout(self, initial_text: str, history: List[str], options: Dict[str, str]):
        sys_prompt = (
            "You are an expert Rubik's Cube solver. There is NO current observation this step. Reconstruct "
            "the current state by applying the move history to the INITIAL state. Output EXACTLY one of A–D in <ANSWER> tags."
        )
        hist = ", ".join(history) if history else "(none)"
        user = (
            f"INITIAL textual cube state (step 0):\n{initial_text}\n\n"
            f"Move history so far (applied in order):\n{hist}\n\n"
            f"Candidate Moves:\nA: {options['A']}\nB: {options['B']}\nC: {options['C']}\nD: {options['D']}\n\n"
            "Reply exactly: <ANSWER> A/B/C/D </ANSWER>"
        )
        return sys_prompt, user

    # ---------- misc helpers ----------
    @staticmethod
    def _state_text(vc: VirtualCube) -> str:
        # Using the underlying string form the rest of your suite uses
        return vc._cube.__str__()  # noqa: SLF001 (private access consistent with other tests)

    @staticmethod
    def _det_seed(idx: int, t: int, correct_move: str) -> int:
        # Stable, cross-process deterministic seed (avoid Python's salted hash)
        return (idx * 10007) ^ (t * 1009) ^ sum(ord(c) for c in correct_move)

    # ---------- test entry ----------
    def run(self, num_samples: int):
        solved = 0
        total_depths = 0
        total_calls = 0
        parse_viol = 0

        blackout_total = 0
        blackout_stc = 0  # STC = chose an optimal-first move during blackout steps
        blackout_prog = 0  # made progress during blackout steps

        Hs = [1, 2, 3, 4]
        rec_success = {H: 0 for H in Hs}
        rec_denom = 0
        latencies: List[float] = []

        for idx in tqdm(range(num_samples), desc=f"Persistence-blackout ({self.n_moves} moves)"):
            cube = VirtualCube()
            scramble = cube.scramble(random_seed=idx, n_moves=self.n_moves)
            scramble_copy = copy.deepcopy(scramble)
            teacher_path = str(scramble_copy.reverse()).split()
            
            if not teacher_path:
                continue

            n_steps = len(teacher_path)
            step_cap = self.max_steps if self.max_steps is not None else n_steps
            blk = self._blackout_steps(n_steps)
            initial_text = self._state_text(cube)
            history: List[str] = []

            print(f"Blackout [{self.blackout_mode}] at steps: {blk}")
            print(f"Scramble path: {scramble}")
            print(f"Teacher path: {teacher_path}")

            first_off_idx: Optional[int] = None
            recovered_in_k: Optional[int] = None

            for t, teacher_move in enumerate(teacher_path[:step_cap]):
                if cube.is_solved():
                    break

                rng = random.Random(self._det_seed(idx, t, teacher_move))
                options, gold_letter = self._gen_mcq(teacher_move, rng)
                state_text = self._state_text(cube)

                if (t in blk) and (self.blackout_mode == "all"):
                    sys_p, user_p = self._prompt_blackout(initial_text, history, options)
                    img = None
                    is_blk = True
                elif (t in blk) and (self.blackout_mode == "image"):
                    sys_p, user_p = self._prompt_obs(state_text, options)
                    img = None
                    is_blk = True
                else:
                    sys_p, user_p = self._prompt_obs(state_text, options)
                    img = cube.to_image()
                    is_blk = False

                t0 = time.time()
                resp = self.assistant.generate(
                    user_prompt=user_p,
                    system_prompt=sys_p,
                    image=img,
                    temperature=0.0,
                    max_new_tokens=2**14,
                )
                latencies.append(time.time() - t0)
                total_calls += 1

                pred_letter = self._parse_answer(resp)
                if pred_letter is None:
                    parse_viol += 1

                predicted_move = options.get(pred_letter) if pred_letter else None
                if not predicted_move:
                    # On parse failure pick a deterministic fallback (A)
                    print("Model failed")
                    predicted_move = options["A"]

                good = self._oracle_optimal_first_moves(cube.clone())
                made_progress, d0, d1 = self._progress(cube, predicted_move)
                is_optimal = (predicted_move in good) or (d1 == 0)

                print(
                    f"Good moves right now: {good}\n"
                    f"Teacher move: {teacher_move}\n"
                    f"Predicted move: {predicted_move}\n"
                    f"{made_progress=} {d0=} {d1=}\n"
                    f"--------------------------------------------------"
                )

                if is_blk:
                    blackout_total += 1
                    blackout_stc += int(is_optimal)
                    blackout_prog += int(made_progress)

                if first_off_idx is None and not is_optimal:
                    first_off_idx = t
                if first_off_idx is not None and recovered_in_k is None and is_optimal:
                    recovered_in_k = t - first_off_idx

                cube.apply(predicted_move)
                history.append(predicted_move)

                if cube.is_solved():
                    break

            solved += int(cube.is_solved())
            total_depths += len(history)

            if first_off_idx is not None:
                rec_denom += 1
                if recovered_in_k is not None:
                    for H in Hs:
                        if recovered_in_k <= H:
                            rec_success[H] += 1

        # ---------- metrics ----------
        sr = solved / max(1, num_samples)
        avg_depth = total_depths / max(1, num_samples)
        stc = (blackout_stc / blackout_total) if blackout_total else 0.0
        prog = (blackout_prog / blackout_total) if blackout_total else 0.0
        pvr = (parse_viol / max(1, total_calls))
        avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
        recovery_cdf = {str(H): (rec_success[H] / rec_denom) if rec_denom else 0.0 for H in Hs}

        payload = {
            "model_name": self.assistant.get_name(),
            "test_type": self.test_type,
            "timestamp": datetime.now().isoformat(),
            "n_moves": self.n_moves,
            "blackout_mode": self.blackout_mode,
            "blackout_interval": self.blackout_interval,
            "blackout_phase": self.blackout_phase,
            "max_steps": self.max_steps,
            "num_samples": num_samples,
            "metrics": {
                "stc": stc,  # optimal-first adherence during blackout steps
                "progress_rate": prog,  # progress during blackout steps
                "solve_rate": sr,
                "avg_depth": avg_depth,
                "parse_violation_rate": pvr,
                "avg_latency_s": avg_lat,
                "recovery_cdf": recovery_cdf,
            },
            "counters": {
                "episodes_solved": solved,
                "total_calls": total_calls,
                "blackout_steps": blackout_total,
                "blackout_stc_hits": blackout_stc,
                "blackout_progress_hits": blackout_prog,
                "recovery_denominator": rec_denom,
            },
        }

        save_results(self.config.results_dir / f"{self.test_type}.json", payload)

        logger.info(
            "[%s] mode=%s | STC=%.3f | progress=%.3f | solve%%=%.1f | parse=%.3f | latency=%.0f ms | recovery=%s",
            self.test_type, self.blackout_mode, stc, prog, sr * 100, pvr, avg_lat * 1000, recovery_cdf,
        )

        return payload

    # ---------- MCQ generation ----------
    def _gen_mcq(self, correct_move: str, rng: random.Random):
        pool = [m for m in VirtualCube.AVAILABLE_MOVES if m != correct_move]
        distractors = rng.sample(pool, 3)
        opts = [correct_move] + distractors
        rng.shuffle(opts)
        d = dict(zip("ABCD", opts))
        gold = next(k for k, v in d.items() if v == correct_move)
        return d, gold