# ==================================
# file: cube_bench/tests/learning_curve.py
# ==================================
from __future__ import annotations

import logging
import random
import re
import math
import statistics
from tqdm import tqdm
from collections import Counter, deque
from datetime import datetime
from typing import Dict, Tuple, Set, Deque, Optional, List

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot on headless systems
from matplotlib import pyplot as plt

from ..core import BaseTest
from ..io import save_results
from cube_bench.sim.cube_simulator import VirtualCube
from cube_bench.prompts.prompt_factory import PromptFactory

logger = logging.getLogger(__name__)


class LearningCurveTest(BaseTest):
    """
    Multiple-choice 'learning curve' test under closed-loop control.

    Loop:
      - Scramble cube (depth = self.n_moves)
      - Build MCQ with exactly one correct option (if possible) from the current state
      - Ask model to pick A/B/C/D (image + text provided)
      - Apply model move under an 'accept-progress' policy:
          * If equals ground-truth plan head → apply, advance plan
          * Else if move strictly decreases distance → accept and replan
          * Else apply anyway, but push its inverse to plan (attempt recovery)

    Logged / saved aggregates (JSON):
      - success_rate (SR)
      - sr_ci95 (Wilson 95% CI for SR)
      - p1 (fraction solved in exactly 1 attempt, over all runs)
      - p_le_3 (fraction solved in ≤3 attempts, over all runs; capped by max_attempts)
      - med_at_solved (median # attempts among solved runs, or null if none)
      - avg_attempts_all_maxed (Avg@All: failures counted as max_attempts)
      - attempts_needed (per-run attempts)
      - solved_flags (per-run boolean success)
      - hist_counts (attempt histogram among solved runs)
      - plot_path (PNG histogram saved alongside json)
    """

    # Parse either <ANSWER> X </ANSWER> or "ANSWER: X"
    ANSWER_TAG_RE = re.compile(r"<\s*ANSWER\s*>\s*([ABCD])\s*<\s*/\s*ANSWER\s*>", re.IGNORECASE)
    ANSWER_COLON_RE = re.compile(r"\bANSWER\s*:\s*([ABCD])\b", re.IGNORECASE)

    HIST_FIG_NAME = "learning_curve_hist.png"

    def __init__(
        self,
        assistant,
        config,
        n_moves: int,
        max_attempts: int = 6,
        accept_progress: bool = True,
        verbose: bool = False,
    ):
        super().__init__(assistant, config, n_moves, verbose)
        self.test_type = "learning_curve"
        self.n_moves = int(n_moves)
        self.max_attempts = int(max_attempts)
        self.accept_progress = bool(accept_progress)
        self.verbose = bool(verbose)
        # Full-entropy RNG unaffected by random.seed() or process hash randomization
        self._sys_rng: random.Random = random.SystemRandom()

    # ---------- utilities ----------

    def _vlog(self, *a: object) -> None:
        if self.verbose:
            logger.info(" ".join(str(x) for x in a))

    @staticmethod
    def _inverse(move: str) -> str:
        """
        Return the group-theoretic inverse of a quarter/half-turn in standard notation.
        Example: 'R' -> "R'", "R'" -> 'R', 'R2' -> 'R2'.
        """
        move = move.strip()
        if not move:
            return move
        if move.endswith("2"):
            return move
        if move.endswith("'"):
            return move[:-1]
        return move + "'"

    @staticmethod
    def _parse_answer(text: str) -> Optional[str]:
        m = LearningCurveTest.ANSWER_TAG_RE.search(text)
        if not m:
            m = LearningCurveTest.ANSWER_COLON_RE.search(text)
        return m.group(1).upper() if m else None

    @staticmethod
    def _distance(vc: VirtualCube) -> int:
        return vc.get_distance()

    def _progress(self, vc: VirtualCube, move: str) -> Tuple[bool, int, int]:
        """Return (made_progress, d0, d1) after hypothetically applying `move`."""
        d0 = self._distance(vc)
        c = vc.clone()
        c.apply(move)
        d1 = 0 if c.is_solved() else self._distance(c)
        return (d1 < d0), d0, d1

    def _optimal_first_moves(self, vc: VirtualCube) -> Set[str]:
        """
        Neighbor moves that strictly DECREASE distance or solve immediately.
        Uses the same distance function as _distance for consistency.
        """
        if vc.is_solved():
            return set()

        baseline = vc.get_distance()
        good: Set[str] = set()
        for m in VirtualCube.AVAILABLE_MOVES:
            c = vc.clone()
            c.apply(m)
            if c.is_solved() or c.get_distance() < baseline:
                good.add(m)
        return good

    def _replan(self, vc: VirtualCube) -> Deque[str]:
        """
        Plan from current state using the cube's solver; return empty deque on failure.
        """
        if vc.is_solved():
            return deque()
        try:
            seq = vc.solve()
            return deque(seq.split()) if seq else deque()
        except Exception as e:
            logger.warning(f"[replan] VirtualCube.solve() failed: {e!r}")
            return deque()

    @staticmethod
    def _gen_opts(good_moves: Set[str], rng: random.Random) -> Tuple[Dict[str, str], str]:
        """
        Build MCQ options ensuring:
          - Exactly one 'correct' when possible (chosen from good_moves)
          - Three distractors not in good_moves and not equal to correct
          - Always returns 4 unique options; shuffles A/B/C/D with provided RNG.
        """
        all_moves: List[str] = list(VirtualCube.AVAILABLE_MOVES)

        if good_moves:
            correct = rng.choice(tuple(good_moves))
            pool = [m for m in all_moves if (m != correct and m not in good_moves)]
            if len(pool) >= 3:
                distractors = rng.sample(pool, 3)
            else:
                # fallback: allow any not-equal moves; top-up if needed
                pool = [m for m in all_moves if m != correct]
                distractors = rng.sample(pool, k=min(3, len(pool)))
                # top-up without duplicates if pool < 3 (very rare)
                while len(distractors) < 3:
                    pick = rng.choice(pool)
                    if pick not in distractors:
                        distractors.append(pick)
        else:
            correct = rng.choice(all_moves)
            pool = [m for m in all_moves if m != correct]
            distractors = rng.sample(pool, 3)

        opts = [correct] + distractors[:3]
        rng.shuffle(opts)
        mapping = dict(zip("ABCD", opts))
        gold_letter = next(k for k, v in mapping.items() if v == correct)
        return mapping, gold_letter

    @staticmethod
    def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
        """Wilson score interval for a Bernoulli proportion."""
        if n <= 0:
            return (0.0, 0.0)
        p = successes / n
        denom = 1.0 + (z ** 2) / n
        center = (p + (z ** 2) / (2 * n)) / denom
        margin = z * math.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * n * n)) / denom
        return (max(0.0, center - margin), min(1.0, center + margin))

    # ---------- main ----------

    def run(self, num_samples: int) -> None:
        """
        Execute the test over `num_samples` scrambles.

        Writes:
          - JSON results to {results_dir}/learning_curve.json
          - Histogram PNG to {results_dir}/learning_curve_hist.png
        """
        attempts_needed: List[int] = []
        solved_flags: List[bool] = []

        self._vlog(
            f"[start] test={self.test_type} model={self.assistant.get_name()} "
            f"samples={num_samples} depth={self.n_moves} max_attempts={self.max_attempts} "
            f"accept_progress={self.accept_progress}"
        )

        for idx in tqdm(range(num_samples), desc=f"Learning-curve ({self.n_moves} moves)"):
            cube = VirtualCube()
            scramble = cube.scramble(random_seed=idx, n_moves=self.n_moves)
            plan: Deque[str] = deque(str(scramble.reverse()).split())  # ground-truth inverse

            attempts = 0
            self._vlog(f"[sample {idx}] start d={self._distance(cube)} scramble={list(scramble)}")

            while not cube.is_solved() and attempts < self.max_attempts:
                # Fully random option construction each attempt
                rng = self._sys_rng

                if not plan:
                    plan = self._replan(cube)
                    if not plan:
                        self._vlog(f"[sample {idx}] replan: empty; abort sample")
                        break

                good = self._optimal_first_moves(cube)
                correct_move = plan[0] if plan else None
                if not correct_move:
                    self._vlog(f"[sample {idx}] plan head missing; abort sample")
                    break

                options, gold_letter = self._gen_opts(good, rng)

                # Prefer public text representation; fallback to internal if needed
                try:
                    state_text = str(cube)
                except Exception:
                    state_text = cube._cube.__str__()  # noqa: SLF001

                d_cur = self._distance(cube)
                kwargs = {
                    "n_moves": d_cur,  # or len(plan) if you want plan-depth semantics
                    "n_moves_optimal": max(d_cur - 1, 0),
                    "textual_representation": state_text,
                    "move_A": options["A"],
                    "move_B": options["B"],
                    "move_C": options["C"],
                    "move_D": options["D"],
                }
                sys_prompt, user_prompt = PromptFactory.get("learning_curve", **kwargs)

                resp = self.assistant.generate(
                    user_prompt=user_prompt,
                    system_prompt=sys_prompt,
                    image=cube.to_image(),
                    max_new_tokens=2**16,
                    temperature=0.0,
                    # stop=["</ANSWER>"],  # uncomment if your backend supports stop sequences
                )

                pred_letter = self._parse_answer(resp or "")
                predicted_move = options.get(pred_letter) if pred_letter else None
                if not predicted_move:
                    attempts += 1
                    logger.warning(f"[sample {idx}] parse-failure; pred_letter={pred_letter!r}")
                    continue

                attempts += 1
                made_progress, d0, d1 = self._progress(cube, predicted_move)

                if predicted_move == correct_move:
                    cube.apply(predicted_move)
                    plan.popleft()
                    decision = "APPLY_MATCH"
                elif predicted_move in good:
                    cube.apply(predicted_move)
                    plan = self._replan(cube)
                    decision = "APPLY_DECREASE(replan)"
                else:
                    cube.apply(predicted_move)
                    plan.appendleft(self._inverse(predicted_move))
                    decision = "APPLY_WITH_INVERSE"

                self._vlog(
                    f"[sample {idx}] attempt={attempts} d:{d0}->{d1} "
                    f"pred={predicted_move} gold={gold_letter} correct={correct_move} "
                    f"good={predicted_move in good} progress={made_progress} action={decision}"
                )

            attempts_needed.append(attempts)
            solved = cube.is_solved()
            solved_flags.append(solved)
            self._vlog(f"[sample {idx}] done solved={solved} attempts={attempts}")

        # --- Aggregates & histogram ---
        n = max(1, len(attempts_needed))
        solved_n = int(sum(solved_flags))
        success_rate = solved_n / n

        # Histogram among solved
        counts = Counter([a for a, s in zip(attempts_needed, solved_flags) if s])
        xs = list(range(1, self.max_attempts + 1))
        ys = [counts.get(k, 0) for k in xs]

        # CI and probabilities (over ALL runs)
        ci_lo, ci_hi = self._wilson_ci(solved_n, n)
        p1 = counts.get(1, 0) / n
        kmax = min(3, self.max_attempts)
        p_le_3 = sum(counts.get(k, 0) for k in range(1, kmax + 1)) / n

        # Med@Solved
        solved_attempts = [a for a, s in zip(attempts_needed, solved_flags) if s]
        med_at_solved: Optional[float]
        med_at_solved = statistics.median(solved_attempts) if solved_attempts else None

        # Average attempts across all runs, counting failures as max_attempts
        avg_attempts_all_maxed = sum(
            (a if s else self.max_attempts) for a, s in zip(attempts_needed, solved_flags)
        ) / n

        # (Legacy) raw average of attempts regardless of success (kept for reference)
        avg_attempts_all = sum(attempts_needed) / n

        # Plot histogram
        fig_path = self.config.results_dir / self.HIST_FIG_NAME
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plt.figure(figsize=(8, 5))
            plt.bar(xs, ys)
            plt.xticks(xs)
            plt.xlabel("Attempts (when solved)")
            plt.ylabel("Number of runs")
            unsolved_n = int(len(solved_flags) - solved_n)
            plt.title(
                f"Solve Attempts Distribution (N={len(solved_flags)}, "
                f"Solved={solved_n}, Unsolved={unsolved_n}, SR={success_rate:.2%})"
            )
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=160)
        finally:
            plt.close()

        # Persist results
        payload = {
            "model_name": self.assistant.get_name(),
            "test_type": self.test_type,
            "n_moves": self.n_moves,
            "timestamp": datetime.now().isoformat(),
            "max_attempts": self.max_attempts,
            "accept_progress": self.accept_progress,
            # per-run
            "attempts_needed": attempts_needed,
            "solved_flags": solved_flags,
            # aggregates for table
            "n": n,
            "solved_n": solved_n,
            "success_rate": success_rate,           # SR
            "sr_ci95": [ci_lo, ci_hi],              # 95% CI
            "p1": p1,                               # P(1)
            "p_le_3": p_le_3,                       # P(≤3)
            "med_at_solved": med_at_solved,         # Med@Solved
            "avg_attempts_all_maxed": avg_attempts_all_maxed,  # Avg@All (failures=max_attempts)
            # extras
            "avg_attempts_all": avg_attempts_all,   # legacy mean (not used in table)
            "hist_counts": {int(k): int(v) for k, v in counts.items()},
            "plot_path": str(fig_path),
        }
        save_results(self.config.results_dir / f"{self.test_type}.json", payload)

        self._vlog(
            f"[end] SR={success_rate:.2%} CI95=({ci_lo:.3f},{ci_hi:.3f}) "
            f"P(1)={p1:.3f} P(≤3)={p_le_3:.3f} "
            f"Med@Solved={med_at_solved if med_at_solved is not None else 'NA'} "
            f"Avg@All={avg_attempts_all_maxed:.2f} "
            f"results={self.config.results_dir / f'{self.test_type}.json'} "
            f"plot={fig_path}"
        )
