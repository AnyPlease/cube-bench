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
    Multiple-choice 'learning curve' test, starting from the first Closed-Loop failure.

    Protocol for each scramble:
      1) Scramble cube (depth = self.n_moves).
      2) Run a Closed-Loop control phase under an 'accept-progress' policy:
           - If equals ground-truth plan head → apply, advance plan.
           - Else if move strictly decreases distance → accept and replan.
           - Else: this is the FIRST non-progress step (failure).
         A parse failure (no valid A/B/C/D) is also treated as a failure.
      3) From the resulting post-error state (and updated plan), run up to
         `max_attempts` additional MCQ decisions under the same policy.
         These are the "learning-curve" attempts.

    Metrics are computed over episodes that enter the learning-curve regime
    (i.e., scrambles where a failure actually occurred in step 2):

      - success_rate (SR)     : fraction of post-error episodes that solve
                                within max_attempts.
      - sr_ci95              : Wilson 95% CI for SR.
      - p1                   : P(1) = fraction of post-error episodes solved
                                in exactly 1 attempt.
      - p_le_3               : P(≤3) = fraction solved in ≤3 attempts
                                (capped by max_attempts).
      - med_at_solved        : median # attempts among solved post-error episodes
                                (None if none solved).
      - avg_attempts_all_maxed:
                                Avg@All post-error episodes, counting failures as
                                max_attempts.
      - attempts_needed      : per-episode attempts (post-error only).
      - solved_flags         : per-episode boolean success (post-error only).
      - hist_counts          : histogram of attempts among solved post-error episodes.
      - plot_path            : PNG histogram saved alongside json.

    We also log:
      - total_scrambles      : total scrambles sampled.
      - episodes_with_failure: how many scrambles produced a failure and thus
                               entered the learning-curve regime.
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

        For each scramble:
          - Run Closed-Loop until first failure (non-progress move or parse error).
          - From the resulting post-error state, run up to max_attempts additional
            decisions and log attempts / success.

        Writes:
          - JSON results to {results_dir}/learning_curve.json
          - Histogram PNG to {results_dir}/learning_curve_hist.png
        """
        attempts_needed: List[int] = []   # post-error attempts
        solved_flags: List[bool] = []     # success after post-error attempts
        pre_fail_reasons: List[str] = []

        total_scrambles = 0
        episodes_with_failure = 0

        self._vlog(
            f"[start] test={self.test_type} model={self.assistant.get_name()} "
            f"samples={num_samples} depth={self.n_moves} max_attempts={self.max_attempts} "
            f"accept_progress={self.accept_progress}"
        )

        for idx in tqdm(range(num_samples), desc=f"Learning-curve ({self.n_moves} moves)"):
            total_scrambles += 1

            cube = VirtualCube()
            scramble = cube.scramble(random_seed=idx, n_moves=self.n_moves)
            plan: Deque[str] = deque(str(scramble.reverse()).split())  # ground-truth inverse

            self._vlog(
                f"[sample {idx}] pre-phase start d={self._distance(cube)} "
                f"scramble={list(scramble)}"
            )

            # --- Phase 1: Closed-Loop prelude until first failure ---
            failure_happened = False
            failure_reason: Optional[str] = None

            while not cube.is_solved():
                rng = self._sys_rng

                if not plan:
                    plan = self._replan(cube)
                    if not plan:
                        self._vlog(f"[sample {idx}] pre-phase replan: empty; abort scramble")
                        failure_reason = "pre_replan_empty"
                        break

                good = self._optimal_first_moves(cube)
                correct_move = plan[0] if plan else None
                if not correct_move:
                    self._vlog(f"[sample {idx}] pre-phase plan head missing; abort scramble")
                    failure_reason = "pre_plan_missing"
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

                # Parse failure → first failure point
                if not predicted_move:
                    self._vlog(
                        f"[sample {idx}] pre-phase parse failure; "
                        f"pred_letter={pred_letter!r}, resp={resp!r}"
                    )
                    failure_happened = True
                    failure_reason = "parse_error"
                    break

                made_progress, d0, d1 = self._progress(cube, predicted_move)

                if predicted_move in good:
                    # Progress step: behave like Closed-Loop and keep going.
                    if predicted_move == correct_move:
                        cube.apply(predicted_move)
                        plan.popleft()
                        decision = "APPLY_MATCH(pre)"
                    else:
                        cube.apply(predicted_move)
                        plan = self._replan(cube)
                        decision = "APPLY_DECREASE(pre_replan)"

                    self._vlog(
                        f"[sample {idx}] pre-phase d:{d0}->{d1} "
                        f"pred={predicted_move} gold={gold_letter} correct={correct_move} "
                        f"good=True progress={made_progress} action={decision}"
                    )
                    continue

                # Non-progress step → first failure point
                cube.apply(predicted_move)
                plan.appendleft(self._inverse(predicted_move))
                decision = "APPLY_WITH_INVERSE(pre_error)"

                self._vlog(
                    f"[sample {idx}] pre-phase FAILURE d:{d0}->{d1} "
                    f"pred={predicted_move} gold={gold_letter} correct={correct_move} "
                    f"good=False progress={made_progress} action={decision}"
                )

                failure_happened = True
                failure_reason = "non_progress"
                break

            # If cube solved with no failure, or pre-phase aborted without a proper failure:
            if not failure_happened or cube.is_solved():
                self._vlog(
                    f"[sample {idx}] no eligible failure for learning-curve "
                    f"(failure={failure_happened}, reason={failure_reason}, "
                    f"solved={cube.is_solved()}); skipping LC episode"
                )
                continue

            # We have a post-error state; enter learning-curve regime.
            episodes_with_failure += 1
            pre_fail_reasons.append(failure_reason or "unknown")

            self._vlog(
                f"[sample {idx}] LC-phase start from failure={failure_reason} "
                f"d_post={self._distance(cube)}"
            )

            # --- Phase 2: Learning-curve attempts from post-error state ---
            attempts = 0

            while not cube.is_solved() and attempts < self.max_attempts:
                rng = self._sys_rng

                if not plan:
                    plan = self._replan(cube)
                    if not plan:
                        self._vlog(f"[sample {idx}] LC-phase replan: empty; abort LC episode")
                        break

                good = self._optimal_first_moves(cube)
                correct_move = plan[0] if plan else None
                if not correct_move:
                    self._vlog(f"[sample {idx}] LC-phase plan head missing; abort LC episode")
                    break

                options, gold_letter = self._gen_opts(good, rng)

                try:
                    state_text = str(cube)
                except Exception:
                    state_text = cube._cube.__str__()  # noqa: SLF001

                d_cur = self._distance(cube)
                kwargs = {
                    "n_moves": d_cur,
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
                )

                pred_letter = self._parse_answer(resp or "")
                predicted_move = options.get(pred_letter) if pred_letter else None

                if not predicted_move:
                    attempts += 1
                    logger.warning(
                        f"[sample {idx}] LC-phase parse failure; "
                        f"pred_letter={pred_letter!r}, resp={resp!r}"
                    )
                    continue

                attempts += 1
                made_progress, d0, d1 = self._progress(cube, predicted_move)

                if predicted_move == correct_move:
                    cube.apply(predicted_move)
                    plan.popleft()
                    decision = "APPLY_MATCH(LC)"
                elif predicted_move in good:
                    cube.apply(predicted_move)
                    plan = self._replan(cube)
                    decision = "APPLY_DECREASE(LC_replan)"
                else:
                    cube.apply(predicted_move)
                    plan.appendleft(self._inverse(predicted_move))
                    decision = "APPLY_WITH_INVERSE(LC)"

                self._vlog(
                    f"[sample {idx}] LC-phase attempt={attempts} d:{d0}->{d1} "
                    f"pred={predicted_move} gold={gold_letter} correct={correct_move} "
                    f"good={predicted_move in good} progress={made_progress} action={decision}"
                )

            attempts_needed.append(attempts)
            solved = cube.is_solved()
            solved_flags.append(solved)
            self._vlog(
                f"[sample {idx}] LC-phase done solved={solved} attempts={attempts}"
            )

        # --- Aggregates & histogram (over post-error episodes only) ---
        n = len(attempts_needed)
        if n == 0:
            # No episodes entered LC regime; produce a mostly-zero payload.
            success_rate = 0.0
            ci_lo, ci_hi = 0.0, 0.0
            solved_n = 0
            counts: Counter[int] = Counter()
            p1 = 0.0
            p_le_3 = 0.0
            med_at_solved: Optional[float] = None
            avg_attempts_all_maxed = 0.0
            avg_attempts_all = 0.0
            xs = list(range(1, self.max_attempts + 1))
            ys = [0 for _ in xs]
        else:
            solved_n = int(sum(solved_flags))
            success_rate = solved_n / n

            # Histogram among solved
            counts = Counter([a for a, s in zip(attempts_needed, solved_flags) if s])
            xs = list(range(1, self.max_attempts + 1))
            ys = [counts.get(k, 0) for k in xs]

            # CI and probabilities (over post-error episodes)
            ci_lo, ci_hi = self._wilson_ci(solved_n, n)
            p1 = counts.get(1, 0) / n
            kmax = min(3, self.max_attempts)
            p_le_3 = sum(counts.get(k, 0) for k in range(1, kmax + 1)) / n

            # Med@Solved
            solved_attempts = [a for a, s in zip(attempts_needed, solved_flags) if s]
            med_at_solved = statistics.median(solved_attempts) if solved_attempts else None

            # Average attempts across all episodes, counting failures as max_attempts
            avg_attempts_all_maxed = sum(
                (a if s else self.max_attempts) for a, s in zip(attempts_needed, solved_flags)
            ) / n

            # (Legacy) raw average of attempts regardless of success
            avg_attempts_all = sum(attempts_needed) / n

        # Plot histogram
        fig_path = self.config.results_dir / self.HIST_FIG_NAME
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plt.figure(figsize=(8, 5))
            plt.bar(xs, ys)
            plt.xticks(xs)
            plt.xlabel("Attempts (when solved)")
            plt.ylabel("Number of post-error episodes")
            unsolved_n = int(n - solved_n)
            plt.title(
                f"Solve Attempts Distribution (post-error; N={n}, "
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
            # bookkeeping
            "total_scrambles": total_scrambles,
            "episodes_with_failure": episodes_with_failure,
            "pre_fail_reasons": pre_fail_reasons,
            # per-episode (post-error only)
            "attempts_needed": attempts_needed,
            "solved_flags": solved_flags,
            # aggregates for table (post-error only)
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
            f"[end] total_scrambles={total_scrambles} "
            f"episodes_with_failure={episodes_with_failure} "
            f"SR={success_rate:.2%} CI95=({ci_lo:.3f},{ci_hi:.3f}) "
            f"P(1)={p1:.3f} P(≤3)={p_le_3:.3f} "
            f"Med@Solved={med_at_solved if med_at_solved is not None else 'NA'} "
            f"Avg@All={avg_attempts_all_maxed:.2f} "
            f"results={self.config.results_dir / f'{self.test_type}.json'} "
            f"plot={fig_path}"
        )
