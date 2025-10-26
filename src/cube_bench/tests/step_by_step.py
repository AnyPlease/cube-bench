# ==================================
# file: cube_bench/tests/step_by_step.py
# ==================================
from __future__ import annotations
import logging
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from typing import Dict, Tuple, Optional, List, Any, Set

from tqdm import tqdm
import pycuber as pc  # type: ignore
import kociemba
from copy import deepcopy
import hashlib

from ..core import BaseTest
from ..io import save_results
from cube_bench.sim.cube_simulator import VirtualCube
from cube_bench.prompts.prompt_factory import PromptFactory

logger = logging.getLogger(__name__)


class StepByStepTest(BaseTest):
    """Closed-loop: ask for next move at each step, accept optimal or teacher."""

    # Primary format: <ANSWER> C </ANSWER>
    OPTIONS_RE = re.compile(r"<ANSWER>\s*([ABCD])\s*</ANSWER>", re.IGNORECASE)
    # Fallbacks seen in the wild:
    ALT_RE_1 = re.compile(r"\bANSWER\s*[:=]\s*([ABCD])\b", re.IGNORECASE)   # ANSWER: C
    ALT_RE_2 = re.compile(r"<([ABCD])>", re.IGNORECASE)                     # <C>

    # Abstention: accept <ANSWER> IDK </ANSWER>, ANSWER: IDK/E, or "I don't know"
    IDK_RE = re.compile(
        r"(?:<ANSWER>\s*(IDK)\s*</ANSWER>)|(?:\bANSWER\s*[:=]\s*(?:IDK|E)\b)|(?:I\s*DON'?T\s*KNOW)",
        re.IGNORECASE
    )

    def __init__(
        self,
        assistant,
        config,
        n_moves: int,
        verbose: bool = False,
        idk_enabled: bool = True,
        idk_weight: float = 0.25,
        idk_policy: str = "teacher_on_abstain",
        idk_conf_threshold: Optional[float] = None
    ):
        super().__init__(assistant, config, n_moves, verbose)
        self.test_type = "step_by_step"
        self.n_moves = n_moves
        self.per_step_totals: List[int] = [0] * n_moves
        self.per_step_correct: List[int] = [0] * n_moves
        # first error step index (1-indexed). If a sample has no error, nothing is added.
        self.first_error_step: List[int] = []
        self.confusion = defaultdict(Counter)
        self.latencies: List[float] = []
        self.verbose = verbose
        # Full-entropy RNG (OS-backed; unaffected by random.seed() and hash randomization)
        self._sys_rng: random.Random = random.SystemRandom()

        # ---- IDK (self-contained; no config access) ----
        self.idk_enabled = bool(idk_enabled)
        self.idk_weight = float(idk_weight)           # recommend ≤ 0.25
        self.idk_policy = str(idk_policy)             # "teacher_on_abstain" | "skip_item"
        self.idk_conf_threshold = 50  # Optional[float], e.g., 60.0 (%)
        self.per_step_idk: List[int] = [0] * n_moves  # coverage by step

        if self.idk_enabled:
            logger.info(f"IDK Enabled -- Policy: {self.idk_policy} -- Weight: {self.idk_weight}")

    def _eval(self, text: str, gold_letter: str) -> Tuple[bool, Optional[str]]:
        """Parse model output; return (is_correct, predicted_letter or 'IDK'/None)."""
        if self.idk_enabled and self.IDK_RE.search(text or ""):
            return (False, "IDK")
        m = self.OPTIONS_RE.search(text or "") or self.ALT_RE_1.search(text or "") or self.ALT_RE_2.search(text or "")
        if not m:
            logger.warning("Could not parse model's output")
            logger.info(f"Model's full response:\n\n{text}")
            return False, None
        pred = m.group(1).upper()
        return (pred == gold_letter), pred

    def _move_makes_progress(self, vc: VirtualCube, move: str) -> Tuple[bool, int, int]:
        """Return (decreases_distance, d0, d1). Uses oracle classification, but also reports FTM dists."""
        d0 = vc.get_distance()
        c = vc.clone()
        c.apply(move)
        if c.is_solved():
            return True, d0, 0  # FIX: was (True, 1, 0)
        d1 = c.get_distance()
        return (d1 < d0), d0, d1

    def _optimal_first_moves(self, vc: VirtualCube) -> set[str]:
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

    def _gen_opts_balanced(
        self,
        vc: VirtualCube,
        teacher_move: str,
        rng: Optional[random.Random] = None,
    ) -> Tuple[Dict[str, str], str]:
        """
        Build MCQ options with a controlled mix of distractors.
        Policy: include exactly one additional progress move if available; fill the rest with non-progress.
        Returns (options_map, gold_letter)
        """
        rng = rng or self._sys_rng
        good = self._optimal_first_moves(vc) - {teacher_move}
        bad = [m for m in VirtualCube.AVAILABLE_MOVES if m != teacher_move and m not in good]

        picks: List[str] = []
        if good:
            picks.append(rng.choice(list(good)))
        need = 3 - len(picks)

        if len(bad) >= need:
            picks += rng.sample(bad, need)
        else:
            # Defensive top-up to ensure uniqueness
            pool = [m for m in VirtualCube.AVAILABLE_MOVES if m != teacher_move and m not in picks]
            while len(picks) < 3:
                pick = rng.choice(pool)
                if pick not in picks:
                    picks.append(pick)

        opts = [teacher_move] + picks[:3]
        rng.shuffle(opts)
        d = dict(zip("ABCD", opts))
        gold = next(k for k, v in d.items() if v == teacher_move)
        return d, gold

    def _build_step_by_step_prompts(self, kwargs: Dict[str, Any]) -> Tuple[str, str]:
        """
        Return (sys_prompt, user_prompt).
        If self.idk_enabled is True, use the IDK-aware prompt; otherwise use PromptFactory.
        This method is fully self-contained (no config reads).
        """
        idk_enabled = bool(getattr(self, "idk_enabled", False))
        if not idk_enabled:
            # Original behavior: defer to your PromptFactory template
            return PromptFactory.get("step_by_step", **kwargs)

        n_moves = kwargs.get("n_moves", getattr(self, "n_moves", 2))
        text_state = kwargs.get("textual_representation", "")
        A, B, C, D = kwargs["move_A"], kwargs["move_B"], kwargs["move_C"], kwargs["move_D"]
        conf_thr = getattr(self, "idk_conf_threshold", None)

        sys_prompt = (
            "You are an expert Rubik's-Cube solver. Pick exactly ONE move (A, B, C, D or IDK) "
            "that most reduces the cube's distance to solved.\n\n"
            "Context\n"
            f"- The cube is {n_moves} moves from solved under God's distance (HTM).\n"
            "- You will receive a textual cube state (ground truth) and an image (reference only). "
            "Use the TEXT ONLY to decide.\n\n"
            "Decision rule (deterministic)\n"
            "1) For each candidate, internally simulate that move on the textual state and estimate the resulting distance d1 (HTM).\n"
            "2) If any candidate solves the cube (d1=0), choose that candidate.\n"
            "3) Otherwise choose the candidate with the lowest d1.\n"
            "4) If there is a tie on d1, break ties by letter: A ≺ B ≺ C ≺ D.\n"
        )
        if conf_thr is not None:
            sys_prompt += f"5) If you are <{int(conf_thr)}% confident, abstain with IDK.\n"

        sys_prompt += (
            "\nOutput format (STRICT)\n"
            "Return exactly one of the following on a single line:\n"
            "<ANSWER> A </ANSWER>\n"
            "<ANSWER> B </ANSWER>\n"
            "<ANSWER> C </ANSWER>\n"
            "<ANSWER> D </ANSWER>\n"
            "<ANSWER> IDK </ANSWER>\n\n"
            "No explanations, no extra text.\n"
        )

        user_prompt = (
            f"The cube is {n_moves} moves away from being solved.\n\n"
            "Textual Cube State (ground truth):\n"
            f"{text_state}\n\n"
            "Candidate moves:\n"
            f"A: {A}\n"
            f"B: {B}\n"
            f"C: {C}\n"
            f"D: {D}\n"
            "E: I don't know (abstain)\n\n"
            "Respond with exactly one line (A/B/C/D or IDK):\n"
            "<ANSWER> X </ANSWER>"
        )

        return sys_prompt, user_prompt

    def run(self, num_samples: int):
        """
        Closed-loop multi-step evaluation with optional abstention ("IDK").
        - If self.idk_enabled and model abstains, apply teacher move (default) or stop episode per policy.
        - Logs selective metrics: coverage, selective_accuracy, APA.
        """
        solve_depths: List[int] = []
        all_sample_logs: List[Dict[str, Any]] = []

        # Selective-metrics counters
        n_correct = n_wrong = n_idk = 0
        total_decisions = 0

        for idx in tqdm(range(num_samples), desc=f"Step-by-step ({self.n_moves} moves)"):
            cube = VirtualCube()
            scramble = cube.scramble(random_seed=idx, n_moves=self.n_moves)
            scramble_copy = deepcopy(scramble)
            solution_path = str(scramble_copy.reverse()).split()
            correct_steps = 0
            teacher_help = 0

            if self.verbose:
                logger.info(f"[sample {idx}] Scramble: {scramble}")
                logger.info(f"[sample {idx}] Teacher path: {solution_path}")

            sample_log = {
                "sample_id": idx,
                "scramble": str(scramble),
                "solution_path": solution_path,
                "steps_data": [],
            }

            for step_i, teacher_move in enumerate(solution_path):
                if cube.is_solved():
                    break

                # Prefer public API if available
                try:
                    state_text = str(cube)
                except Exception:
                    state_text = cube._cube.__str__()  # noqa: SLF001

                # Stable, deterministic RNG seed per (depth, sample, step)
                seed_bytes = f"{self.n_moves}:{idx}:{step_i}".encode()
                seed_int = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
                step_rng = random.Random(seed_int)

                # Balanced options (teacher + controlled mix)
                options, gold_letter = self._gen_opts_balanced(cube, teacher_move, rng=step_rng)

                kwargs = {
                    "n_moves": self.n_moves - correct_steps - teacher_help,
                    "n_moves_optimal": (self.n_moves - 1) - correct_steps - teacher_help,
                    "textual_representation": state_text,
                    "move_A": options["A"],
                    "move_B": options["B"],
                    "move_C": options["C"],
                    "move_D": options["D"],
                    "metric": "HTM (Half-Turn Metric)",
                }
                sys_prompt, user_prompt = self._build_step_by_step_prompts(kwargs)

                print(f"User Prompt = {user_prompt}")
                print(f"System Prompt = {sys_prompt}")

                t0 = time.time()
                resp = self.assistant.generate(
                    user_prompt=user_prompt,
                    system_prompt=sys_prompt,
                    image=cube.to_image(),
                    max_new_tokens=2**16,  # you can safely reduce to 128 if you want
                    temperature=0.0,       # lock decoding for evaluation
                )
                self.latencies.append(time.time() - t0)

                is_correct, pred_letter = self._eval(resp, gold_letter)
                options_move = options.get(pred_letter) if pred_letter and pred_letter != "IDK" else None

                # Bookkeeping: totals per step & global decisions
                total_decisions += 1
                self.per_step_totals[step_i] += 1

                # ---- Parse failure: end episode, log it, do NOT apply any move ----
                if pred_letter is None:
                    sample_log["steps_data"].append({
                        "step": step_i,
                        "cube_state": state_text,
                        "options": options,
                        "correct_letter": gold_letter,
                        "full_response": resp,
                        "predicted_letter": None,
                        "is_correct": False,
                        "parse_fail": True,
                    })
                    self.first_error_step.append(step_i + 1)
                    if self.verbose:
                        logger.info(f"[sample {idx}] Parse failure at step {step_i + 1}; ending episode.")
                    break

                # ---- Abstention branch ----
                if self.idk_enabled and pred_letter == "IDK":
                    logger.info("Model responded with IDK.")
                    n_idk += 1
                    self.per_step_idk[step_i] += 1

                    sample_log["steps_data"].append({
                        "step": step_i,
                        "cube_state": state_text,
                        "options": options,
                        "correct_letter": gold_letter,
                        "full_response": resp,
                        "predicted_letter": "IDK",
                        "is_correct": False,
                        "abstained": True,
                        "idk_policy": self.idk_policy,
                    })

                    if self.idk_policy == "teacher_on_abstain":
                        # Keep episode length stable with teacher assist; do not increment correct_steps.
                        cube.apply(teacher_move)
                        teacher_help += 1
                        continue
                    else:  # "skip_item": stop this sample at the abstention step
                        self.first_error_step.append(step_i + 1)
                        break

                # ---- Answered branch (A/B/C/D parsed) ----
                if self.verbose:
                    logger.info(f"Model's chosen option: {pred_letter} -> {options_move}")

                # Per-step accuracy (legacy)
                self.per_step_correct[step_i] += int(is_correct)

                # Global answered counters
                if pred_letter and pred_letter != "IDK":
                    if is_correct:
                        n_correct += 1
                    else:
                        n_wrong += 1

                if options_move is not None:
                    self.confusion[teacher_move][options_move] += 1

                # Log answered step
                sample_log["steps_data"].append({
                    "step": step_i,
                    "cube_state": state_text,
                    "options": options,
                    "correct_letter": gold_letter,
                    "full_response": resp,
                    "predicted_letter": pred_letter,
                    "is_correct": is_correct,
                })

                # Accept teacher or any oracle-optimal move; else progress-aware handling
                good_moves = self._optimal_first_moves(cube)
                if self.verbose:
                    logger.info(f"[Sample: {idx} Step: {step_i}] Oracle-good moves: {sorted(good_moves)}")

                # Only evaluate progress if we actually parsed a concrete move
                made_progress = False
                if options_move is not None:
                    made_progress, d0, d1 = self._move_makes_progress(cube, options_move)

                if options_move and (is_correct or options_move in good_moves):
                    cube.apply(options_move)
                    if is_correct:
                        # exact teacher match -> count toward teacher-adherence
                        correct_steps += 1
                else:
                    if options_move and made_progress:
                        # Not teacher/optimal but still reduces distance: allow it to proceed
                        cube.apply(options_move)
                    else:
                        # No progress (or no valid move) -> end episode at first error, do NOT apply any move
                        self.first_error_step.append(step_i + 1)
                        if self.verbose:
                            logger.info(f"[sample {idx}] First error at step {step_i + 1}")
                        break

            all_sample_logs.append(sample_log)
            solve_depths.append(correct_steps)

        # -------- Aggregate & save --------
        avg_depth = (sum(solve_depths) / len(solve_depths)) if solve_depths else 0.0
        perfect = sum(1 for d in solve_depths if d == self.n_moves)
        step_acc = [c / t if t else 0.0 for c, t in zip(self.per_step_correct, self.per_step_totals)]
        first_err_hist = Counter(self.first_error_step)
        avg_latency = (sum(self.latencies) / len(self.latencies)) if self.latencies else 0.0

        # Selective metrics
        answered = n_correct + n_wrong
        coverage_overall = (answered / total_decisions) if total_decisions else 0.0
        selective_acc = (n_correct / answered) if answered else 0.0
        apa = ((n_correct + self.idk_weight * n_idk) / total_decisions) if total_decisions else 0.0

        coverage_by_step = []
        for t, z in zip(self.per_step_totals, self.per_step_idk):
            cov = ((t - z) / t) if t else 0.0
            coverage_by_step.append(cov)

        logger.info(f"Average Correct Steps (teacher-adherence): {avg_depth:.2f} / {self.n_moves}")
        logger.info(f"Perfect Solves: {perfect}/{len(solve_depths)} ({(perfect/len(solve_depths))*100:.2f}%)")
        logger.info(f"Per-step accuracy: {[round(x, 3) for x in step_acc]}")
        per_step_ns = [int(t) for t in self.per_step_totals]
        logger.info("Per-step accuracy: %s | Per-step Ns: %s", [round(x, 3) for x in step_acc], per_step_ns)
        logger.info(f"Avg latency: {avg_latency*1000:.1f} ms")
        logger.info(
            "Selective metrics — coverage=%.3f, selective_acc=%.3f, IDK=%d, APA(%.2f)=%.3f",
            coverage_overall, selective_acc, n_idk, self.idk_weight, apa
        )

        save_results(
            self.config.results_dir / f"{self.test_type}.json",
            {
                "model_name": self.assistant.get_name(),
                "test_type": self.test_type,
                "n_moves_scrambled": self.n_moves,
                "timestamp": datetime.now().isoformat(),
                "average_solve_depth": avg_depth,
                "perfect_solves_ratio": perfect / max(1, len(solve_depths)),
                # "solve_depths_per_sample": solve_depths,
                "step_accuracy": step_acc,
                "first_error_hist": dict(first_err_hist),
                "confusion_matrix": {k: dict(v) for k, v in self.confusion.items()},
                "avg_latency_ms": avg_latency * 1000,
                "num_samples": len(solve_depths),
                # --- New: selective/abstention summaries ---
                "selective": {
                    "coverage_overall": coverage_overall,
                    "coverage_by_step": coverage_by_step,
                    "selective_accuracy": selective_acc,
                    "n_correct": n_correct,
                    "n_wrong": n_wrong,
                    "n_idk": n_idk,
                    "total_decisions": total_decisions,
                },
                "abstention": {
                    "enabled": self.idk_enabled,
                    "policy": self.idk_policy,
                    "idk_weight": self.idk_weight,
                    "apa": apa,
                    "conf_threshold": self.idk_conf_threshold,
                },
            },
        )
        # save_results(self.config.results_dir / "step_by_step_responses.json", all_sample_logs)
