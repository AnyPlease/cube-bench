# =====================================
# file: cube_bench/tests/invariance_sweep.py
# =====================================
from __future__ import annotations
import logging
import random
import re
from tqdm import tqdm
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from ..core import BaseTest
from ..io import save_results

# Support either module path for VirtualCube
try:
    from cube_bench.sim.virtual_cube import VirtualCube
except Exception:
    from cube_bench.sim.cube_simulator import VirtualCube  # fallback

# PromptFactory is optional; we provide a fallback prompt
try:
    from cube_bench.prompts.prompt_factory import PromptFactory
    HAS_PROMPT_FACTORY = True
except Exception:
    PromptFactory = None  # type: ignore
    HAS_PROMPT_FACTORY = False

logger = logging.getLogger(__name__)


class InvarianceSweepTest(BaseTest):
    """
    Closed-loop single-step MCQ under visual perturbations and recolor conflict.
    Variants supported by VirtualCube: clean, occl, bright, recolor.
    """

    # Accept either XML-like or colon answer formats (A-D, case-insensitive)
    OPTIONS_RE = re.compile(
        r"(?:<ANSWER>\s*([ABCD])\s*</ANSWER>|ANSWER:\s*([ABCD]))",
        re.IGNORECASE,
    )

    def __init__(
        self,
        assistant,
        config,
        n_moves: int = 3,
        verbose: bool = False,
        balance_gold_letters: bool = True,   # NEW: ensure uniform A/B/C/D as gold
        add_labels: Optional[bool] = True,   # expose label toggling (can bias)
        max_new_tokens: int = 64,            # sane default
    ):
        # Be compatible with different BaseTest signatures
        try:
            super().__init__(assistant, config)
        except TypeError:
            super().__init__(assistant, config, n_moves, verbose)  # type: ignore

        self.test_type = "invariance_sweep"
        self.n_moves = n_moves
        self.verbose = verbose
        self.balance_gold_letters = balance_gold_letters
        self.add_labels = add_labels if add_labels is not None else True
        self.max_new_tokens = max_new_tokens

        # Bijective recolor mapping for the 'recolor' conflict condition (maximally challenging)
        self.recolor_map = {
            "white": "yellow", "yellow": "white",
            "orange": "red",   "red": "orange",
            "green": "blue",   "blue": "green",
        }

    # ---------- helpers ----------
    def _parse(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        m = self.OPTIONS_RE.search(text)
        if not m:
            return None
        return (m.group(1) or m.group(2)).upper()

    def _gen_opts(
        self,
        correct_move: str,
        rng: random.Random,
        force_letter: str | None = None
    ) -> Tuple[Dict[str, str], str]:
        moves = getattr(
            VirtualCube,
            "AVAILABLE_MOVES",
            ["U","U'","U2","D","D'","D2","L","L'","L2","R","R'","R2","F","F'","F2","B","B'","B2"]
        )
        pool = [m for m in moves if m != correct_move]
        distractors = rng.sample(pool, 3)

        # Shuffle distractors only
        rng.shuffle(distractors)

        if force_letter is None:
            # Old behavior: shuffle all 4 and let RNG place gold anywhere
            opts = [correct_move] + distractors
            rng.shuffle(opts)
            d = dict(zip("ABCD", opts))
            gold = next(k for k, v in d.items() if v == correct_move)
            return d, gold

        # New behavior: force gold letter, randomize distractor placement in the remaining slots
        letters = list("ABCD")
        letters.remove(force_letter)
        d = {force_letter: correct_move}
        for L, mv in zip(letters, distractors):
            d[L] = mv
        return d, force_letter

    def _fallback_prompts(self, **kw) -> tuple[str, str]:
        """
        Minimal, model-agnostic prompts if PromptFactory is unavailable.
        Encourages strict A–D answer with our accepted formats.
        """
        sys_prompt = (
            "You are given an image of a Rubik's Cube net and a multiple-choice question.\n"
            "You will also receive a TEXTUAL serialization of the state.\n"
            "If the image and the TEXT conflict (e.g., recolored stickers), FOLLOW THE TEXT.\n"
            "Choose exactly one option: A, B, C, or D.\n"
            "Respond using either the XML tag <ANSWER> A </ANSWER> or 'ANSWER: A' with just one letter."
        )
        user_prompt = (
            f"TEXT (state serialization):\n{kw.get('textual_representation')}\n\n"
            f"Start from a {kw.get('n_moves')}-move scramble. Which next move best adheres to the known "
            f"inverse-scramble teacher path?\n\n"
            f"A) {kw.get('move_A')}\n"
            f"B) {kw.get('move_B')}\n"
            f"C) {kw.get('move_C')}\n"
            f"D) {kw.get('move_D')}\n\n"
            "Answer with exactly one letter (A–D) using <ANSWER> A </ANSWER> or 'ANSWER: A'."
        )
        return sys_prompt, user_prompt

    # ---------- main ----------
    def run(self, num_samples: int):
        # NOTE: 'rot90' is omitted because VirtualCube._augment_image doesn't implement it.
        variants = ["clean", "occl", "bright", "recolor"]

        if self.verbose:
            logger.info(
                f"[InvarianceSweep] model={self.assistant.get_name()} "
                f"n_moves={self.n_moves} samples={num_samples} variants={variants} "
                f"recolor_map={self.recolor_map} add_labels={self.add_labels} "
                f"balance_gold_letters={self.balance_gold_letters}"
            )

        acc: Dict[str, List[int]] = defaultdict(list)
        lat: Dict[str, List[float]] = defaultdict(list)
        parse_ok: Dict[str, List[int]] = defaultdict(list)
        pred_label_hist: Dict[str, Counter] = defaultdict(Counter)
        gold_hist: Counter = Counter()  # NEW: track gold-letter prior across all items
        errors = Counter()
        N = Counter()

        for idx in tqdm(range(num_samples), desc="Invariance sweep test", disable=not self.verbose):
            # Deterministic RNG per sample for reproducibility
            rng = random.Random(idx)  # CHANGED: per-sample deterministic RNG

            cube = VirtualCube()
            scramble = cube.scramble(random_seed=idx, n_moves=self.n_moves)

            # Teacher inverse (first optimal step for MCQ)
            sol_moves = str(scramble.reverse()).split()
            if not sol_moves:
                if self.verbose:
                    logger.warning(f"[{idx:04d}] Empty solution after scramble; skipping sample.")
                continue
            correct_move = sol_moves[0]

            # Build a single A–D option set per sample (reused across variants)
            forced = "ABCD"[idx % 4] if self.balance_gold_letters else None  # CHANGED: balance prior
            options, gold_letter = self._gen_opts(correct_move, rng, force_letter=forced)
            gold_hist[gold_letter] += 1  # NEW

            # Prompts (PromptFactory if available; else fallback)
            kwargs = {
                "n_moves": self.n_moves,
                "n_moves_optimal": max(0, self.n_moves - 1),
                "textual_representation": str(cube),  # stable, public
                "move_A": options["A"], "move_B": options["B"],
                "move_C": options["C"], "move_D": options["D"],
                "instruction_conflict_policy": (
                    "If image and text conflict (e.g., recolor), follow the TEXT and ignore misleading colors."
                ),
            }

            if HAS_PROMPT_FACTORY:
                try:
                    sys_prompt, user_prompt = PromptFactory.get("invariance_sweep", **kwargs)  # type: ignore
                    if self.verbose and idx == 0:
                        logger.info("[Prompts] Using PromptFactory template 'invariance_sweep'.")
                except Exception as e:
                    sys_prompt, user_prompt = self._fallback_prompts(**kwargs)
                    if self.verbose and idx == 0:
                        logger.info(f"[Prompts] PromptFactory failed with {e!r}; using fallback prompts.")
            else:
                sys_prompt, user_prompt = self._fallback_prompts(**kwargs)
                if self.verbose and idx == 0:
                    logger.info("[Prompts] PromptFactory not available; using fallback prompts.")

            # Render all variants (VirtualCube handles recolor internally)
            try:
                images = cube.render_variants(
                    variants,
                    recolor_map=self.recolor_map,
                    return_type="pil",
                    cell_size=60, sticker_border=2, face_gap=40, dpi=100,
                    add_labels=self.add_labels,  # CHANGED: configurable
                )
            except Exception as e:
                logger.error(f"[{idx:04d}] Render failed for all variants: {e!r}")
                for v in variants:
                    errors[v] += 1
                continue

            # Query the assistant once per variant with the SAME A–D options
            for vname in variants:
                t0 = time.time()
                try:
                    resp = self.assistant.generate(  # type: ignore
                        user_prompt=user_prompt,
                        system_prompt=sys_prompt,
                        image=images[vname],
                        max_new_tokens=self.max_new_tokens,  # CHANGED
                    )
                except Exception as e:
                    errors[vname] += 1
                    if self.verbose:
                        logger.warning(f"[{idx:04d} {vname}] Generation failed: {e!r}")
                    continue
                latency = time.time() - t0

                # Normalize response to text
                if not isinstance(resp, str):
                    resp_text = getattr(resp, "text", None)
                    if resp_text is None:
                        resp_text = str(resp)
                else:
                    resp_text = resp

                pred_letter = self._parse(resp_text)
                parsed = int(pred_letter is not None)
                parse_ok[vname].append(parsed)
                if pred_letter:
                    pred_label_hist[vname][pred_letter] += 1

                is_correct = int(pred_letter == gold_letter)
                acc[vname].append(is_correct)
                lat[vname].append(latency)
                N[vname] += 1

                if self.verbose:
                    logger.info(
                        f"[{idx:04d} {vname}] gold={gold_letter}:{correct_move} "
                        f"pred={pred_letter} correct={bool(is_correct)} parsed={bool(parsed)} "
                        f"latency_ms={latency*1000:.1f}"
                    )

        # ---------- aggregation ----------
        def _avg(xs: List[float]) -> float:
            return (sum(xs) / len(xs)) if xs else 0.0

        clean_acc = _avg(acc["clean"]) if acc["clean"] else 0.0
        summary, counts = {}, {}
        for v in variants:
            n = N[v]
            va = _avg(acc[v])
            vl = _avg(lat[v])
            pr = _avg(parse_ok[v])
            prior_pred = {k: int(vv) for k, vv in pred_label_hist[v].items()}
            summary[v] = {
                "n": int(n),
                "accuracy": va,
                "avg_latency_s": vl,
                "delta_vs_clean": (va - clean_acc) if v != "clean" else 0.0,
                "parse_rate": pr,
                "pred_label_prior_counts": prior_pred,  # renamed for clarity
                "errors": int(errors[v]),
            }
            counts[v] = {
                "correct": int(sum(acc[v])),
                "total": int(n),
                "parsed": int(sum(parse_ok[v])),
            }

        if self.verbose:
            logger.info("[InvarianceSweep] Per-variant summary:")
            for v in variants:
                s = summary[v]
                logger.info(
                    f"  - {v:7s} n={s['n']} acc={s['accuracy']:.3f} "
                    f"ΔvsClean={s['delta_vs_clean']:+.3f} "
                    f"parse_rate={s['parse_rate']:.3f} "
                    f"latency_ms={s['avg_latency_s']*1000:.1f} "
                    f"errors={s['errors']} pred_labels={s['pred_label_prior_counts']}"
                )

        # Constants from VirtualCube’s augmentations (document for reproducibility)
        perturb_params = {
            "occl_band_frac_h": 0.15,     # ~15% height (see _augment_image)
            "bright_factor": 0.8,         # darker jitter (see _augment_image)
            "recolor_map": self.recolor_map,
            "note": "No rotation condition because VirtualCube does not implement 'rot90'.",
        }

        # ---------- save ----------
        payload = {
            "model_name": self.assistant.get_name(),
            "test_type": self.test_type,
            "n_moves": self.n_moves,
            "timestamp": datetime.now().isoformat(),
            "variants": variants,
            "perturb_params": perturb_params,
            "summary": summary,
            "counts": counts,
            "gold_prior_counts": dict(gold_hist),  # NEW: verify A/B/C/D balance
            "plot_data": {
                "x": variants,
                "y_accuracy": [summary[v]["accuracy"] for v in variants],
                "y_delta_vs_clean": [summary[v]["delta_vs_clean"] for v in variants],
                "y_latency_s": [summary[v]["avg_latency_s"] for v in variants],
                "y_parse_rate": [summary[v]["parse_rate"] for v in variants],
            },
            "meta": {
                "balance_gold_letters": self.balance_gold_letters,
                "add_labels": self.add_labels,
                "max_new_tokens": self.max_new_tokens,
            },
        }
        out_path = self.config.results_dir / f"{self.test_type}.json"
        save_results(out_path, payload)
        logger.info(f"[InvarianceSweep] Results saved -> {out_path}")
