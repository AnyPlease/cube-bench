# =================================
# file: cube_bench/tests/solve_moves.py
# =================================
from __future__ import annotations
import logging
import random
import re
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Tuple, Any, List, Optional
from pathlib import Path
from copy import deepcopy

from ..core import BaseTest
from ..io import save_results

logger = logging.getLogger(__name__)

# VirtualCube import (same style as your step_by_step)
from cube_bench.sim.cube_simulator import VirtualCube

# Optional PromptFactory (matches your step_by_step usage), with safe fallback
try:
    from cube_bench.prompts.prompt_factory import PromptFactory
    _HAS_FACTORY = True
except Exception:
    _HAS_FACTORY = False

# --- Robust parsing (matches StepByStepTest patterns) ---
PRIMARY_RE = re.compile(r"<ANSWER>\s*([ABCD])\s*</ANSWER>", re.IGNORECASE)
ALT_1_RE   = re.compile(r"\bANSWER\s*[:=]\s*([ABCD])\b", re.IGNORECASE)   # ANSWER: C
ALT_2_RE   = re.compile(r"<([ABCD])>", re.IGNORECASE)                     # <C>
MOVE_RE = re.compile(r"<ANSWER>\s*([URFDLB](?:2|')?)\s*</ANSWER>", re.IGNORECASE)

class SolveMovesTest(BaseTest):
    """
    Dynamic MCQ using VirtualCube: generate states on-the-fly (1 move from solved).
    Prompt modalities: mixed | image | text (PromptFactory if available; otherwise prompts.json).
    """
    def __init__(self, assistant, config, prompt_type: str = "mixed", n_moves: int = 1, verbose: bool = False):
        super().__init__(assistant, config, n_moves, verbose)
        self.test_type = "prediction"
        self.prompt_type = prompt_type
        self.n_moves = n_moves
        self.verbose = verbose

    # ---------- helpers ----------
    def _parse_answer(self, text: str, options: Dict[str, str] | None = None) -> Optional[str]:
        m = PRIMARY_RE.search(text) or ALT_1_RE.search(text) or ALT_2_RE.search(text)
        if m:
            return m.group(1).upper()
        # Fallback: model returned a move like R', U2, etc.
        if options:
            m2 = MOVE_RE.search(text)
            if m2:
                move = m2.group(1).upper()
                # map move -> letter using current options
                for letter, mv in options.items():
                    if mv.upper() == move:
                        return letter
        return None

    def _gen_opts(self, correct_move: str, rng: random.Random, force_letter: str | None = None) -> Tuple[Dict[str, str], str]:

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

    def _observe_text(self, cube: VirtualCube) -> str:
        # Keep parity with step_by_step: prefer __str__, fallback to observe("text") if you expose it
        try:
            return str(cube) # VirtualCube has __str__() method
        except Exception:
            try:
                return cube.observe("text")
            except Exception:
                return ""

    def _teacher_first_move(self, scramble_obj) -> str:
        """
        Derive the immediate solving move from the scramble, mirroring your step_by_step logic:
        teacher path = reverse(scramble).
        For n_moves == 1, that's just the single inverse move.
        """
        try:
            s_copy = deepcopy(scramble_obj)
            path = str(s_copy.reverse()).split()
            return path[0] if path else None
        except Exception:
            # conservative fallback: compute from the last scramble token
            s = str(scramble_obj).split()
            if not s:
                return self._sys_rng.choice(VirtualCube.AVAILABLE_MOVES)
            last = s[-1]
            if last.endswith("2"):
                return last 
            if last.endswith("'"):
                return last[:-1]
            return last + "'"

    def _build_sample(self, idx: int) -> Dict[str, Any]:
        if self.n_moves != 1:
            raise ValueError(
                "SolveMovesTest currently supports n_moves==1. "
                "For >1, integrate a teacher solver (e.g., Kociemba) to get the gold first move."
            )

        cube = VirtualCube()
        scramble = cube.scramble(random_seed=idx, n_moves=1)  # consistent with step_by_step signature
        teacher_move = self._teacher_first_move(scramble)     # e.g., "F'" for scramble "F"

        stable_rng = random.Random(idx)
        forced = "ABCD"[idx % 4]
        options, gold_letter = self._gen_opts(teacher_move, stable_rng, force_letter=forced)

        sample = {
            "id": idx,
            "image": cube.to_image() if self.prompt_type in ("image", "mixed") else None,  # in-memory image
            "text_state": self._observe_text(cube),
            "options": options,                 # {A,B,C,D} -> move
            "correct_letter": gold_letter,      # "A"/"B"/"C"/"D"
            "correct_move": teacher_move,       # move string
            "scramble": str(scramble),          # for logs
        }
        return sample

    # ---------- prompts ----------
    def _build_prompts(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        kwargs = {
            "move_A": sample["options"]["A"],
            "move_B": sample["options"]["B"],
            "move_C": sample["options"]["C"],
            "move_D": sample["options"]["D"],
            "textual_representation": sample["text_state"] if self.prompt_type != "image" else "",
            "metric": "HTM (Half-Turn Metric)",  # harmless hint; same as step_by_step
        }

        if _HAS_FACTORY:
            # Align with your step_by_step style
            sys_prompt, user_prompt = PromptFactory.get("prediction", prompt_type=self.prompt_type, **kwargs)
            return user_prompt, sys_prompt

        # Fallback to prompts.json that BaseTest loads into self.prompts
        if self.prompt_type not in self.prompts.get(self.test_type, {}):
            raise ValueError(f"Prompt type '{self.prompt_type}' not found under '{self.test_type}' in prompts.json")
        sys_prompt = self.prompts[self.test_type][self.prompt_type]["sys"]
        user_tpl  = self.prompts[self.test_type][self.prompt_type]["user"]
        return user_tpl.format(**kwargs), sys_prompt

    # ---------- main ----------
    def run(self, num_samples: int) -> Tuple[List[Tuple[str, int]], List[int]]:
        """
        Returns:
          wrong_pairs: List[(model_answer_letter or None, sample_id)]
          acc_bits:    List[int] 1/0 per item
        This shape matches your reflection pipeline.
        """
        wrong_pairs: List[Tuple[str, int]] = []
        acc_bits: List[int] = []
        parse: int = 0

        for i in tqdm(range(num_samples), desc="Solve move test"):
            rng = random.Random(i) 

            sample = self._build_sample(i)
            user_prompt, sys_prompt = self._build_prompts(sample)

            resp = self.assistant.generate(
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                image=sample["image"],         # in-memory image if any
                max_new_tokens=(2**16),
                do_sample=False,
                temperature=0.1,
                top_p=1.0,
            )

            pred_letter = self._parse_answer(resp or "", sample['options'])
            ok = int(pred_letter == sample["correct_letter"])
            acc_bits.append(ok)

            if pred_letter:
                parse += 1

            logger.info(
                f"\nModel Predicted: {pred_letter}"
                f"\nOk : {ok}"
            )

            if not ok:
                wrong_pairs.append((pred_letter if pred_letter else None, sample["id"]))
                if self.verbose:
                    logger.info(
                        "Wrong #%d: pred=%s gold=%s options=%s scramble=%s",
                        sample["id"], pred_letter, sample["correct_letter"],
                        sample["options"], sample["scramble"]
                    )

            if self.verbose and (i + 1) % 10 == 0:
                logger.info("Progress: %d/%d (acc=%.3f)", i + 1, num_samples, sum(acc_bits) / len(acc_bits))

        avg_acc = (sum(acc_bits) / len(acc_bits)) if acc_bits else 0.0
        logger.info("SolveMoves avg accuracy (%s): %.3f", self.prompt_type, avg_acc)
        logger.info(f"Parsed rate: {(parse / num_samples) * 100}")

        # Persist a compact run record
        save_results(
            Path(self.config.results_dir) / f"solve_moves_{self.prompt_type}.json",
            {
                "model_name": self.assistant.get_name(),
                "test_type": self.test_type,
                "prompt_type": self.prompt_type,
                "timestamp": datetime.now().isoformat(),
                "average_accuracy": avg_acc,
                "num_samples": num_samples,
                "meta": {
                    "n_moves": self.n_moves,
                    "generator": "VirtualCube",
                    "prompt_source": "PromptFactory" if _HAS_FACTORY else "prompts.json",
                },
            },
        )

        return wrong_pairs, acc_bits
