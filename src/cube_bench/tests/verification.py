# ==================================
# file: cube_bench/tests/verification.py
# ==================================
from __future__ import annotations
import logging
import random
import re
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Tuple, Optional, List

from ..core import BaseTest
from ..io import save_results
from cube_bench.sim.cube_simulator import VirtualCube

logger = logging.getLogger(__name__)

class VerificationTest(BaseTest):
    """Cross-modal Yes/No consistency using VirtualCube (no datasets)."""

    YES_NO_RE = re.compile(r"Answer:\s*(Yes|No)\b", re.IGNORECASE)

    # Moves that are guaranteed to change the Front face (avoid B/B'/B2 only).
    _FRONT_AFFECTING = (
        "F", "F'", "F2",
        "U", "U'", "U2",
        "D", "D'", "D2",
        "L", "L'", "L2",
        "R", "R'", "R2",
    )

    def __init__(self, assistant, config, n_moves: int = 3, verbose: bool = False):
        """
        n_moves: scramble depth for generating diverse states (unrelated to Yes/No label).
        """
        super().__init__(assistant, config, n_moves, verbose)
        self.test_type = "verification"
        self._sys_rng = random.SystemRandom()

    def _eval(self, response: str, expected: str):
        m = self.YES_NO_RE.search(response or "")
        if not m:
            logger.warning("Could not parse model's response.")
            return 0, None
        pred = m.group(1)
        return int(pred.lower() == expected.lower()), pred

    def _front_text(self, cube: VirtualCube) -> str:
        """Return a compact textual view for the front face; fall back to full text."""
        # Prefer a structured API if your VirtualCube exposes it; otherwise str(cube).
        # If you later add `cube.face_to_text('F')`, replace this logic.
        try:
            return cube.front_face()
        except Exception:
            try:
                return cube.observe("text")
            except Exception:
                return "<<<unavailable>>>"

    def _build_sample(self, idx: int) -> Dict:
        """
        Generate a single verification sample:
          - textual description derived from cube T
          - image rendered from cube I
          - expected label: 'Yes' if I==T, else 'No'
        """
        # Base cube for text
        text_cube = VirtualCube()
        text_cube.scramble(random_seed=idx, n_moves=self.n_moves)
        front_text = self._front_text(text_cube)

        # 50/50: matched vs mismatched image
        matched = (idx % 2 == 0)
        if matched:
            img_cube = text_cube
            expected = "Yes"
            mv = None
        else:
            img_cube = text_cube.clone()
            # Apply one move guaranteed to change the Front face
            mv = self._sys_rng.choice(self._FRONT_AFFECTING)
            img_cube.apply(mv)
            expected = "No"

        img = img_cube.to_image()  # in-memory image object (like your other tests)

        return {
            "index": idx,
            "front_text": front_text,
            "image": img,
            "expected": expected,
            "mismatch_move": mv,
        }

    def run(self, num_samples: int) -> Tuple[List[int], float]:
        sys_prompt, user_tpl = self.prompts.get("verification")

        accuracies: List[int] = []

        # --- (2) Anti-gaming counters ---
        parsed = 0
        yes_preds = 0
        tp = tn = fp = fn = 0  # confusion matrix on parsed only

        for i in tqdm(range(num_samples), desc="Verification Test"):
            sample = self._build_sample(i)
            user_prompt = user_tpl.format(front_face=sample["front_text"])

            resp = self.assistant.generate(
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                image=sample["image"],
                max_new_tokens=2**14,
                temperature=0.0,
                top_p=1.0,
            )

            ok, pred = self._eval(resp, sample["expected"])
            accuracies.append(ok)

            # Update anti-gaming metrics
            if pred is not None:
                parsed += 1
                if pred.lower() == "yes":
                    yes_preds += 1

                exp_yes = (sample["expected"].lower() == "yes")
                pred_yes = (pred.lower() == "yes")

                if exp_yes and pred_yes:
                    tp += 1
                elif exp_yes and not pred_yes:
                    fn += 1
                elif (not exp_yes) and (not pred_yes):
                    tn += 1
                else:  # not exp_yes and pred_yes
                    fp += 1

            if self.verbose:
                logger.info(
                    f"Sample {sample['index']}, Expected: {sample['expected']}, "
                    f"Model prediction: {pred}"
                )

        total = num_samples if num_samples else 1
        avg_acc = (sum(accuracies) / total) if accuracies else 0.0

        # Rates over parsed predictions
        parse_rate = parsed / total
        yes_rate = (yes_preds / parsed) if parsed else 0.0

        # Balanced accuracy over parsed predictions
        pos = tp + fn
        neg = tn + fp
        tpr = (tp / pos) if pos else 0.0
        tnr = (tn / neg) if neg else 0.0
        bal_acc = 0.5 * (tpr + tnr) if (pos or neg) else 0.0

        logger.info(
            "Verification metrics: acc=%.3f, bal_acc=%.3f, parse_rate=%.3f, yes_rate=%.3f, "
            "TP=%d TN=%d FP=%d FN=%d, unparsed=%d",
            avg_acc, bal_acc, parse_rate, yes_rate, tp, tn, fp, fn, total - parsed
        )

        # --- (6) Richer results payload ---
        save_results(
            self.config.results_dir / "verification.json",
            {
                "model_name": self.assistant.get_name(),
                "test_type": "verification",
                "timestamp": datetime.now().isoformat(),
                "average_accuracy": avg_acc,
                "num_samples": num_samples,
                "metrics": {
                    "accuracy": avg_acc,
                    "balanced_accuracy": bal_acc,
                    "parse_rate": parse_rate,
                    "parse_violation": 1.0 - parse_rate,
                    "yes_rate": yes_rate,
                    "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                    "unparsed": total - parsed,
                    "support": {"pos": pos, "neg": neg},
                },
                "meta": {
                    "generator": "VirtualCube",
                    "scramble_depth": self.n_moves,
                    "front_affecting_mismatch": True,
                },
            },
        )

        return accuracies, avg_acc
