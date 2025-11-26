# =============================
# file: cube_bench/orchestrator.py
# =============================
from __future__ import annotations
import gc
import logging
from typing import Dict, Type

from .config import Config
from .tests import (
    SolveMovesTest,
    VerificationTest,
    ReconstructionTest,
    StepByStepTest,
    LearningCurveTest,
    MoveEffectTest,
    InvarianceSweepTest,
    ReflectionTest
)
from cube_bench.runtime.model_assistant import ModelAssistant

logger = logging.getLogger(__name__)


class TestOrchestrator:
    def __init__(self, model_name: str, config: Config, backend: str = "hf"):
        logger.info("Initializing assistant backend=%s engine=%s", model_name, backend)
        self.assistant = ModelAssistant(model_name, engine=backend)
        self.config = config

        # map CLI names to classes
        self.registry: Dict[str, Type] = {
            "prediction": SolveMovesTest,
            "verification": VerificationTest,
            "reconstruction": ReconstructionTest,
            "step-by-step": StepByStepTest,
            "learning-curve": LearningCurveTest,
            "move-effect": MoveEffectTest,
            "invariance-sweep": InvarianceSweepTest,
            "reflection": ReflectionTest,
        }

    def run_test(
        self, 
        test_type: str, 
        difficulty: str, 
        prompt_type: str = "image", 
        num_samples: int = 1, 
        n_moves: int = 3, 
        verbose: bool = False,
        reflection_type: str = "Redacted",
        reflection_prompts: Path | None = None,
        max_reflections: int | None = None,
    ):
        if test_type not in self.registry:
            raise ValueError(f"Unknown test type '{test_type}'. Available: {list(self.registry)}")

        # construct appropriate test instance and run
        if test_type == "prediction":
            test = SolveMovesTest(self.assistant, self.config, prompt_type, n_moves=n_moves, verbose=verbose)
            return test.run(num_samples)
        
        elif test_type == "reflection":
            rp = reflection_prompts or (self.config.prompts_path.parent / "reflection.json")
            test = ReflectionTest(
                assistant=self.assistant,
                config=self.config,
                reflection_prompts=rp,
                reflection_type=reflection_type,
                prompt_type=prompt_type,
                max_reflections=max_reflections,
                verbose=verbose,
            )
            # Reflection ignores num_samples; returns a summary dict
            return test.run(num_samples)
        elif test_type in ("verification", "reconstruction"):
            test = self.registry[test_type](self.assistant, self.config, n_moves=n_moves, verbose=verbose)
            return test.run(num_samples)
        else:
            test = self.registry[test_type](self.assistant, self.config, n_moves=n_moves, verbose=verbose)  # type: ignore[arg-type]
            return test.run(num_samples)

    def cleanup(self):
        logger.info("Cleaning up assistant resources…")
        if hasattr(self, "assistant") and self.assistant is not None:
            self.assistant.cleanup()
            del self.assistant
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("Assistant resources cleaned up.")
