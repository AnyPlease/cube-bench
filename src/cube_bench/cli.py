# cube_bench/cli.py
import argparse
import logging
from pathlib import Path

from .config import Config
from .orchestrator import TestOrchestrator

def setup_logging(level: str = "INFO", log_file: Path | None = None):
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
    )

def main():
    parser = argparse.ArgumentParser(description="Run Rubik's Cube MLLM evaluations")

    parser.add_argument("--model", required=True, type=str,
        choices=[
            "qwen2.5-7b","qwen2.5-32b","gemma3","llama4",
            "gemini2.5-pro","gemini2.5-flash","internvl3_5-38b","glm4.5v","qwen3-vl-thinking"
        ])
    parser.add_argument("--test", required=True, type=str,
        choices=[
            "prediction","verification","reconstruction","step-by-step",
            "learning-curve","move-effect","invariance-sweep",
            "persistence-blackout","reflection"  # NEW
        ])
    parser.add_argument("--difficulty", default="easy", choices=["easy","medium","hard","all"])
    parser.add_argument("--prompt", default="image", choices=["mixed","image","text"])
    parser.add_argument("--samples","-n", type=int, default=100)
    parser.add_argument("--moves","-m", type=int, default=3)
    parser.add_argument("--backend", default="hf", choices=["hf","vllm"])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--dataset", type=Path, default=Path("test-data/rubiks_dataset/rubiks_dataset.json"))
    parser.add_argument("--prompts", type=Path, default=Path("scripts/prompts/prompts.json"))

    # NEW: reflection-specific knobs
    parser.add_argument("--reflection-type", default="Redacted",
                        choices=["Unguided","Redacted","Unredacted"],
                        help="Reflection bundle to use from reflection prompts JSON.")
    parser.add_argument("--reflection-prompts", type=Path, default=Path("/home/dana0001/wj84_scratch/dana0001/rubiks-cube-thesis/scripts/cube_bench/prompts/reflection.json"),
                        help="Path to reflection prompt bundles JSON.")
    parser.add_argument("--max-reflections", type=int, default=None,
                        help="Cap number of wrong items to reflect on (cost control).")

    # logging
    parser.add_argument("--log", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    setup_logging(args.log, args.log_file)

    config = Config(dataset_path=args.dataset, prompts_path=args.prompts, results_dir=args.results)

    orch = None
    try:
        orch = TestOrchestrator(model_name=args.model, config=config, backend=args.backend)
        result = orch.run_test(
            test_type=args.test,
            difficulty=args.difficulty,
            prompt_type=args.prompt,
            num_samples=args.samples,
            n_moves=args.moves,
            verbose=(args.log == "DEBUG"),
            # pass through reflection knobs (others ignore them)
            reflection_type=args.reflection_type,
            reflection_prompts=args.reflection_prompts,
            max_reflections=args.max_reflections,
        )
        if not args.quiet:
            if args.test == "reflection":
                # result is a summary dict
                print(f"✅ Reflection done → {result}")
            else:
                print(f"✅ Finished: test={args.test} n={args.samples} moves={args.moves} → results in {config.results_dir}")
    except Exception:
        logging.exception("An error occurred during evaluation")
        raise
    finally:
        if orch:
            pass
            # orch.cleanup()   # turn cleanup back on


if __name__ == "__main__":
    main()
