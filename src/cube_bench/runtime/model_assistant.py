# =================================================================================================
#  Modular Model-Strategy Framework
#  ---------------------------------------------------------------------------------
#  - One registry line → new model
#  - Shared prompt builder & utilities
#  - HuggingFace (HF) or vLLM engines
#  - Vision-ready (PIL or Path), efficient, and multi-GPU friendly
# =================================================================================================

from __future__ import annotations

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from cube_bench.sim.cube_simulator import VirtualCube

import torch
torch.set_float32_matmul_precision("high")

from PIL import Image

logger = logging.getLogger("assistant")
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

# ---- Optional imports (keep file importable without deps)
try:
    from transformers import (
        AutoProcessor,
        # AutoTokenizer,
        AutoModelForCausalLM,  # generic
        # AutoConfig,
    )
except Exception as e:  # pragma: no cover
    AutoProcessor = AutoTokenizer = AutoModelForCausalLM = AutoConfig = AutoModel = None  # type: ignore
    logger.info(f"[transformers] Import optional: {e}")

try:
    from vllm import LLM, SamplingParams
except Exception as e:  # pragma: no cover
    LLM = SamplingParams = None  # type: ignore
    logger.info(f"[vLLM] Import: {e}")

# Keep vLLM quiet but preserve our INFO logs
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("vllm.core").setLevel(logging.WARNING)

# --------------------------------------------------------------------------------------------------
# 1) Dataclasses & small utilities
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str                 # local path, HF repo, or API name
    strategy_hf: Optional[Type["ModelStrategy"]] = None
    strategy_vllm: Optional[Type["ModelStrategy"]] = None
    dtype: torch.dtype = torch.bfloat16
    supports_image: bool = True

@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = True

def _as_pil(img: Optional[Union[Image.Image, Path, str]]) -> Optional[Image.Image]:
    """Normalize a single image input to a RGB PIL.Image (or None)."""
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    # Accept Path or str
    return Image.open(str(img)).convert("RGB")


def _to_device(batch: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Any:
    # Supports plain dicts or HF BatchFeature (which has .to)
    try:
        return batch.to(device)
    except Exception:
        pass
    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                if dtype and v.dtype.is_floating_point:
                    out[k] = v.to(device, dtype=dtype, non_blocking=True)
                else:
                    out[k] = v.to(device, non_blocking=True)
            else:
                out[k] = v
        return out
    return batch

# --------------------------------------------------------------------------------------------------
# 2) Prompt builder (single source of truth)
# --------------------------------------------------------------------------------------------------

class PromptBuilder:
    def __init__(self, processor: "AutoProcessor") -> None:
        self.processor = processor

    def build_hf_inputs(
        self,
        user_prompt: str,
        system_prompt: str,
        image: Optional[Union[Image.Image, Path]] = None,
        reference: str = "",
    ) -> Dict[str, Any]:
        pil = _as_pil(image)

        # Compose multimodal-style messages
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": ([{"type": "image", "image": pil}] if pil is not None else []) +
                                    [{"type": "text", "text": user_prompt}]},
        ]
        if reference:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": reference}]})

        # Prefer true multimodal chat templates (works if you loaded AutoProcessor for a *-hf model)
        try:
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError as e:
            # Template is text-only → flatten “parts” and process images separately
            if "concatenate str (not \"list\") to str" not in str(e):
                raise

            tok = getattr(self.processor, "tokenizer", self.processor)  # tokenizer for text
            image_processor = getattr(self, "image_processor", getattr(self.processor, "image_processor", None))

            # 1) Flatten content -> string + image placeholders
            image_token = getattr(tok, "image_token", "<image>")
            flat_msgs, images = [], []
            for m in messages:
                parts = []
                for p in m.get("content", []):
                    t = p.get("type")
                    if t == "image":
                        parts.append(image_token)
                        images.append(p.get("image") or p.get("url") or p.get("path"))
                    elif t == "text":
                        parts.append(p.get("text", ""))
                flat_msgs.append({"role": m["role"], "content": "".join(parts)})

            chat_text = tok.apply_chat_template(flat_msgs, add_generation_prompt=True, tokenize=False)

            # 2) ALWAYS tokenize text with the tokenizer (no images kwarg here!)
            text_inputs = tok(chat_text, return_tensors="pt", padding=True)

            # 3) If there’s an image, process it with an image processor and merge
            if images:
                if image_processor is None:
                    raise RuntimeError(
                        "Your self.processor is a tokenizer (no images=). "
                        "Load a multimodal AutoProcessor for the model or attach self.image_processor."
                    )
                vision_inputs = image_processor(images=images, return_tensors="pt")
                text_inputs.update(vision_inputs)

            return text_inputs


    def build_vllm_request(self, user_prompt: str, system_prompt: str,
                       image: Optional[Union[Image.Image, Path]] = None) -> Dict[str, Any]:
        pil = _as_pil(image)
        # Use HF multimodal chat template to insert the placeholder token(s)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",
            "content": ([{"type": "image"}] if pil is not None else []) + [{"type": "text", "text": user_prompt}]},
        ]
        prompt_txt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        req = {"prompt": prompt_txt}
        if pil is not None:
            req["multi_modal_data"] = {"image": pil}  # PIL image directly (no temp file)
        return req


# --------------------------------------------------------------------------------------------------
# 3) Strategy base class
# --------------------------------------------------------------------------------------------------

class ModelStrategy(ABC):
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.processor: Optional["AutoProcessor"] = None
        self.model: Any = None     # HF model OR vLLM engine
        self.prompt_builder: Optional[PromptBuilder] = None
        self._device_for_inputs: str = "cpu"   # where to stage inputs

    # ---- Lifecycle hooks
    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        image: Optional[Union[Image.Image, Path]],
        gen_cfg: GenerationConfig,
        reference: str = "",
    ) -> str:
        pass

    # ---- Helpers
    def _ensure_processor(self) -> None:
        assert AutoProcessor is not None, "transformers not installed"
        # InternVL needs its remote (non-fast) tokenizer with image tokens.
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.spec.path, 
                trust_remote_code=True, 
                use_fast=True
            )

        except Exception as e:
            logger.warning(f"Could not load fast tokenizer for {self.spec.name}: {e}")
            logger.warning(f"Falling back to slow tokenizer.")
            
            self.processor = AutoProcessor.from_pretrained(
                self.spec.path, 
                trust_remote_code=True, 
                # use_fast=False
            )

        self.prompt_builder = PromptBuilder(self.processor)

    def cleanup(self) -> None:
        logger.debug("[%s] cleanup", self.spec.name)
        try:
            del self.model, self.processor, self.prompt_builder
        except Exception:
            pass
        self.model = self.processor = self.prompt_builder = None  # type: ignore
        torch.cuda.empty_cache()

# --------------------------------------------------------------------------------------------------
# 4) HuggingFace base strategy (shared generation & optional batching)
# --------------------------------------------------------------------------------------------------

class HuggingFaceStrategy(ModelStrategy):
    def load(self) -> None:
        self._ensure_processor()
        self.model = self._load_model_instance()
        
        # If model is sharded we won't have single .device; inputs go to cuda:0 if available
        if torch.cuda.is_available():
            self._device_for_inputs = "cuda:0"
        elif torch.backends.mps.is_available():
            self._device_for_inputs = "mps"
        else:
            self._device_for_inputs = "cpu"

        logger.info("[%s] HF model ready (inputs on %s)", self.spec.name, self._device_for_inputs)

    @abstractmethod
    def _load_model_instance(self):
        "To be implemented in model class individually."

    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        image: Optional[Union[Image.Image, Path]],
        gen_cfg: GenerationConfig,
        reference: str = "",
    ) -> str:

        assert self.prompt_builder is not None and self.processor is not None
        inputs = self.prompt_builder.build_hf_inputs(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            image=image if self.spec.supports_image else None,
            reference=reference,
        )
        
        # 1) Move to the right device (do NOT cast yet; HF BatchFeature.to() ignores dtype anyway)
        inputs = _to_device(inputs, torch.device(self._device_for_inputs), dtype=None)

        # 2) Align vision input dtype with the model’s vision tower (prevents float vs bf16 conv2d crash)
        vision_dtype = None
        vt = getattr(self.model, "vision_tower", None)
        try:
            vision_dtype = next(vt.parameters()).dtype if vt is not None else next(self.model.parameters()).dtype
        except StopIteration:
            vision_dtype = next(self.model.parameters()).dtype
        if "pixel_values" in inputs and torch.is_tensor(inputs["pixel_values"]):
            pv = inputs["pixel_values"]
            if pv.dtype != vision_dtype:
                inputs["pixel_values"] = pv.to(dtype=vision_dtype, non_blocking=True)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                do_sample=gen_cfg.do_sample,
                top_p=gen_cfg.top_p,
                max_new_tokens=gen_cfg.max_new_tokens,
                temperature = gen_cfg.temperature
            )

        # Decode only the generated continuation
        gen_ids = out[:, input_len:]
        text = self.processor.decode(gen_ids[0], skip_special_tokens=True)
        return text

    # simple batch API (strings, same system prompt & no images for now)
    def generate_batch(self, prompts: Iterable[str], system_prompt: str = "You are a helpful assistant.", gen_cfg: Optional[GenerationConfig] = None,) -> List[str]:

        assert self.prompt_builder is not None and self.processor is not None
        gen_cfg = gen_cfg or GenerationConfig()
        msgs = [
            [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": p}]},
            ]
            for p in prompts
        ]
        # pack via processor
        enc = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_tensors="pt", padding=True, return_dict=True
        )
        enc = _to_device(enc, torch.device(self._device_for_inputs), dtype=self.spec.dtype)
        input_lens = (enc["input_ids"] != self.processor.tokenizer.pad_token_id).sum(-1)

        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                do_sample=gen_cfg.do_sample,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                max_new_tokens=gen_cfg.max_new_tokens,
            )

        results: List[str] = []
        for i in range(out.shape[0]):
            gen_ids = out[i, input_lens[i] :]
            results.append(self.processor.decode(gen_ids, skip_special_tokens=True))
        return results

# ---- concrete HF models

class GemmaStrategy(HuggingFaceStrategy):
    def _load_model_instance(self):
        from transformers import Gemma3ForConditionalGeneration  # type: ignore
        return Gemma3ForConditionalGeneration.from_pretrained(
            self.spec.path, dtype=self.spec.dtype, device_map="auto"
        ).eval()

class LlamaStrategy(HuggingFaceStrategy):
    def _load_model_instance(self):
        from transformers import Llama4ForConditionalGeneration  # type: ignore
        return Llama4ForConditionalGeneration.from_pretrained(
            self.spec.path, dtype=self.spec.dtype, device_map="auto"
        ).eval()

class QwenVLStrategy(HuggingFaceStrategy):
    def _load_model_instance(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration # type: ignore
        if self.spec.name == "qwen3-vl-thinking":
            return Qwen3VLMoeForConditionalGeneration.from_pretrained(
            self.spec.path, dtype="auto", device_map="auto"
        ).eval()

        else:
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.spec.path, dtype="auto", device_map="auto"
        ).eval()
    
    def generate(self,user_prompt: str,system_prompt: str, image: Optional[Union[Image.Image, Path]],gen_cfg: GenerationConfig,reference: str = "") -> str:
        model = self.model
        processor = AutoProcessor.from_pretrained(self.spec.path)

        # Build messages like the card example (user role only)
        content = []
        if image is not None:
            if isinstance(image, (str, Path)):
                content.append({"type": "image", "image": str(image)})
            else:  # PIL.Image.Image
                # ensure RGB just in case
                pil = image if image.mode == "RGB" else image.convert("RGB")
                content.append({"type": "image", "image": pil})
        content.append({"type": "text", "text": user_prompt})

        messages = [{"role": "user", "content": content}]

        # Preparation for inference (same args as the card)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        device = getattr(self.model, "device", torch.device("cuda"))
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        # Inference: Generation of the output (only max_new_tokens like the card)
        max_new = int(getattr(gen_cfg, "max_new_tokens", 128))
        generated_ids = model.generate(**inputs, max_new_tokens=max_new)

        # Trim prompt tokens from the output (same as the card)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode (same flags as the card)
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0] if output_text else ""


# ---- InternVL3 (HF) ----

class InternVL3_5Strategy(HuggingFaceStrategy):
    """
    InternVL3-78B-Instruct loader & generator.
    Uses custom device_map splitting for multi-GPU; falls back to device_map="auto".
    """

    def _load_model_instance(self):
        return AutoModelForImageTextToText.from_pretrained(
            self.spec.path,
            dtype=self.spec.dtype,
            trust_remote_code=True,
            device_map="auto").eval()

# ---- GLM-4.5V (HF) ----

class GLM45VStrategy(HuggingFaceStrategy):
    """
    GLM-4.5V (MoE) loader & generator.
    Uses the official Transformers class Glm4vMoeForConditionalGeneration.
    """
    def _load_model_instance(self):
        from transformers import Glm4vMoeForConditionalGeneration, AutoModelForConditionalGeneration
        return AutoModelForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.spec.path,
            dtype="auto",   # bf16 recommended
            device_map="auto",
        ).eval()


# --------------------------------------------------------------------------------------------------
# 5) vLLM local strategy
# --------------------------------------------------------------------------------------------------

class VllmStrategy(ModelStrategy):
    def load(self) -> None:
        assert LLM is not None, "vLLM package not installed"
        self._ensure_processor()

        self.model = LLM(
            model=self.spec.path,
            dtype="auto",
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.80,
            max_model_len=2**16,
        )
        self._device_for_inputs = "cpu"
        logger.info("vLLM model ready.")

    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        image: Optional[Union[Image.Image, Path]],
        gen_cfg: GenerationConfig,
        reference: str = "",
    ) -> str:

        assert self.prompt_builder is not None, "vLLM prompt builder missing"

        print(f"Temp={gen_cfg.temperature}")

        req = self.prompt_builder.build_vllm_request(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            image=image if self.spec.supports_image else None,
        )
        # ensure we hand vLLM a PIL image, not a temp path
        if "multi_modal_data" in req and isinstance(req["multi_modal_data"].get("image"), str):
            req["multi_modal_data"]["image"] = _as_pil(image)

        params = SamplingParams(
            max_tokens=gen_cfg.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            stop_token_ids=[self.processor.tokenizer.eos_token_id],
        )
        out = self.model.generate([req], sampling_params=params, use_tqdm=False)[0]
        return out.outputs[0].text.strip()

# --------------------------------------------------------------------------------------------------
# 6) Remote Gemini (API) strategy – optional, off GPU
# --------------------------------------------------------------------------------------------------

class GeminiStrategy(ModelStrategy):
    def load(self) -> None:
        logger.info("[gemini] remote strategy initialized - will use API key from env")

    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        image: Optional[Union[Image.Image, Path]],
        gen_cfg: GenerationConfig,
        reference: str = "",
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generates a response and returns it along with token usage statistics.
        Returns:
            A tuple containing:
            - The response text (str).
            - A dictionary with token counts (Dict[str, int]).
        """
        import google.generativeai as genai

        model = genai.GenerativeModel(
            model_name=self.spec.path,
            system_instruction=system_prompt,
            generation_config={"max_output_tokens": gen_cfg.max_new_tokens},
        )
        contents: List[Any] = [user_prompt]
        if image is not None:
            contents.append(_as_pil(image))

        # 1. (Optional) Estimate input tokens BEFORE the call
        input_token_estimate = model.count_tokens(contents).total_tokens
        logger.info(f"[gemini] Estimated input tokens: {input_token_estimate}")

        # 2. Make the API call
        resp = model.generate_content(contents)
        response_text = ""
        token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        # 3. Get exact token counts AFTER the call from usage metadata
        try:
            # It's good practice to check if a response was blocked
            if resp.candidates:
                response_text = resp.text
            else:
                response_text = "Response was blocked." # Or handle as needed

            # The usage_metadata is available even if the response is blocked
            if hasattr(resp, 'usage_metadata'):
                token_usage["input_tokens"] = resp.usage_metadata.prompt_token_count
                token_usage["output_tokens"] = resp.usage_metadata.candidates_token_count
                token_usage["total_tokens"] = resp.usage_metadata.total_token_count
        
            logger.info(
                f"\n[gemini] Estimated thinking tokens: {token_usage['total_tokens'] - (token_usage['input_tokens'] + token_usage['output_tokens'])}"
                f"\n[gemini] Estimated output tokens: {token_usage['output_tokens']}"
            )

        except Exception as e:
            logger.warning(f"[gemini] Could not retrieve usage metadata from response. Error: {e}")
            # The response text might still be available in some error cases
            if hasattr(resp, 'text'):
                response_text = resp.text

        # print(f"Token usuage: {token_usage}")
        # print(f"Model's full response: {resp}")

        return response_text

# --------------------------------------------------------------------------------------------------
# 7) Registry + factory
# --------------------------------------------------------------------------------------------------

def get_strategy(name: str, engine: str, REGISTRY: Dict[str, ModelSpec]) -> ModelStrategy:
    try:
        spec = REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(REGISTRY)}") from exc

    if engine == "vllm":
        if spec.strategy_vllm is None:
            raise ValueError(f"Model '{name}' does not support vLLM backend")
        strat = spec.strategy_vllm(spec)
    elif engine == "hf":
        if spec.strategy_hf is None:
            raise ValueError(f"Model '{name}' does not support HuggingFace backend")
        strat = spec.strategy_hf(spec)
    else:
        raise ValueError(f"Unknown engine '{engine}', expected 'hf' or 'vllm'")

    strat.load()
    return strat

# --------------------------------------------------------------------------------------------------
# 8) Assistant façade
# --------------------------------------------------------------------------------------------------

class ModelAssistant:
    MODEL_REGISTRY: Dict[str, ModelSpec] = {
        "gemma3": ModelSpec("gemma3", "google/gemma-3-27b-it", GemmaStrategy, VllmStrategy),
        "gemma3-4b": ModelSpec("gemma3-4b", "google/gemma-3-4b-it", GemmaStrategy, VllmStrategy),
        "llama4": ModelSpec("llama4", "meta-llama/Llama-4-Scout-17B-16E-Instruct", LlamaStrategy, VllmStrategy),
        "qwen2.5-7b": ModelSpec("qwen2.5-7b", "Qwen/Qwen2.5-VL-7B-Instruct", QwenVLStrategy, VllmStrategy),
        "qwen2.5-32b": ModelSpec("qwen2.5-32b", "Qwen/Qwen2.5-VL-32B-Instruct", QwenVLStrategy, VllmStrategy),
        "qwen3-vl-thinking": ModelSpec("qwen3-vl-thinking", "Qwen/Qwen3-VL-30B-A3B-Thinking", QwenVLStrategy, VllmStrategy),
        "gemini2.5-pro": ModelSpec("gemini-pro", "gemini-2.5-pro", GeminiStrategy, GeminiStrategy, supports_image=True),
        # InternVL3 (HF only for now)
        "internvl3_5-38b": ModelSpec(
            "internvl3_5-38b",
            "OpenGVLab/InternVL3_5-38B",
            InternVL3_5Strategy,
            None,
            dtype=torch.bfloat16,
            supports_image=True,
        ),
        "glm4.5v": ModelSpec(
            "glm4.5v",
            "Zai/GLM-4.5V",  # or "zai-org/GLM-4.5V"
            GLM45VStrategy,
            VllmStrategy,                 # vLLM works if your build has GLM-4.5V recipe; fall back to 'hf' if not
            dtype=torch.bfloat16,
            supports_image=True,
        ),
    }

    def __init__(self, backend: str, engine: str) -> None:
        logger.info("Initializing assistant backend=%s engine=%s", backend, engine)
        self.backend = backend
        self.engine = engine
        if backend not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{backend}'")
        self.strategy = get_strategy(backend, self.engine, self.MODEL_REGISTRY)

    def get_name(self) -> str:
        return self.MODEL_REGISTRY[self.backend].name

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 128,
        image: Optional[Union[Image.Image, Path]] = None,
        reference: str = "",
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> str:

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        return self.strategy.generate(
            user_prompt, system_prompt, image, gen_cfg, reference
        )

    def cleanup(self) -> None:
        self.strategy.cleanup()

# --------------------------------------------------------------------------------------------------
# 9) Prompt-file helper (unchanged)
# --------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_prompts() -> Dict[str, Any]:
    path = Path("prompts.json")
    if not path.exists():
        raise FileNotFoundError("prompts.json file not found")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# --------------------------------------------------------------------------------------------------
# 10) Tiny demo
# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3", help="Backend model name")
    parser.add_argument("--engine", default="hf", choices=["hf", "vllm"], help="Execution engine")
    args = parser.parse_args()

    backend = args.model
    engine  = args.engine

    cube = VirtualCube()

    assistant = ModelAssistant(backend, engine=engine)
    try:
        print(f"\n=========== Response (backend: {backend}, engine: {engine}) ===========")
        out = assistant.generate(
            system_prompt="You are a good assistant.",
            user_prompt="Describe me the image you see. What are the colors on the front face?",
            image=cube.to_image(),
            max_new_tokens=2**12,
        )
        print(out)
    finally:
        assistant.cleanup()
