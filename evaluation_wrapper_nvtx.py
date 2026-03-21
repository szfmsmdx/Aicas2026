"""
NVTX-instrumented wrapper for profiling Qwen3-VL stages and key functions.

This file keeps the original evaluation_wrapper.py untouched.
"""
from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Dict, Iterator

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from PIL import Image
except ImportError:
    class Image:  # pragma: no cover
        pass


class VLMModelNVTX:
    """Profiling-focused wrapper with stage/function NVTX ranges."""

    def __init__(self, model_path: str, device: str = "cuda:0"):
        self._device = device
        self.model_path = model_path
        self._nvtx_enabled = True

        print(f"[VLMModelNVTX] Loading processor from {model_path}...")
        self._processor = AutoProcessor.from_pretrained(model_path)

        print("[VLMModelNVTX] Loading model with FP16...")
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self._model.eval()

        self._install_nvtx_hooks()
        print(f"[VLMModelNVTX] Model loaded successfully on {device}")

    # ------------------------------------------------------------------
    # NVTX utilities
    # ------------------------------------------------------------------
    @contextmanager
    def nvtx_range(self, name: str) -> Iterator[None]:
        enabled = self._nvtx_enabled and torch.cuda.is_available()
        if enabled:
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if enabled:
                torch.cuda.nvtx.range_pop()

    @contextmanager
    def _set_nvtx_enabled(self, enabled: bool) -> Iterator[None]:
        old = self._nvtx_enabled
        self._nvtx_enabled = enabled
        try:
            yield
        finally:
            self._nvtx_enabled = old

    def _is_decode_from_tensor(self, tensor: torch.Tensor | None) -> bool:
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.dim() < 2:
            return False
        return tensor.shape[1] == 1

    def _infer_llm_phase(self, args: tuple, kwargs: dict) -> str:
        # Decoder layer modules usually receive hidden_states as first arg.
        candidate = None
        if args:
            candidate = args[0]
        if isinstance(candidate, torch.Tensor) and candidate.dim() >= 2:
            return "decode" if candidate.shape[1] == 1 else "prefill"

        # Higher-level modules may use named tensors.
        for key in ("hidden_states", "inputs_embeds", "input_ids"):
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor) and value.dim() >= 2:
                return "decode" if value.shape[1] == 1 else "prefill"

        # Conservative fallback: prefill.
        return "prefill"

    def _wrap_module_forward(self, module, label_getter):
        if module is None:
            return
        if getattr(module, "_nvtx_wrapped", False):
            return

        original_forward = module.forward

        @functools.wraps(original_forward)
        def wrapped_forward(*args, **kwargs):
            label = label_getter(args, kwargs)
            with self.nvtx_range(label):
                return original_forward(*args, **kwargs)

        module.forward = wrapped_forward
        module._nvtx_wrapped = True

    def _install_nvtx_hooks(self):
        core_model = getattr(self._model, "model", None)
        if core_model is None:
            return

        visual = getattr(core_model, "visual", None)
        if visual is not None:
            self._wrap_module_forward(
                visual,
                lambda _args, _kwargs: "stage.visual_prefill",
            )

            blocks = getattr(visual, "blocks", None)
            if blocks is not None:
                for idx, block in enumerate(blocks):
                    attn = getattr(block, "attn", None)
                    mlp = getattr(block, "mlp", None)
                    self._wrap_module_forward(
                        attn,
                        lambda _args, _kwargs, i=idx: f"func.vision.block{i}.attn",
                    )
                    self._wrap_module_forward(
                        mlp,
                        lambda _args, _kwargs, i=idx: f"func.vision.block{i}.mlp",
                    )

        language_model = getattr(core_model, "language_model", None)
        if language_model is not None:
            self._wrap_module_forward(
                language_model,
                lambda args, kwargs: (
                    "stage.llm_decode"
                    if self._infer_llm_phase(args, kwargs) == "decode"
                    else "stage.llm_prefill"
                ),
            )

            layers = getattr(language_model, "layers", None)
            if layers is not None:
                for idx, layer in enumerate(layers):
                    self_attn = getattr(layer, "self_attn", None)
                    mlp = getattr(layer, "mlp", None)

                    self._wrap_module_forward(
                        self_attn,
                        lambda args, kwargs, i=idx: (
                            f"func.text.{self._infer_llm_phase(args, kwargs)}.layer{i}.self_attn"
                        ),
                    )
                    self._wrap_module_forward(
                        mlp,
                        lambda args, kwargs, i=idx: (
                            f"func.text.{self._infer_llm_phase(args, kwargs)}.layer{i}.mlp"
                        ),
                    )

        lm_head = getattr(self._model, "lm_head", None)
        if lm_head is not None:
            self._wrap_module_forward(
                lm_head,
                lambda args, kwargs: f"func.text.{self._infer_llm_phase(args, kwargs)}.lm_head",
            )

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------
    def prepare_inputs(self, image: Image.Image, question: str):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]

        with self.nvtx_range("stage.input_prepare"):
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._device)
        return inputs

    def run_prefill_once(self, image: Image.Image, question: str) -> Dict:
        inputs = self.prepare_inputs(image, question)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with self.nvtx_range("stage.prefill"):
            with torch.no_grad():
                outputs = self._model(
                    **inputs,
                    use_cache=True,
                    return_dict=True,
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with self.nvtx_range("stage.postprocess"):
            next_token = outputs.logits[:, -1:].argmax(dim=-1)

        return {
            "next_token": next_token,
            "past_key_values": outputs.past_key_values,
            "input_ids": inputs.input_ids,
        }

    def run_decode_generate(self, image: Image.Image, question: str, max_new_tokens: int = 64) -> Dict:
        inputs = self.prepare_inputs(image, question)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with self.nvtx_range("stage.decode"):
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    use_cache=True,
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with self.nvtx_range("stage.postprocess"):
            input_len = inputs.input_ids.shape[1]
            generated_ids = output_ids[0][input_len:]
            text = self._processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        return {
            "text": text,
            "token_count": int(generated_ids.shape[0]),
        }

    # ------------------------------------------------------------------
    # Compatibility properties
    # ------------------------------------------------------------------
    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device
