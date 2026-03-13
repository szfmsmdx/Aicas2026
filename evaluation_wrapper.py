"""
AICAS 2026 - Participant Core Modification File

Participants should modify the VLMModel class to implement optimizations.

Note:
- Benchmark directly calls self.model.generate() for performance testing.
- Your optimizations should modify self.model or its operators in __init__ via Monkey Patch.
- The generate() method is optional and mainly for debugging.
"""
from typing import Dict
import os
import time
import contextlib
try:
    from PIL import Image
except ImportError:
    # For testing without PIL
    class Image:
        pass
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


class VLMModel:
    """
    Participant optimization class - modify this to implement optimizations.
    
    Optimization Architecture:
    - Split optimizations into separate methods for isolation and testing
    - Enable/disable each optimization independently in __init__
    - Each optimization method can be tested individually
    
    Important Notes:
    1. Benchmark directly calls self.model.generate() for performance testing.
    2. Your optimizations should modify self.model or its operators via Monkey Patch.
    3. All optimizations are applied in __init__ by calling optimization methods.
    """
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        Initialize model and apply optimizations.
        
        Args:
            model_path: Qwen3-VL-2B-Instruct model path
            device: CUDA device, e.g., "cuda:0"
        """
        self._device = device
        self.model_path = model_path
        
        # Load processor
        print(f"[VLMModel] Loading processor from {model_path}...")
        self._processor = AutoProcessor.from_pretrained(model_path)
        
        # Load model
        print(f"[VLMModel] Loading model with FP16...")
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device
        )
        self._model.eval()

        # Optional: profiling wrapper (enabled via env var)
        self._profile_state = {"enabled": False, "count": 0, "limit": 0}
        if os.getenv("PROFILE", "0") == "1":
            self._enable_generate_profiler()
        if os.getenv("NVTX", "0") == "1":
            self._enable_nvtx_ranges()
        
        # Track applied optimizations
        self._optimizations_applied = []
        
        # ================================================================
        # Participant Optimization Area - Enable/disable optimizations here
        # Uncomment the optimization methods you want to apply
        # ================================================================
        
        # 1. Vision Encoder Acceleration
        # self._optimize_vision_encoder()
        
        # 2. KV Cache Management
        # self._optimize_kv_cache()
        
        # 3. Cross-modal Connector Optimization
        # self._optimize_cross_modal_connector()
        
        # 4. Flash Attention Optimization
        # self._enable_flash_attention()
        
        # 5. Quantization
        # self._apply_quantization()
        
        # Optional: Explore model structure before optimization
        # self._explore_model_structure()
        
        # ================================================================
        
        print(f"[VLMModel] Model loaded successfully on {device}")
        if self._optimizations_applied:
            print(f"[VLMModel] Applied optimizations: {', '.join(self._optimizations_applied)}")
    
    # ================================================================
    # Optimization Methods - Implement your optimizations here
    # ================================================================
    
    def _explore_model_structure(self):
        """
        Helper method to explore model structure.
        
        Use this to understand the model architecture before implementing optimizations.
        This helps identify where to apply monkey patches.
        """
        print("=" * 60)
        print("Model Structure Exploration")
        print("=" * 60)
        
        # Explore vision model structure
        if hasattr(self._model, 'vision_model'):
            print(f"Vision Model: {type(self._model.vision_model)}")
            if hasattr(self._model.vision_model, 'encoder'):
                if hasattr(self._model.vision_model.encoder, 'layers'):
                    print(f"  Vision Encoder Layers: {len(self._model.vision_model.encoder.layers)}")
                    # Show first layer structure
                    if len(self._model.vision_model.encoder.layers) > 0:
                        print(f"  First Layer Type: {type(self._model.vision_model.encoder.layers[0])}")
        else:
            print("Vision Model: Not found (model structure may differ)")
        
        # Explore language model structure
        if hasattr(self._model, 'model'):
            print(f"Language Model: {type(self._model.model)}")
            if hasattr(self._model.model, 'layers'):
                print(f"  Language Model Layers: {len(self._model.model.layers)}")
        else:
            print("Language Model: Not found (model structure may differ)")
        
        # Explore cross-modal components
        cross_modal_attrs = ['connector', 'cross_attn', 'cross_attention', 'proj', 'projector']
        found_components = []
        for attr in cross_modal_attrs:
            if hasattr(self._model, attr):
                found_components.append(attr)
        if found_components:
            print(f"Cross-modal Components: {', '.join(found_components)}")
        else:
            print("Cross-modal Components: Explore manually (structure may vary)")
        
        print("=" * 60)
        print("Tip: Use print(self._model) to see full model structure")
        print("=" * 60)
    
    def _optimize_vision_encoder(self):
        """
        Optimize Vision Encoder for high-resolution image inputs.
        
        Optimization Directions:
        1. Patch embedding convolution optimization
        2. Vision Transformer attention mechanism optimization
        3. Layer normalization optimization
        4. Memory-efficient image processing
        
        Implementation Steps:
        1. Inspect model structure: call self._explore_model_structure()
        2. Identify bottlenecks using profiling tools (PyTorch Profiler, nsys, etc.)
        3. Implement optimized operators (Triton/CUDA kernels)
        4. Replace original operators via monkey patch
        
        Target Components:
        - self._model.vision_model (if exists)
        - Vision encoder layers and attention mechanisms
        - Convolution operations in patch embedding
        """
        # TODO: Implement your Vision Encoder optimization here
        # 
        # Example workflow:
        # 1. from your_optimization import optimized_attention, optimized_conv
        # 2. Inspect: print(self._model.vision_model) to find target layers
        # 3. Replace: layer.self_attn.forward = optimized_attention
        # 4. Test: Run benchmark to verify improvement
        
        if 'vision_encoder' not in self._optimizations_applied:
            self._optimizations_applied.append('vision_encoder')
    
    def _optimize_kv_cache(self):
        """
        Optimize KV Cache management to reduce memory fragmentation.
        
        Optimization Directions:
        1. Memory layout optimization (contiguous memory allocation)
        2. Fragmentation-free allocation strategies
        3. Efficient cache reuse patterns
        4. Dynamic cache sizing
        
        Implementation Steps:
        1. Understand current KV cache implementation in model layers
        2. Design memory-efficient cache allocation strategy
        3. Implement custom KV cache allocator if needed
        4. Apply optimizations via monkey patch or config modification
        
        Target Components:
        - self._model.config (cache configuration)
        - Attention layers (KV cache allocation)
        - Generation loop (cache management)
        """
        # Enable KV Cache first
        self._model.config.use_cache = True
        if hasattr(self._model.config, 'pad_token_id'):
            if self._model.config.pad_token_id is None:
                self._model.config.pad_token_id = self._model.config.eos_token_id
        
        # TODO: Implement advanced KV Cache optimizations here
        # 
        # Example workflow:
        # 1. from your_optimization import FragmentationFreeKVCache
        # 2. for layer in self._model.model.layers:
        # 3.     layer.attention.custom_kv_cache = FragmentationFreeKVCache()
        # 4. Test: Monitor memory usage and generation speed
        
        if 'kv_cache' not in self._optimizations_applied:
            self._optimizations_applied.append('kv_cache')
    
    def _optimize_cross_modal_connector(self):
        """
        Optimize Cross-modal Connector computation efficiency.
        
        Optimization Directions:
        1. Cross-attention mechanism optimization
        2. Vision-to-language projection optimization
        3. Multi-modal fusion layer efficiency
        4. Feature alignment and transformation optimization
        
        Implementation Steps:
        1. Identify cross-modal components using self._explore_model_structure()
        2. Profile cross-modal operations to find bottlenecks
        3. Implement optimized cross-attention or projection kernels
        4. Replace original operations via monkey patch
        
        Note: Qwen3-VL's cross-modal structure may vary.
        Use model exploration to identify actual component names and locations.
        """
        # TODO: Implement your Cross-modal Connector optimization here
        # 
        # Example workflow:
        # 1. Explore: self._explore_model_structure() to find connector components
        # 2. from your_optimization import optimized_cross_attention
        # 3. Identify: Inspect model to find cross-attention layers
        # 4. Replace: connector.cross_attention.forward = optimized_cross_attention
        # 5. Test: Verify accuracy and performance improvements
        
        if 'cross_modal' not in self._optimizations_applied:
            self._optimizations_applied.append('cross_modal')
    
    def _enable_flash_attention(self):
        """
        Enable or implement Flash Attention optimization.
        
        Implementation Approaches:
        
        Approach 1: Enable PyTorch's Built-in Flash Attention (Simple)
            - Uses torch.backends.cuda.enable_flash_sdp(True)
            - Easy to enable but limited customization
            - May not work for all attention patterns in Qwen3-VL
        
        Approach 2: Implement Custom Flash Attention (Advanced, Recommended)
            - Write custom Triton/CUDA kernels for attention computation
            - Replace torch.nn.functional.scaled_dot_product_attention
            - Full control over attention computation and memory layout
            - Better performance potential but requires more implementation effort
        
        Recommended: Implement Approach 2 for better performance gains.
        Use profiling to identify which attention operations benefit most from optimization.
        """
        # TODO: Choose and implement your Flash Attention approach
        
        # Approach 1: Simple (enable PyTorch built-in)
        # torch.backends.cuda.enable_flash_sdp(True)
        
        # Approach 2: Advanced (custom implementation - recommended)
        # from your_optimization import custom_flash_attention
        # torch.nn.functional.scaled_dot_product_attention = custom_flash_attention
        # 
        # Or replace at layer level:
        # for layer in self._model.model.layers:
        #     layer.self_attn.forward = custom_attention_with_flash
        
        if 'flash_attention' not in self._optimizations_applied:
            self._optimizations_applied.append('flash_attention')
    
    def _apply_quantization(self):
        """
        Apply quantization to reduce model size and speed up inference.
        
        Optimization Directions:
        1. INT8 quantization (8-bit integer)
        2. FP8 quantization (8-bit floating point)
        3. Mixed precision quantization
        4. Dynamic vs static quantization
        
        Implementation Steps:
        1. Choose quantization strategy based on accuracy/performance trade-off
        2. Use quantization libraries (BitsAndBytes, TensorRT, etc.)
        3. Calibrate quantized model on validation data
        4. Verify accuracy preservation
        
        Note: Quantization may require reloading the model with quantization config.
        Consider applying quantization before other optimizations if model reload is needed.
        """
        # TODO: Implement your quantization here
        # 
        # Example workflow:
        # 1. from transformers import BitsAndBytesConfig
        # 2. quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        # 3. Note: May need to reload model with quantization config
        # 4. Test: Verify accuracy and performance improvements
        
        if 'quantization' not in self._optimizations_applied:
            self._optimizations_applied.append('quantization')

    def _enable_generate_profiler(self):
        """
        Wrap model.generate with torch.profiler + NVTX ranges.

        Controlled by env vars:
        - PROFILE=1 (enable wrapper)
        - PROFILE_DIR=profile (output directory)
        - PROFILE_TAG=mm-dd-hh-mm (default timestamp)
        - PROFILE_STEPS=1 (number of generate calls to profile)
        - PROFILE_SKIP=0 (skip first N generate calls, e.g. warmup)
        - NVTX=1 (enable NVTX ranges)
        """
        profile_dir = os.getenv("PROFILE_DIR", "profile")
        os.makedirs(profile_dir, exist_ok=True)
        profile_tag = os.getenv("PROFILE_TAG", time.strftime("%m-%d-%H-%M"))
        limit = int(os.getenv("PROFILE_STEPS", "1"))
        nvtx_enabled = os.getenv("NVTX", "1") == "1"

        original_generate = self._model.generate
        skip = int(os.getenv("PROFILE_SKIP", "0"))
        self._profile_state = {"enabled": True, "count": 0, "limit": limit, "total": 0, "skip": skip}

        def _nvtx_range(name: str):
            if not nvtx_enabled or not torch.cuda.is_available():
                return contextlib.nullcontext()
            return torch.cuda.nvtx.range(name)

        def wrapped_generate(*args, **kwargs):
            self._profile_state["total"] += 1
            if self._profile_state["total"] <= self._profile_state["skip"]:
                return original_generate(*args, **kwargs)

            if self._profile_state["count"] >= self._profile_state["limit"]:
                return original_generate(*args, **kwargs)

            self._profile_state["count"] += 1
            max_new_tokens = kwargs.get("max_new_tokens", None)
            step_tag = f"{profile_tag}_tok{max_new_tokens}" if max_new_tokens is not None else profile_tag
            trace_path = os.path.join(profile_dir, f"trace_{step_tag}_{self._profile_state['count']}.json")
            table_path = os.path.join(profile_dir, f"table_{step_tag}_{self._profile_state['count']}.txt")

            with _nvtx_range("benchmark_generate"):
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False
                ) as prof:
                    out = original_generate(*args, **kwargs)

            try:
                prof.export_chrome_trace(trace_path)
                with open(table_path, "w", encoding="utf-8") as f:
                    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
            except Exception as e:
                print(f"[VLMModel] Profiler export failed: {e}")

            return out

        self._model.generate = wrapped_generate
        print(f"[VLMModel] Profiling enabled: dir={profile_dir}, tag={profile_tag}, steps={limit}, skip={skip}")

    def _enable_nvtx_ranges(self):
        """
        Add NVTX ranges to key stages for Nsight Systems/Compute.

        Controlled by env vars:
        - NVTX=1 (enable ranges)
        - NVTX_VERBOSE=1 (print which modules were wrapped)

        Ranges:
        - vision_encoder
        - cross_modal
        - prefill / decode (based on past_key_values)
        - preprocess (processor.apply_chat_template)
        """
        if not torch.cuda.is_available():
            print("[VLMModel] NVTX requested but CUDA is not available.")
            return

        verbose = os.getenv("NVTX_VERBOSE", "0") == "1"

        def _nvtx_range(name: str):
            return torch.cuda.nvtx.range(name)

        def _wrap_method(obj, method_name: str, range_name: str):
            original = getattr(obj, method_name, None)
            if original is None or not callable(original):
                return False
            if getattr(original, "_nvtx_wrapped", False):
                return False

            def wrapped(*args, **kwargs):
                with _nvtx_range(range_name):
                    return original(*args, **kwargs)

            wrapped._nvtx_wrapped = True
            setattr(obj, method_name, wrapped)
            return True

        def _wrap_lm_forward(obj, method_name: str):
            original = getattr(obj, method_name, None)
            if original is None or not callable(original):
                return False
            if getattr(original, "_nvtx_wrapped", False):
                return False

            def wrapped(*args, **kwargs):
                past = kwargs.get("past_key_values", None)
                range_name = "decode" if past else "prefill"
                with _nvtx_range(range_name):
                    return original(*args, **kwargs)

            wrapped._nvtx_wrapped = True
            setattr(obj, method_name, wrapped)
            return True

        wrapped = []

        # Processor preprocessing stage
        if _wrap_method(self._processor, "apply_chat_template", "preprocess"):
            wrapped.append("processor.apply_chat_template -> preprocess")

        # Vision encoder stage (common attribute names)
        for attr in ["vision_model", "vision_tower", "visual", "vision_encoder"]:
            vision_obj = getattr(self._model, attr, None)
            if vision_obj is not None:
                if _wrap_method(vision_obj, "forward", "vision_encoder"):
                    wrapped.append(f"model.{attr}.forward -> vision_encoder")
                    break

        # Cross-modal connector stage (best-effort)
        for attr in ["connector", "projector", "proj", "cross_attn", "cross_attention", "mm_projector"]:
            cm_obj = getattr(self._model, attr, None)
            if cm_obj is not None:
                if _wrap_method(cm_obj, "forward", "cross_modal"):
                    wrapped.append(f"model.{attr}.forward -> cross_modal")
                    break

        # Prefill / decode ranges
        for attr in ["model", "language_model"]:
            lm_obj = getattr(self._model, attr, None)
            if lm_obj is not None and _wrap_lm_forward(lm_obj, "forward"):
                wrapped.append(f"model.{attr}.forward -> prefill/decode")
                break
        if _wrap_lm_forward(self._model, "forward"):
            wrapped.append("model.forward -> prefill/decode")

        if verbose and wrapped:
            print("[VLMModel] NVTX wrapped modules:")
            for item in wrapped:
                print(f"  - {item}")
        elif verbose:
            print("[VLMModel] NVTX enabled but no target modules were wrapped.")
    
    # Required properties for benchmark
    @property
    def processor(self):
        """
        Required by benchmark for input processing.
        
        Benchmark uses this to prepare inputs with unified tokenizer.
        """
        return self._processor
    
    @property
    def model(self):
        """
        Required by benchmark for direct model.generate() calls.
        
        Benchmark directly calls self.model.generate() for performance testing.
        Your optimizations should modify this model object or its operators.
        """
        return self._model
    
    @property
    def device(self):
        """
        Required by benchmark for device information.
        """
        return self._device
    
    def generate(
        self, 
        image: Image.Image, 
        question: str, 
        max_new_tokens: int = 128
    ) -> Dict:
        """
        Generate answer (optional method, mainly for debugging).
        
        Note: Benchmark uses self.model.generate() directly for performance testing.
        This method is provided for convenience and debugging purposes.
        
        Args:
            image: PIL Image object
            question: Question text
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Dict: {
                "text": str,        # Generated text answer
                "token_count": int  # Generated token count
            }
        """
        # Build Qwen3-VL message format
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]
        
        # Process inputs
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self._device)
        
        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                use_cache=True
            )
        
        # Extract generated tokens (remove input part)
        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[0][input_len:]
        
        # Decode
        text = self._processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return {
            "text": text,
            "token_count": len(generated_ids)
        }
