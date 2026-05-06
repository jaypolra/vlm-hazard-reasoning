# -*- coding: utf-8 -*-
"""
05_vlm_inference.py
────────────────────────────────────────────────────────────────────────────
VLM inference script for the SDI industrial hazard detection project.

Reads experiment_queue.csv row by row, sends frame + prompt to a VLM,
writes structured results back into the CSV.

Hardware: 3x NVIDIA RTX PRO 6000 Blackwell @ 96GB each (288GB total VRAM)
Running:  vLLM EngineCore on GPU 0+1 (already loaded model)
          GPU 2 free (~96GB) — use for a second independent model

Supported backends (swap via --model flag, no code changes needed):
  --model vllm          → Hit the already-running vLLM server (OpenAI-compat API)
  --model gemma4        → Gemma 4 / Gemma 3 local (transformers, GPU 2 or multi-GPU)
  --model gemini        → Google Gemini Flash/Pro (API)
  --model gpt4o         → OpenAI GPT-4o (API)
  --model llava         → LLaVA local (via Ollama or transformers)

Usage examples:
  # Use the already-running vLLM server (fastest — model already loaded!)
  python 05_vlm_inference.py --model vllm --limit 5
  python 05_vlm_inference.py --model vllm --phase 2 --resume

  # Load Gemma-3-27B fresh on GPU 2 (96GB free)
  CUDA_VISIBLE_DEVICES=2 python 05_vlm_inference.py --model gemma4 --limit 5

  # Load across all 3 GPUs with tensor parallelism
  python 05_vlm_inference.py --model gemma4 --tp 3 --limit 5

  # API-based (Gemini, GPT-4o)
  python 05_vlm_inference.py --model gemini --phase 1 --variant direct --limit 20
  python 05_vlm_inference.py --model gpt4o --phase 2 --resume

Key env vars:
  VLLM_BASE_URL       → vLLM server URL (default: http://localhost:8000/v1)
  VLLM_MODEL_NAME     → model name as seen by vLLM server (auto-detected)
  GEMMA_MODEL_ID      → HuggingFace model ID (default: google/gemma-3-27b-it)
  GEMMA_TP_SIZE       → tensor_parallel_size for transformers (default: 1)
  GOOGLE_API_KEY      → for --model gemini
  OPENAI_API_KEY      → for --model gpt4o

Required packages (install with pip):
  openai                 → vLLM API + GPT-4o (same client!)
  google-generativeai    → Gemini API
  transformers pillow torch torchvision accelerate
                         → local Gemma 4 / LLaVA

Always run with: python -X utf8 05_vlm_inference.py [args]
(Windows cp1252 encoding will crash on non-ASCII chars without -X utf8)

OUTPUT COLUMNS written back to experiment_queue.csv:
  vlm_model, vlm_response, hazard_detected, severity, safe_or_unsafe,
  confidence, reason, recommended_action, inference_time_sec, run_timestamp
"""

import os
import sys
import csv
import re
import time
import base64
import argparse
import logging
from pathlib import Path
from datetime import datetime

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─── Path resolution ──────────────────────────────────────────────────────────
# The script can run from the project root on any OS.
# Frame paths in the CSV are Windows-style (.\phase1_...) — we normalize them.
SCRIPT_DIR = Path(__file__).parent.resolve()
QUEUE_CSV = SCRIPT_DIR / "experiment_queue.csv"
PROMPTS_DIR = SCRIPT_DIR / "prompts"

# Add prompts/ to path so we can import build_prompt
sys.path.insert(0, str(PROMPTS_DIR))

# ─── Lazy imports for optional heavy deps ─────────────────────────────────────
# These are only imported when the corresponding backend is requested.

def _import_pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        log.error("Pillow not installed. Run: pip install pillow")
        sys.exit(1)

def _import_gemini():
    try:
        import google.generativeai as genai
        return genai
    except ImportError:
        log.error("google-generativeai not installed. Run: pip install google-generativeai")
        sys.exit(1)

def _import_openai():
    try:
        import openai
        return openai
    except ImportError:
        log.error("openai not installed. Run: pip install openai")
        sys.exit(1)

def _import_transformers():
    try:
        import transformers
        import torch
        return transformers, torch
    except ImportError:
        log.error("transformers or torch not installed. Run: pip install transformers torch")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE PARSER
# Extracts structured fields from the VLM's free-text response.
# ═══════════════════════════════════════════════════════════════════════════════

def parse_vlm_response(text: str) -> dict:
    """
    Parse the structured fields from a VLM response.
    
    Expected format (from OUTPUT_FORMAT in prompt_templates.py):
        HAZARD_DETECTED: YES / NO / UNCERTAIN
        SEVERITY: CRITICAL / HIGH / MEDIUM / LOW / NONE
        CONFIDENCE: HIGH / MEDIUM / LOW
        WHAT_I_SEE: ...
        HAZARD_ANALYSIS: ...
        SAFE_OR_UNSAFE: SAFE / UNSAFE / UNCONFIRMABLE
        REASON: ...
        RECOMMENDED_ACTION: ...
    
    Returns dict with parsed values (empty string if not found).
    """
    def _extract(pattern: str, text: str, default: str = "") -> str:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            val = m.group(1).strip()
            # Strip Markdown bold/italic markers (e.g. **YES** → YES)
            val = val.strip("*").strip()
            # Collapse multiline to single line for CSV safety
            val = " ".join(val.split())
            return val[:500]  # cap at 500 chars
        return default

    return {
        "hazard_detected":     _extract(r"HAZARD_DETECTED\s*:\s*([^\n]+)", text),
        "severity":            _extract(r"SEVERITY\s*:\s*([^\n]+)", text),
        "confidence":          _extract(r"CONFIDENCE\s*:\s*([^\n]+)", text),
        "safe_or_unsafe":      _extract(r"SAFE_OR_UNSAFE\s*:\s*([^\n]+)", text),
        "reason":              _extract(r"REASON\s*:\s*([^\n]+)", text),
        "recommended_action":  _extract(r"RECOMMENDED_ACTION\s*:\n?(.*?)(?=\n[A-Z_]+:|$)", text),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME PATH RESOLVER
# CSV paths are Windows-style relative paths. We resolve them cross-platform.
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_frame_path(raw_path: str, base_dir: Path) -> Path:
    """
    Resolve a frame path from the CSV to an absolute path.

    Handles:
      - Windows-style relative paths (.\phase1_...) on Linux/Windows
      - Windows absolute paths (D:\vlm_hazard_dataset\...) on Linux
      - Already-absolute paths
    """
    import re
    # Normalize backslashes to forward slashes
    normalized = raw_path.replace("\\", "/")
    # Strip Windows drive letter prefix when running on Linux/Mac
    # e.g. "D:/vlm_hazard_dataset/..." -> "vlm_hazard_dataset/..."
    if os.name != "nt":
        normalized = re.sub(r"^[A-Za-z]:/", "", normalized)
        # If the path still starts with the project root dir name, strip it
        # e.g. "vlm_hazard_dataset/your_mkv_files/..." -> "your_mkv_files/..."
        project_root_name = base_dir.name  # e.g. "vlm_hazard_dataset"
        if normalized.startswith(project_root_name + "/"):
            normalized = normalized[len(project_root_name) + 1:]
    # Strip leading ./ or / for relative path joining
    normalized = normalized.lstrip("./")
    resolved = base_dir / normalized
    return resolved


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND: GEMMA 4 (Local — via transformers)
# Supports: google/gemma-3-4b-it, google/gemma-3-12b-it, google/gemma-3-27b-it
#           google/gemma-4-12b-it (when released), etc.
# Requires: transformers>=4.45, torch, accelerate, pillow
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND: vLLM (local vLLM server — OpenAI-compatible API)
# Hits the already-running vLLM EngineCore on GPUs 0+1.
# This is the fastest option — model is already loaded, no startup time.
# Requires: openai package (pip install openai), vLLM server running
# ═══════════════════════════════════════════════════════════════════════════════

class VLLMBackend:
    """
    vLLM server backend via OpenAI-compatible REST API.

    The vLLM EngineCore is already running on this machine (GPUs 0+1).
    Default endpoint: http://localhost:8000/v1

    To find the loaded model name:
      curl http://localhost:8000/v1/models

    Override with env vars:
      VLLM_BASE_URL=http://localhost:8000/v1
      VLLM_MODEL_NAME=google/gemma-3-27b-it   (or whatever is loaded)
      VLLM_API_KEY=EMPTY                        (vLLM default)
    """

    DEFAULT_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    DEFAULT_API_KEY  = os.environ.get("VLLM_API_KEY", "EMPTY")

    def __init__(self):
        openai = _import_openai()
        Image = _import_pil()
        self._Image = Image

        self._client = openai.OpenAI(
            base_url=self.DEFAULT_BASE_URL,
            api_key=self.DEFAULT_API_KEY,
        )

        # Auto-detect the loaded model name from the vLLM /models endpoint
        self._model_name = os.environ.get("VLLM_MODEL_NAME", "")
        if not self._model_name:
            try:
                models = self._client.models.list()
                if models.data:
                    self._model_name = models.data[0].id
                    log.info(f"vLLM auto-detected model: {self._model_name}")
                else:
                    self._model_name = "unknown"
                    log.warning("vLLM: no models found at server — is it running?")
            except Exception as e:
                log.error(f"Cannot reach vLLM server at {self.DEFAULT_BASE_URL}: {e}")
                log.error("  Check: is vLLM running? Try: curl http://localhost:8000/v1/models")
                sys.exit(1)

        log.info(f"vLLM backend ready: {self._model_name} @ {self.DEFAULT_BASE_URL}")

    def run(self, image_path: Path, prompt_text: str) -> str:
        """
        Send image + prompt to vLLM via OpenAI vision API format.
        vLLM supports vision models with base64 image_url format.
        """
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        ext = image_path.suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    @property
    def model_name(self) -> str:
        return f"vllm/{self._model_name}"


class Gemma4Backend:
    """
    Local Gemma 4 / Gemma 3 multimodal inference via HuggingFace transformers.

    Hardware: 3x RTX PRO 6000 Blackwell @ 96GB each.
    GPU 2 is free (~96GB) — ideal for Gemma-3-27B standalone.
    Set CUDA_VISIBLE_DEVICES=2 to pin to the free GPU.
    Or use --tp 3 to tensor-parallelize across all 3 GPUs.

    Model options (set via GEMMA_MODEL_ID env var):
      google/gemma-3-4b-it    ~8GB   — fast testing
      google/gemma-3-12b-it   ~24GB  — good balance
      google/gemma-3-27b-it   ~54GB  — best quality (default, fits on GPU 2 alone)
      google/paligemma2-28b-ft-docci-448  — vision-specialist alternative

    Tensor parallel (multi-GPU):
      Set GEMMA_TP_SIZE=2 or =3, or pass --tp N to the CLI.
      With TP=3: 3x96GB = 288GB — can load any existing model.
    """

    DEFAULT_MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-3-27b-it")
    DEFAULT_TP_SIZE  = int(os.environ.get("GEMMA_TP_SIZE", "1"))

    def __init__(self, model_id: str = None, tp_size: int = None):
        transformers, torch = _import_transformers()
        Image = _import_pil()
        self._Image = Image
        self._torch = torch

        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.tp_size  = tp_size  or self.DEFAULT_TP_SIZE

        log.info(f"Loading Gemma model: {self.model_id}")
        log.info(f"  tp_size={self.tp_size} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

        if not torch.cuda.is_available():
            log.error("No CUDA GPU detected. Gemma 4 requires a GPU.")
            sys.exit(1)

        n_gpus = torch.cuda.device_count()
        log.info(f"  Available GPUs: {n_gpus}")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            log.info(f"    GPU {i}: {props.name} | {total/1e9:.0f}GB total | {free/1e9:.1f}GB free")

        dtype = torch.bfloat16

        # device_map strategy:
        #   tp_size=1 → auto (single GPU, picks by memory)
        #   tp_size>1 → balanced across N GPUs
        if self.tp_size > 1:
            device_map = {"balanced"}
            # Restrict to first tp_size GPUs
            max_mem = {i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.92 / 1e9)}GiB"
                       for i in range(min(self.tp_size, n_gpus))}
            log.info(f"  Multi-GPU max_memory: {max_mem}")
        else:
            device_map = "auto"
            max_mem = None

        log.info(f"  Loading processor...")
        try:
            self.processor = transformers.AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                local_files_only=True
            )
        except Exception:
            self.processor = transformers.AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )

        log.info(f"  Loading model weights (dtype={dtype})...")
        kwargs = dict(
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if max_mem:
            kwargs["max_memory"] = max_mem

        try:
            self.model = transformers.AutoModelForImageTextToText.from_pretrained(
                self.model_id, **kwargs, local_files_only=True
            )
        except Exception:
            self.model = transformers.AutoModelForImageTextToText.from_pretrained(
                self.model_id, **kwargs
            )
        self.model.eval()

        # Report actual device placement
        if hasattr(self.model, "hf_device_map"):
            devices = set(str(v) for v in self.model.hf_device_map.values())
            log.info(f"  Model loaded on devices: {devices}")
        log.info(f"  Ready: {self.model_id}")

        # ─── GEMMA 3 VISION WEIGHT FIX ──────────────────────────────────────────
        # Detect if vision weights failed to load due to 'vision_model' key mismatch
        # Note: In Gemma3ForConditionalGeneration, vision_tower is under .model
        vt = getattr(self.model.model, "vision_tower", None)
        if vt is None:
            log.warning("  Could not find vision_tower in model.model. Skipping surgery...")
            return

        vt_params = [k for k, _ in vt.named_parameters()]
        is_blind = len(vt_params) > 0 and vt_params[0].endswith(".weight") and \
                   not vt_params[0].startswith("encoder") # Check if it's missing the expected prefix
        
        # Or simpler: check if the first layer has zero/initial weights vs checkpoint
        # But the MISSING/UNEXPECTED report is the most reliable.
        # Let's perform a manual reload if we detect the 'vision_model' key in the config or via a trial.
        
        log.info("  Applying Gemma 3 vision weight re-mapping fix...")
        try:
            from safetensors.torch import load_file
            from huggingface_hub import snapshot_download
            import glob

            # 1. Find the local cache directory
            cache_dir = snapshot_download(self.model_id, local_files_only=True)
            safetensor_files = glob.glob(os.path.join(cache_dir, "*.safetensors"))
            
            if not safetensor_files:
                log.warning("  No safetensors found for manual fix. Skipping surgery...")
            else:
                log.info(f"  Performing weight surgery on {len(safetensor_files)} shards...")
                for shard in safetensor_files:
                    sd = load_file(shard)
                    fixed_sd = {}
                    replaced = 0
                    for k, v in sd.items():
                        # The fix: model.vision_tower.vision_model.encoder -> model.vision_tower.encoder
                        if "vision_tower.vision_model" in k:
                            new_k = k.replace("vision_tower.vision_model", "vision_tower")
                            fixed_sd[new_k] = v
                            replaced += 1
                        else:
                            fixed_sd[k] = v
                    
                    if replaced > 0:
                        # Load only the remapped keys back into the model
                        self.model.load_state_dict(fixed_sd, strict=False)
                log.info("  Surgery complete. Vision tower weights should now be active!")
        except Exception as e:
            log.error(f"  Vision weight surgery failed: {e}")
            log.warning("  Model might still be 'blind' (guessing from text only).")

    def run(self, image_path: Path, prompt_text: str) -> str:
        """
        Run vision+text inference. Returns raw response string.
        """
        image = self._Image.open(image_path).convert("RGB")

        # Gemma 3/4 chat-template format (image + text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt_text},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Move inputs to the first model device
        first_device = next(self.model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        with self._torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        response = self.processor.decode(generated, skip_special_tokens=True)
        return response.strip()

    @property
    def model_name(self) -> str:
        return f"gemma4/{self.model_id}"


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND: GEMINI (Google API — Flash/Pro)
# Supports: gemini-1.5-flash, gemini-2.0-flash, gemini-1.5-pro
# Requires: google-generativeai, GOOGLE_API_KEY env var
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiBackend:
    """
    Google Gemini API backend (multimodal).
    
    Default model: gemini-2.0-flash (cheapest, fastest, great vision).
    Override with GEMINI_MODEL_ID env var, e.g.:
      GEMINI_MODEL_ID=gemini-1.5-pro python 05_vlm_inference.py --model gemini
    """

    DEFAULT_MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "gemini-2.0-flash")

    def __init__(self, model_id: str = None):
        genai = _import_gemini()
        Image = _import_pil()
        self._Image = Image

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            log.error("GOOGLE_API_KEY not set. Export it or add to .env")
            sys.exit(1)

        genai.configure(api_key=api_key)
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self._model = genai.GenerativeModel(self.model_id)
        log.info(f"Gemini backend ready: {self.model_id}")

    def run(self, image_path: Path, prompt_text: str) -> str:
        image = self._Image.open(image_path).convert("RGB")
        response = self._model.generate_content([image, prompt_text])
        return response.text.strip()

    @property
    def model_name(self) -> str:
        return f"gemini/{self.model_id}"


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND: GPT-4o (OpenAI API)
# Requires: openai>=1.0, OPENAI_API_KEY env var
# ═══════════════════════════════════════════════════════════════════════════════

class GPT4oBackend:
    """
    OpenAI GPT-4o API backend (multimodal via base64 image upload).
    
    Default model: gpt-4o (best vision quality).
    Override with OPENAI_MODEL_ID env var.
    """

    DEFAULT_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "gpt-4o")

    def __init__(self, model_id: str = None):
        openai = _import_openai()
        self._openai = openai

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            log.error("OPENAI_API_KEY not set.")
            sys.exit(1)

        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self._client = openai.OpenAI(api_key=api_key)
        log.info(f"GPT-4o backend ready: {self.model_id}")

    def run(self, image_path: Path, prompt_text: str) -> str:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        ext = image_path.suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")

        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    @property
    def model_name(self) -> str:
        return f"gpt4o/{self.model_id}"


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND: LLaVA (Local via Ollama, or transformers fallback)
# ═══════════════════════════════════════════════════════════════════════════════

class LLaVABackend:
    """
    LLaVA local backend via Ollama (or transformers if Ollama not available).
    
    Ollama mode (default — easiest): requires 'ollama' CLI and model pulled:
      ollama pull llava:13b
    
    Override model with LLAVA_MODEL_ID env var:
      LLAVA_MODEL_ID=llava:34b python 05_vlm_inference.py --model llava
    """

    DEFAULT_MODEL_ID = os.environ.get("LLAVA_MODEL_ID", "llava:13b")

    def __init__(self, model_id: str = None):
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self._mode = "ollama"  # try ollama first

        try:
            import ollama
            self._ollama = ollama
            log.info(f"LLaVA via Ollama: {self.model_id}")
        except ImportError:
            log.warning("ollama package not installed. Trying transformers fallback...")
            self._mode = "transformers"
            transformers, torch = _import_transformers()
            log.info(f"LLaVA via transformers (limited support): {self.model_id}")
            # For full transformers LLaVA support, use llava-hf/* model IDs
            self._setup_transformers_llava(transformers, torch)

    def _setup_transformers_llava(self, transformers, torch):
        """Setup LLaVA via transformers (llava-hf models)."""
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        hf_model_id = os.environ.get("LLAVA_HF_MODEL_ID", "llava-hf/llava-1.5-13b-hf")
        self._processor = AutoProcessor.from_pretrained(hf_model_id)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            hf_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self._torch = torch
        self._Image = _import_pil()

    def run(self, image_path: Path, prompt_text: str) -> str:
        if self._mode == "ollama":
            return self._run_ollama(image_path, prompt_text)
        else:
            return self._run_transformers(image_path, prompt_text)

    def _run_ollama(self, image_path: Path, prompt_text: str) -> str:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        response = self._ollama.chat(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": prompt_text,
                "images": [img_b64],
            }],
        )
        return response["message"]["content"].strip()

    def _run_transformers(self, image_path: Path, prompt_text: str) -> str:
        image = self._Image.open(image_path).convert("RGB")
        prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with self._torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=1024)

        text = self._processor.decode(output[0], skip_special_tokens=True)
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1].strip()
        return text

    @property
    def model_name(self) -> str:
        return f"llava/{self.model_id}"


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

BACKENDS = {
    "vllm":    VLLMBackend,     # already-running vLLM server (fastest!)
    "gemma4":  Gemma4Backend,
    "gemma":   Gemma4Backend,   # alias
    "gemini":  GeminiBackend,
    "gpt4o":   GPT4oBackend,
    "gpt-4o":  GPT4oBackend,    # alias
    "llava":   LLaVABackend,
}


# ═══════════════════════════════════════════════════════════════════════════════
# CSV READ / WRITE
# We read the entire CSV into memory, modify rows in place, write back.
# This is safe for 1,044 rows (< 300KB).
# ═══════════════════════════════════════════════════════════════════════════════

def load_queue(csv_path: Path) -> tuple[list[dict], list[str]]:
    """Load experiment_queue.csv. Returns (rows, fieldnames)."""
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    log.info(f"Loaded {len(rows)} rows from {csv_path.name}")
    return rows, fieldnames


def save_queue(csv_path: Path, rows: list[dict], fieldnames: list[str]):
    """Write all rows back to experiment_queue.csv (atomic-ish via temp file)."""
    tmp_path = csv_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    # Replace original
    tmp_path.replace(csv_path)


def row_is_done(row: dict) -> bool:
    """Return True if this row already has a successful VLM response filled in.

    ERROR rows (frame not found, inference crash) are treated as NOT done
    so they will be retried on the next --resume run.
    """
    resp = row.get("vlm_response", "").strip()
    if not resp:
        return False
    # Treat error results as not done — they need a retry
    if resp.startswith("ERROR:") or row.get("hazard_detected", "") == "ERROR":
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# FILTER ROWS
# ═══════════════════════════════════════════════════════════════════════════════

def filter_rows(
    rows: list[dict],
    phase: str = "both",
    variant: str = "all",
    resume: bool = False,
    limit: int = None,
) -> list[tuple[int, dict]]:
    """
    Return (original_index, row) pairs that match the filter criteria.
    
    Args:
        rows:    all rows from the CSV
        phase:   "1" | "2" | "both"
        variant: "direct" | "domain" | "cot" | "multi_q" | "all"
        resume:  if True, skip rows that already have vlm_response
        limit:   max number of rows to return (for cost control)
    """
    selected = []
    for i, row in enumerate(rows):
        # Phase filter
        if phase == "1" and row["phase"] != "phase1":
            continue
        if phase == "2" and row["phase"] != "phase2":
            continue

        # Variant filter
        if variant != "all" and row.get("variant") != variant:
            continue

        # Resume filter — skip already-done rows
        if resume and row_is_done(row):
            continue

        selected.append((i, row))

        if limit and len(selected) >= limit:
            break

    log.info(f"Rows selected for inference: {len(selected)}")
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-ROW INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_row(backend, row: dict, base_dir: Path) -> dict:
    """
    Run VLM inference on a single queue row.
    
    Returns a dict of updated column values to merge into the row.
    """
    from prompt_templates import build_prompt

    # Resolve frame path (Windows-style → Linux-compatible)
    raw_path = row.get("frame_path", "")
    frame_path = resolve_frame_path(raw_path, base_dir)

    if not frame_path.exists():
        log.warning(f"Frame not found: {frame_path}")
        return {
            "vlm_model": backend.model_name,
            "vlm_response": "ERROR: frame not found",
            "hazard_detected": "ERROR",
            "severity": "",
            "safe_or_unsafe": "",
            "confidence": "",
            "reason": f"Frame path not found: {frame_path}",
            "recommended_action": "",
            "inference_time_sec": "",
            "run_timestamp": datetime.now().isoformat(),
        }

    # Build prompt
    variant = row.get("variant", "direct")
    mode = row.get("mode", "generic")
    hazard_key = row.get("hazard_key", "") or None

    try:
        prompt_text = build_prompt(variant=variant, mode=mode, hazard_key=hazard_key)
    except ValueError as e:
        log.warning(f"Prompt build error: {e}")
        prompt_text = build_prompt(variant="direct", mode=mode)

    # Run inference
    t0 = time.time()
    try:
        raw_response = backend.run(frame_path, prompt_text)
        inference_time = round(time.time() - t0, 2)
    except Exception as e:
        log.error(f"Inference error on {frame_path.name}: {e}")
        return {
            "vlm_model": backend.model_name,
            "vlm_response": f"ERROR: {e}",
            "hazard_detected": "ERROR",
            "severity": "",
            "safe_or_unsafe": "",
            "confidence": "",
            "reason": str(e)[:200],
            "recommended_action": "",
            "inference_time_sec": round(time.time() - t0, 2),
            "run_timestamp": datetime.now().isoformat(),
        }

    # Parse structured fields
    parsed = parse_vlm_response(raw_response)

    return {
        "vlm_model": backend.model_name,
        "vlm_response": raw_response[:2000],  # cap for CSV
        "hazard_detected": parsed["hazard_detected"],
        "severity": parsed["severity"],
        "safe_or_unsafe": parsed["safe_or_unsafe"],
        "confidence": parsed["confidence"],
        "reason": parsed["reason"],
        "recommended_action": parsed["recommended_action"],
        "inference_time_sec": inference_time,
        "run_timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VLM inference for industrial hazard detection (SDI/CIVS project)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with Gemma 4 (local), first 5 rows
  python -X utf8 05_vlm_inference.py --model gemma4 --limit 5

  # Run Phase 1, direct variant only, with Gemini API
  python -X utf8 05_vlm_inference.py --model gemini --phase 1 --variant direct --limit 20

  # Full Phase 2 run with Gemma 4, resume if interrupted
  python -X utf8 05_vlm_inference.py --model gemma4 --phase 2 --resume

  # GPT-4o full run, both phases
  python -X utf8 05_vlm_inference.py --model gpt4o --phase both --resume

  # Use specific Gemma model size
  GEMMA_MODEL_ID=google/gemma-3-27b-it python -X utf8 05_vlm_inference.py --model gemma4 --limit 10
        """
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(BACKENDS.keys()),
        help="VLM backend: vllm (running server) | gemma4 | gemini | gpt4o | llava",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        dest="tp_size",
        help="Tensor parallel size for Gemma4 backend (default: 1 = single GPU). "
             "Use --tp 2 or --tp 3 to spread across multiple GPUs.",
    )
    parser.add_argument(
        "--phase",
        default="both",
        choices=["1", "2", "both"],
        help="Which phase to run (default: both)",
    )
    parser.add_argument(
        "--variant",
        default="all",
        choices=["direct", "domain", "cot", "multi_q", "all"],
        help="Which prompt variant to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of rows to run (cost control for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows that already have vlm_response filled in",
    )
    parser.add_argument(
        "--csv",
        default=str(QUEUE_CSV),
        help=f"Path to experiment_queue.csv (default: {QUEUE_CSV})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without calling the VLM",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save CSV after every N completed rows (default: 10)",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        log.error(f"CSV not found: {csv_path}")
        sys.exit(1)

    # Load queue
    rows, fieldnames = load_queue(csv_path)

    # Filter rows
    selected = filter_rows(
        rows,
        phase=args.phase,
        variant=args.variant,
        resume=args.resume,
        limit=args.limit,
    )

    if not selected:
        log.info("No rows to process. All done or filters returned nothing.")
        return

    if args.dry_run:
        log.info(f"DRY RUN — would process {len(selected)} rows with model={args.model}")
        if args.model in ("gemma4", "gemma"):
            model_id = os.environ.get("GEMMA_MODEL_ID", Gemma4Backend.DEFAULT_MODEL_ID)
            tp = args.tp_size or int(os.environ.get("GEMMA_TP_SIZE", "1"))
            log.info(f"  Gemma model: {model_id} | tp_size={tp}")
        elif args.model == "vllm":
            log.info(f"  vLLM server: {VLLMBackend.DEFAULT_BASE_URL}")
        for idx, (i, row) in enumerate(selected[:10]):
            log.info(f"  [{idx+1}] row={i+2} | {row['video_id']} | {row['phase']} | "
                     f"{row['hazard_key']} | {row['variant']} | {row['frame_file']}")
        if len(selected) > 10:
            log.info(f"  ... and {len(selected) - 10} more rows")
        return

    # Initialize backend (loads model / API client)
    backend_cls = BACKENDS[args.model]

    if args.model in ("gemma4", "gemma") and args.tp_size:
        backend = backend_cls(tp_size=args.tp_size)
    else:
        backend = backend_cls()

    # Run inference
    log.info(f"Starting inference: {len(selected)} rows | model={backend.model_name}")
    completed = 0
    errors = 0

    for run_idx, (queue_idx, row) in enumerate(selected, 1):
        log.info(
            f"[{run_idx}/{len(selected)}] "
            f"row={queue_idx+2} | {row['video_id']} | {row['phase']} | "
            f"{row['hazard_key']} | {row['variant']} | {row['frame_file']}"
        )

        updates = run_one_row(backend, row, SCRIPT_DIR)

        # Merge updates into the original row
        rows[queue_idx].update(updates)

        if "ERROR" in updates.get("hazard_detected", ""):
            errors += 1
            log.warning(f"  → ERROR: {updates['reason'][:80]}")
        else:
            completed += 1
            log.info(
                f"  → {updates['hazard_detected']} | {updates['severity']} | "
                f"{updates['safe_or_unsafe']} | {updates['inference_time_sec']}s"
            )

        # Periodic save (so we don't lose progress on crash)
        if run_idx % args.save_every == 0:
            save_queue(csv_path, rows, fieldnames)
            log.info(f"  [checkpoint] Saved after {run_idx} rows")

    # Final save
    save_queue(csv_path, rows, fieldnames)
    log.info("=" * 60)
    log.info(f"DONE. Completed={completed}, Errors={errors}, Total={len(selected)}")
    log.info(f"Results written to: {csv_path}")


if __name__ == "__main__":
    main()
