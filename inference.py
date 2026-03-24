"""
Inference Module

Provides a unified interface for local Vision-Language Model inference
via either Ollama's HTTP API or mlx-lm's Python API.
"""

import base64
import io
import json
import logging
import time

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# Default max image dimension (pixels) sent to models.
# Keeps memory and payload size manageable on <=16 GB machines.
DEFAULT_MAX_IMAGE_DIM = 1024

SYSTEM_PROMPT = """\
You are a board-certified radiologist assistant performing a structured analysis \
of a medical image. You MUST follow this exact 4-step Chain-of-Thought framework. \
Clearly label each step.

**Step 1 – Modality Identification:**
Identify the imaging modality and specific scan type (e.g., "PA Chest X-ray", \
"Axial T2-weighted MRI of the brain", "Contrast-enhanced abdominal CT"). \
Note the anatomical region and patient positioning if discernible.

**Step 2 – Observations:**
Describe all key findings systematically. Note any abnormalities such as \
hyperechoic/hypodense regions, masses, lesions, fractures, misalignment, \
fluid collections, or abnormal opacities. Also note relevant normal findings. \
Use precise anatomical terminology.

**Step 3 – Diagnostic Synthesis:**
Based on your observations, provide a preliminary diagnostic assessment. \
List differential diagnoses ranked by likelihood. Assign a confidence score \
(Low / Moderate / High) to your primary assessment and briefly justify it.

**Step 4 – Referral Recommendation:**
Recommend the appropriate medical specialist(s) for follow-up:
- **Internist** – for systemic, metabolic, or chronic conditions
- **Surgeon** – for fractures, acute surgical findings, or operable lesions
- **General Practitioner** – for initial evaluation or low-acuity findings
- **Other specialist** – name the specialty if applicable (e.g., Pulmonologist, Neurologist)
Suggest specific next clinical steps (additional imaging, lab work, biopsy, etc.).

IMPORTANT: All findings are PROVISIONAL and require verification by a licensed \
physician. This analysis does not constitute a medical diagnosis.\
"""


def _resize_if_needed(image, max_dim=DEFAULT_MAX_IMAGE_DIM):
    """Down-scale a PIL Image so its largest side is at most *max_dim* pixels.

    Uses LANCZOS resampling.  Returns the (possibly resized) image.
    """
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    logger.info("Resizing image from %dx%d to %dx%d (max_dim=%d)", w, h, new_w, new_h, max_dim)
    return image.resize((new_w, new_h), Image.LANCZOS)


def _image_to_base64(image, max_dim=DEFAULT_MAX_IMAGE_DIM):
    """Convert a PIL Image or numpy array to a base64-encoded JPEG string.

    The image is first down-scaled to *max_dim* (longest side) and encoded as
    JPEG quality-85 to reduce memory and payload size vs. the previous PNG path.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image = _resize_if_needed(image, max_dim)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    logger.info("Base64 payload size: %.1f KB", len(encoded) / 1024)
    return encoded


class OllamaBackend:
    """Inference backend using Ollama's local HTTP API."""

    def __init__(self, model_name="llama3.2-vision", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = self._validate_url(base_url)

    @staticmethod
    def _validate_url(url):
        """Validate that the Ollama URL points to localhost only (SSRF prevention)."""
        from urllib.parse import urlparse

        parsed = urlparse(url.rstrip("/"))
        if parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(
                f"Ollama URL must point to localhost (got '{parsed.hostname}'). "
                "Remote Ollama servers are blocked to prevent SSRF."
            )
        return url.rstrip("/")

    def is_available(self):
        """Check if Ollama is running and the model is accessible."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                return any(self.model_name in m for m in models)
        except requests.ConnectionError:
            pass
        return False

    def list_models(self):
        """List available Ollama models with details.

        Returns a list of model dicts on success (may be empty if no models
        are installed), or None if the server cannot be reached.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                return resp.json().get("models", [])
        except requests.ConnectionError:
            pass
        return None

    def list_model_names(self):
        """List available Ollama model names (strings only).

        Returns a list of name strings on success, or None if the server
        cannot be reached.
        """
        models = self.list_models()
        if models is None:
            return None
        return [m["name"] for m in models]

    def pull_model(self, model_name, stream=True):
        """Pull/download a model from the Ollama registry.

        Yields progress dicts when stream=True: {"status", "completed", "total"}.
        Returns final status dict when stream=False.
        """
        payload = {"model": model_name, "stream": stream}
        if stream:
            resp = requests.post(
                f"{self.base_url}/api/pull",
                json=payload,
                stream=True,
                timeout=10,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    yield json.loads(line)
        else:
            resp = requests.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=3600,
            )
            resp.raise_for_status()
            yield resp.json()

    def delete_model(self, model_name):
        """Delete a model from Ollama. Returns True on success."""
        resp = requests.delete(
            f"{self.base_url}/api/delete",
            json={"model": model_name},
            timeout=30,
        )
        return resp.status_code == 200

    def show_model(self, model_name):
        """Get detailed info about a model."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/show",
                json={"model": model_name},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
        except requests.ConnectionError:
            pass
        return None

    def list_running(self):
        """List currently running/loaded models."""
        try:
            resp = requests.get(f"{self.base_url}/api/ps", timeout=5)
            if resp.status_code == 200:
                return resp.json().get("models", [])
        except requests.ConnectionError:
            pass
        return []

    def analyze(
        self,
        image,
        user_prompt="Analyze this medical image.",
        metadata=None,
        max_image_dim=DEFAULT_MAX_IMAGE_DIM,
        on_token=None,
        on_log=None,
    ):
        """Send an image to Ollama for analysis (streaming).

        Parameters
        ----------
        on_token : callable(str) | None
            Called with each token fragment as it arrives.
        on_log : callable(str) | None
            Called with backend log messages (timing, model info, etc.).
        max_image_dim : int
            Maximum image dimension in pixels sent to the model.
        """
        t0 = time.monotonic()
        if on_log:
            on_log(f"[backend] Encoding image (max {max_image_dim}px)…")

        img_b64 = _image_to_base64(image, max_dim=max_image_dim)

        context_parts = [user_prompt]
        if metadata:
            context_parts.append("\nDICOM Metadata Context:")
            for k, v in metadata.items():
                context_parts.append(f"  {k}: {v}")

        payload = {
            "model": self.model_name,
            "system": SYSTEM_PROMPT,
            "prompt": "\n".join(context_parts),
            "images": [img_b64],
            "stream": True,
        }

        if on_log:
            on_log(f"[backend] Sending to Ollama ({self.model_name})…")

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=600,
        )
        resp.raise_for_status()

        tokens = []
        token_count = 0
        first_token_time = None
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            fragment = chunk.get("response", "")
            if fragment:
                if first_token_time is None:
                    first_token_time = time.monotonic()
                    if on_log:
                        on_log(
                            f"[ollama] First token after "
                            f"{first_token_time - t0:.1f}s"
                        )
                tokens.append(fragment)
                token_count += 1
                if on_token:
                    on_token(fragment)
            if chunk.get("done"):
                eval_duration = chunk.get("eval_duration", 0)
                prompt_eval_count = chunk.get("prompt_eval_count", 0)
                eval_count = chunk.get("eval_count", 0)
                if on_log:
                    elapsed = time.monotonic() - t0
                    tps = (
                        eval_count / (eval_duration / 1e9)
                        if eval_duration
                        else 0
                    )
                    on_log(
                        f"[ollama] Done — {eval_count} tokens in {elapsed:.1f}s "
                        f"({tps:.1f} tok/s) | prompt tokens: {prompt_eval_count}"
                    )

        return "".join(tokens) or "No response received."


class MlxLmBackend:
    """Inference backend using the mlx-lm library (Apple Silicon optimized)."""

    def __init__(self, model_name="mlx-community/GLM-4.5V-9B-4bit"):
        self.model_name = model_name
        self._model = None
        self._processor = None

    def is_available(self):
        """Check if mlx-lm is installed and importable."""
        try:
            import mlx_lm  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is None:
            from mlx_lm import load
            self._model, self._processor = load(self.model_name)

    def analyze(
        self,
        image,
        user_prompt="Analyze this medical image.",
        metadata=None,
        max_image_dim=DEFAULT_MAX_IMAGE_DIM,
        on_token=None,
        on_log=None,
    ):
        """Run inference using mlx-lm.

        Attempts streaming via ``mlx_lm.stream_generate`` so that
        *on_token* is called per-chunk, matching OllamaBackend's
        behaviour.  Falls back to ``mlx_lm.generate`` (single-shot)
        when ``stream_generate`` is unavailable or does not support
        the ``image`` kwarg, in which case *on_token* is called once
        with the complete response.
        """
        t0 = time.monotonic()
        self._load_model()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Down-scale before sending to model
        image = _resize_if_needed(image, max_image_dim)

        context_parts = [user_prompt]
        if metadata:
            context_parts.append("\nDICOM Metadata Context:")
            for k, v in metadata.items():
                context_parts.append(f"  {k}: {v}")

        full_prompt = SYSTEM_PROMPT + "\n\n" + "\n".join(context_parts)

        # --- Try streaming first -------------------------------------------
        response = self._try_stream(
            full_prompt, image, on_token, on_log, t0,
        )
        if response is not None:
            return response

        # --- Fallback: single-shot generate --------------------------------
        from mlx_lm import generate

        if on_log:
            on_log(f"[mlx-lm] Running inference ({self.model_name})…")

        result = generate(
            self._model,
            self._processor,
            prompt=full_prompt,
            image=image,
            max_tokens=2048,
            verbose=False,
        )

        if on_token:
            on_token(result)

        if on_log:
            elapsed = time.monotonic() - t0
            on_log(f"[mlx-lm] Done in {elapsed:.1f}s")

        return result

    # ------------------------------------------------------------------
    def _try_stream(self, prompt, image, on_token, on_log, t0):
        """Attempt streaming generation; return None if unsupported."""
        try:
            from mlx_lm import stream_generate
        except ImportError:
            return None

        if on_log:
            on_log(
                f"[mlx-lm] Streaming inference ({self.model_name})…"
            )

        try:
            tokens = []
            first_token_time = None
            for chunk in stream_generate(
                self._model,
                self._processor,
                prompt=prompt,
                image=image,
                max_tokens=2048,
            ):
                text = chunk.text if hasattr(chunk, "text") else str(chunk)
                if not text:
                    continue
                if first_token_time is None:
                    first_token_time = time.monotonic()
                    if on_log:
                        on_log(
                            f"[mlx-lm] First token after "
                            f"{first_token_time - t0:.1f}s"
                        )
                tokens.append(text)
                if on_token:
                    on_token(text)

            if on_log:
                elapsed = time.monotonic() - t0
                on_log(
                    f"[mlx-lm] Done — {len(tokens)} tokens "
                    f"in {elapsed:.1f}s"
                )
            return "".join(tokens)
        except TypeError:
            # stream_generate doesn't accept `image` — fall back
            if on_log:
                on_log(
                    "[mlx-lm] stream_generate does not support "
                    "images, falling back to generate"
                )
            return None


def get_available_backend(
    ollama_model="llama3.2-vision",
    mlx_model="mlx-community/GLM-4.5V-9B-4bit",
    ollama_url="http://localhost:11434",
):
    """Return the first available backend, preferring Ollama."""
    ollama = OllamaBackend(model_name=ollama_model, base_url=ollama_url)
    if ollama.is_available():
        return ollama

    mlx = MlxLmBackend(model_name=mlx_model)
    if mlx.is_available():
        return mlx

    return None
