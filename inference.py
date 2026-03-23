"""
Inference Module

Provides a unified interface for local Vision-Language Model inference
via either Ollama's HTTP API or mlx-lm's Python API.
"""

import base64
import io

import numpy as np
import requests
from PIL import Image

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


def _image_to_base64(image):
    """Convert a PIL Image or numpy array to a base64-encoded PNG string."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class OllamaBackend:
    """Inference backend using Ollama's local HTTP API."""

    def __init__(self, model_name="llama3.2-vision", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

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
        """List available Ollama models with details."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                return resp.json().get("models", [])
        except requests.ConnectionError:
            pass
        return []

    def list_model_names(self):
        """List available Ollama model names (strings only)."""
        return [m["name"] for m in self.list_models()]

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
                    import json
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

    def analyze(self, image, user_prompt="Analyze this medical image.", metadata=None):
        """Send an image to Ollama for analysis."""
        img_b64 = _image_to_base64(image)

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
            "stream": False,
        }
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("response", "No response received.")


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

    def analyze(self, image, user_prompt="Analyze this medical image.", metadata=None):
        """Run inference using mlx-lm."""
        self._load_model()
        from mlx_lm import generate

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        context_parts = [user_prompt]
        if metadata:
            context_parts.append("\nDICOM Metadata Context:")
            for k, v in metadata.items():
                context_parts.append(f"  {k}: {v}")

        full_prompt = SYSTEM_PROMPT + "\n\n" + "\n".join(context_parts)

        response = generate(
            self._model,
            self._processor,
            prompt=full_prompt,
            image=image,
            max_tokens=2048,
            verbose=False,
        )
        return response


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
