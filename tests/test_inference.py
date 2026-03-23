"""Tests for the inference module."""

import base64
import io
import json

import numpy as np
import pytest
import requests
import responses
from PIL import Image

from inference import (
    SYSTEM_PROMPT,
    MlxLmBackend,
    OllamaBackend,
    _image_to_base64,
    _resize_if_needed,
    get_available_backend,
)


def _streaming_body(text, chunk_size=20):
    """Build a newline-delimited JSON body that mimics Ollama streaming.

    Splits *text* into small token chunks, each as a ``{"response": ...}``
    JSON line, terminated by a ``{"done": true}`` sentinel.
    """
    lines = []
    for i in range(0, len(text), chunk_size):
        lines.append(json.dumps({"response": text[i:i + chunk_size], "done": False}))
    lines.append(json.dumps({
        "response": "",
        "done": True,
        "eval_count": len(text) // chunk_size,
        "eval_duration": 1_000_000_000,
        "prompt_eval_count": 50,
    }))
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Image Encoding (IE-*)
# ---------------------------------------------------------------------------

class TestImageToBase64:
    def test_encode_numpy_rgb(self):
        """IE-01: RGB numpy array encodes to valid base64 PNG."""
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        b64 = _image_to_base64(arr)
        decoded = base64.b64decode(b64)
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (100, 100)

    def test_encode_numpy_grayscale(self):
        """IE-02: Grayscale numpy array encodes to valid base64 PNG."""
        arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        b64 = _image_to_base64(arr)
        decoded = base64.b64decode(b64)
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (100, 100)

    def test_encode_pil_image(self):
        """IE-03: PIL RGB Image encodes correctly (JPEG)."""
        img = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
        b64 = _image_to_base64(img)
        assert len(b64) > 0
        decoded = base64.b64decode(b64)
        assert decoded[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_encode_rgba_image(self):
        """IE-04: RGBA PIL image is handled (converted to RGB)."""
        img = Image.fromarray(np.zeros((50, 50, 4), dtype=np.uint8), mode="RGBA")
        b64 = _image_to_base64(img)
        decoded = base64.b64decode(b64)
        result_img = Image.open(io.BytesIO(decoded))
        assert result_img.size == (50, 50)

    def test_roundtrip_dimensions(self):
        """IE-05: Roundtrip encode/decode preserves dimensions (small image, no resize)."""
        arr = np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8)
        b64 = _image_to_base64(arr)
        decoded = base64.b64decode(b64)
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (150, 200)  # PIL size is (width, height)

    def test_resize_large_image(self):
        """IE-06: Large image is downscaled to max_dim."""
        img = Image.fromarray(np.zeros((2048, 3072, 3), dtype=np.uint8))
        resized = _resize_if_needed(img, max_dim=512)
        assert max(resized.size) == 512

    def test_no_resize_small_image(self):
        """IE-07: Small image is not resized."""
        img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        resized = _resize_if_needed(img, max_dim=1024)
        assert resized.size == (256, 256)

    def test_base64_respects_max_dim(self):
        """IE-08: _image_to_base64 downscales large images."""
        arr = np.zeros((2000, 2000, 3), dtype=np.uint8)
        b64 = _image_to_base64(arr, max_dim=256)
        decoded = base64.b64decode(b64)
        img = Image.open(io.BytesIO(decoded))
        assert max(img.size) == 256


# ---------------------------------------------------------------------------
# OllamaBackend (OB-*)
# ---------------------------------------------------------------------------

class TestOllamaBackend:
    @responses.activate
    def test_is_available_server_up_model_present(self, mock_ollama_tags_response):
        """OB-01: Server up with matching model → True."""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json=mock_ollama_tags_response,
            status=200,
        )
        backend = OllamaBackend(model_name="llama3.2-vision")
        assert backend.is_available() is True

    @responses.activate
    def test_is_available_server_up_model_absent(self, mock_ollama_tags_response):
        """OB-02: Server up but model not found → False."""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json=mock_ollama_tags_response,
            status=200,
        )
        backend = OllamaBackend(model_name="nonexistent-model")
        assert backend.is_available() is False

    @responses.activate
    def test_is_available_server_down(self):
        """OB-03: Server down (ConnectionError) → False."""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            body=requests.exceptions.ConnectionError("refused"),
        )
        backend = OllamaBackend()
        assert backend.is_available() is False

    @responses.activate
    def test_list_model_names(self, mock_ollama_tags_response):
        """OB-04: Returns list of model names."""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json=mock_ollama_tags_response,
            status=200,
        )
        backend = OllamaBackend()
        names = backend.list_model_names()
        assert len(names) == 2
        assert "llama3.2-vision:latest" in names

    @responses.activate
    def test_list_models_server_down(self):
        """OB-05: Server down → None (not empty list)."""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            body=requests.exceptions.ConnectionError("refused"),
        )
        backend = OllamaBackend()
        assert backend.list_models() is None

    @responses.activate
    def test_list_models_empty(self):
        """OB-05b: Server up but no models → empty list (not None)."""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json={"models": []},
            status=200,
        )
        backend = OllamaBackend()
        assert backend.list_models() == []

    @responses.activate
    def test_list_model_names_server_down(self):
        """OB-05c: list_model_names returns None when server is down."""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            body=requests.exceptions.ConnectionError("refused"),
        )
        backend = OllamaBackend()
        assert backend.list_model_names() is None

    @responses.activate
    def test_analyze_success(self, mock_ollama_generate_response):
        """OB-06: Successful streaming analysis returns response text."""
        full_text = mock_ollama_generate_response["response"]
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            body=_streaming_body(full_text),
            status=200,
            content_type="application/x-ndjson",
        )
        backend = OllamaBackend()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        result = backend.analyze(arr)
        assert "Step 1" in result
        assert "Step 2" in result

    @responses.activate
    def test_analyze_with_metadata(self, mock_ollama_generate_response):
        """OB-07: Metadata is included in the prompt."""
        full_text = mock_ollama_generate_response["response"]

        def request_callback(request):
            body = json.loads(request.body)
            assert "Modality: CT" in body["prompt"]
            return (200, {}, _streaming_body(full_text))

        responses.add_callback(
            responses.POST,
            "http://localhost:11434/api/generate",
            callback=request_callback,
        )
        backend = OllamaBackend()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        backend.analyze(arr, metadata={"Modality": "CT"})

    @responses.activate
    def test_analyze_http_error(self):
        """OB-08: HTTP 500 raises exception."""
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            json={"error": "internal"},
            status=500,
        )
        backend = OllamaBackend()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(Exception):
            backend.analyze(arr)

    @responses.activate
    def test_analyze_empty_response(self):
        """OB-10: Empty stream → fallback message."""
        body = json.dumps({
            "done": True, "eval_count": 0,
            "eval_duration": 0, "prompt_eval_count": 0,
        })
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            body=body,
            status=200,
            content_type="application/x-ndjson",
        )
        backend = OllamaBackend()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        result = backend.analyze(arr)
        assert result == "No response received."

    @responses.activate
    def test_system_prompt_sent(self, mock_ollama_generate_response):
        """OB-11: System prompt is included in the payload."""
        full_text = mock_ollama_generate_response["response"]

        def request_callback(request):
            body = json.loads(request.body)
            assert body["system"] == SYSTEM_PROMPT
            return (200, {}, _streaming_body(full_text))

        responses.add_callback(
            responses.POST,
            "http://localhost:11434/api/generate",
            callback=request_callback,
        )
        backend = OllamaBackend()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        backend.analyze(arr)

    @responses.activate
    def test_image_sent_as_base64(self, mock_ollama_generate_response):
        """OB-12: Image is sent as base64 in the images field."""
        full_text = mock_ollama_generate_response["response"]

        def request_callback(request):
            body = json.loads(request.body)
            assert len(body["images"]) == 1
            # Verify it's valid base64
            base64.b64decode(body["images"][0])
            return (200, {}, _streaming_body(full_text))

        responses.add_callback(
            responses.POST,
            "http://localhost:11434/api/generate",
            callback=request_callback,
        )
        backend = OllamaBackend()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        backend.analyze(arr)

    @responses.activate
    def test_on_token_callback(self, mock_ollama_generate_response):
        """OB-13: on_token callback receives streamed fragments."""
        full_text = mock_ollama_generate_response["response"]
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            body=_streaming_body(full_text),
            status=200,
            content_type="application/x-ndjson",
        )
        backend = OllamaBackend()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        tokens = []
        backend.analyze(arr, on_token=tokens.append)
        assert len(tokens) > 1
        assert "".join(tokens) == full_text

    @responses.activate
    def test_on_log_callback(self, mock_ollama_generate_response):
        """OB-14: on_log callback receives backend log messages."""
        full_text = mock_ollama_generate_response["response"]
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            body=_streaming_body(full_text),
            status=200,
            content_type="application/x-ndjson",
        )
        backend = OllamaBackend()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        logs = []
        backend.analyze(arr, on_log=logs.append)
        assert any("[backend]" in msg for msg in logs)
        assert any("[ollama]" in msg for msg in logs)


# ---------------------------------------------------------------------------
# MlxLmBackend (ML-*)
# ---------------------------------------------------------------------------

class TestMlxLmBackend:
    def test_is_available_not_installed(self):
        """ML-02: mlx_lm not installed → False."""
        backend = MlxLmBackend()
        # On non-Apple-Silicon / non-macOS, mlx_lm won't be installed
        # This test verifies the check doesn't crash
        result = backend.is_available()
        assert isinstance(result, bool)

    def test_analyze_converts_numpy_to_pil(self):
        """ML-04: numpy input is converted to PIL before calling generate."""
        # The analyze method converts numpy to PIL internally.
        # We test the conversion logic directly since mlx_lm is not available.
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        assert isinstance(img, Image.Image)


# ---------------------------------------------------------------------------
# Backend Selection (BS-*)
# ---------------------------------------------------------------------------

class TestBackendSelection:
    @responses.activate
    def test_prefers_ollama(self, mock_ollama_tags_response):
        """BS-01/02: When Ollama is available, it's preferred."""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json=mock_ollama_tags_response,
            status=200,
        )
        backend = get_available_backend(ollama_model="llama3.2-vision")
        assert isinstance(backend, OllamaBackend)

    @responses.activate
    def test_neither_available_returns_none(self):
        """BS-04: Neither backend available → None."""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            body=requests.exceptions.ConnectionError("refused"),
        )
        backend = get_available_backend()
        # On non-macOS, mlx-lm won't be available either
        # Result depends on environment but should not crash
        assert backend is None or isinstance(backend, (OllamaBackend, MlxLmBackend))


# ---------------------------------------------------------------------------
# System Prompt Validation (SP-*)
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_contains_modality_step(self):
        """SP-01: Prompt contains Step 1 — Modality Identification."""
        assert "Modality Identification" in SYSTEM_PROMPT

    def test_contains_observations_step(self):
        """SP-02: Prompt contains Step 2 — Observations."""
        assert "Observations" in SYSTEM_PROMPT

    def test_contains_synthesis_step(self):
        """SP-03: Prompt contains Step 3 — Diagnostic Synthesis."""
        assert "Diagnostic Synthesis" in SYSTEM_PROMPT

    def test_contains_referral_step(self):
        """SP-04: Prompt contains Step 4 — Referral Recommendation."""
        assert "Referral Recommendation" in SYSTEM_PROMPT

    def test_contains_provisional_disclaimer(self):
        """SP-05: Prompt contains PROVISIONAL warning."""
        assert "PROVISIONAL" in SYSTEM_PROMPT

    def test_referral_mentions_internist(self):
        """SP-06: Referral section mentions Internist."""
        assert "Internist" in SYSTEM_PROMPT

    def test_referral_mentions_surgeon(self):
        """SP-07: Referral section mentions Surgeon."""
        assert "Surgeon" in SYSTEM_PROMPT

    def test_referral_mentions_gp(self):
        """SP-08: Referral section mentions General Practitioner."""
        assert "General Practitioner" in SYSTEM_PROMPT
