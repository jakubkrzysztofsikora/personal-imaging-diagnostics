"""Tests for Ollama model management features."""


import pytest
import requests
import responses

from inference import OllamaBackend

# ---------------------------------------------------------------------------
# list_models (detailed) and list_model_names
# ---------------------------------------------------------------------------

class TestListModels:
    @responses.activate
    def test_list_models_returns_full_details(self, mock_ollama_tags_response):
        """list_models returns full model dicts, not just names."""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json=mock_ollama_tags_response,
            status=200,
        )
        backend = OllamaBackend()
        models = backend.list_models()
        assert len(models) == 2
        assert "name" in models[0]
        assert "size" in models[0]

    @responses.activate
    def test_list_model_names(self, mock_ollama_tags_response):
        """list_model_names returns string list."""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json=mock_ollama_tags_response,
            status=200,
        )
        backend = OllamaBackend()
        names = backend.list_model_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert "llama3.2-vision:latest" in names

    @responses.activate
    def test_list_models_server_down(self):
        """list_models returns None when server is down."""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            body=requests.exceptions.ConnectionError("refused"),
        )
        backend = OllamaBackend()
        assert backend.list_models() is None


# ---------------------------------------------------------------------------
# pull_model
# ---------------------------------------------------------------------------

class TestPullModel:
    @responses.activate
    def test_pull_model_streaming(self):
        """pull_model yields progress updates."""
        stream_body = (
            '{"status": "pulling manifest"}\n'
            '{"status": "downloading", "completed": 500, "total": 1000}\n'
            '{"status": "success"}\n'
        )
        responses.add(
            responses.POST,
            "http://localhost:11434/api/pull",
            body=stream_body,
            status=200,
            stream=True,
        )
        backend = OllamaBackend()
        updates = list(backend.pull_model("llama3.2-vision"))
        assert len(updates) == 3
        assert updates[0]["status"] == "pulling manifest"
        assert updates[-1]["status"] == "success"

    @responses.activate
    def test_pull_model_non_streaming(self):
        """pull_model with stream=False returns single result."""
        responses.add(
            responses.POST,
            "http://localhost:11434/api/pull",
            json={"status": "success"},
            status=200,
        )
        backend = OllamaBackend()
        results = list(backend.pull_model("llama3.2-vision", stream=False))
        assert len(results) == 1
        assert results[0]["status"] == "success"

    @responses.activate
    def test_pull_model_not_found(self):
        """pull_model raises on 404 (model not in registry)."""
        responses.add(
            responses.POST,
            "http://localhost:11434/api/pull",
            json={"error": "model not found"},
            status=404,
        )
        backend = OllamaBackend()
        with pytest.raises(requests.exceptions.HTTPError):
            list(backend.pull_model("nonexistent-model"))


# ---------------------------------------------------------------------------
# delete_model
# ---------------------------------------------------------------------------

class TestDeleteModel:
    @responses.activate
    def test_delete_success(self):
        """delete_model returns True on 200."""
        responses.add(
            responses.DELETE,
            "http://localhost:11434/api/delete",
            status=200,
        )
        backend = OllamaBackend()
        assert backend.delete_model("llama3.2-vision") is True

    @responses.activate
    def test_delete_not_found(self):
        """delete_model returns False on 404."""
        responses.add(
            responses.DELETE,
            "http://localhost:11434/api/delete",
            status=404,
        )
        backend = OllamaBackend()
        assert backend.delete_model("nonexistent") is False


# ---------------------------------------------------------------------------
# show_model
# ---------------------------------------------------------------------------

class TestShowModel:
    @responses.activate
    def test_show_model_success(self):
        """show_model returns model info dict."""
        model_info = {
            "details": {
                "family": "llama",
                "parameter_size": "11B",
                "quantization_level": "Q4_0",
            },
            "template": "{{ .Prompt }}",
        }
        responses.add(
            responses.POST,
            "http://localhost:11434/api/show",
            json=model_info,
            status=200,
        )
        backend = OllamaBackend()
        info = backend.show_model("llama3.2-vision")
        assert info is not None
        assert info["details"]["family"] == "llama"
        assert info["details"]["parameter_size"] == "11B"

    @responses.activate
    def test_show_model_not_found(self):
        """show_model returns None on 404."""
        responses.add(
            responses.POST,
            "http://localhost:11434/api/show",
            status=404,
        )
        backend = OllamaBackend()
        assert backend.show_model("nonexistent") is None

    @responses.activate
    def test_show_model_server_down(self):
        """show_model returns None when server is unreachable."""
        responses.add(
            responses.POST,
            "http://localhost:11434/api/show",
            body=requests.exceptions.ConnectionError("refused"),
        )
        backend = OllamaBackend()
        assert backend.show_model("anything") is None


# ---------------------------------------------------------------------------
# list_running
# ---------------------------------------------------------------------------

class TestListRunning:
    @responses.activate
    def test_list_running_with_models(self):
        """list_running returns loaded model info."""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/ps",
            json={
                "models": [
                    {
                        "name": "llama3.2-vision:latest",
                        "size": 5137025024,
                        "size_vram": 5137025024,
                    }
                ]
            },
            status=200,
        )
        backend = OllamaBackend()
        running = backend.list_running()
        assert len(running) == 1
        assert running[0]["name"] == "llama3.2-vision:latest"

    @responses.activate
    def test_list_running_empty(self):
        """list_running returns [] when no models loaded."""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/ps",
            json={"models": []},
            status=200,
        )
        backend = OllamaBackend()
        assert backend.list_running() == []

    @responses.activate
    def test_list_running_server_down(self):
        """list_running returns [] when server is down."""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/ps",
            body=requests.exceptions.ConnectionError("refused"),
        )
        backend = OllamaBackend()
        assert backend.list_running() == []
