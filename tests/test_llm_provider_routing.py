import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services import llm_service  # noqa: E402


def test_minimax_model_does_not_hijack_an_explicit_ollama_provider():
    assert llm_service.normalize_minimax_chat_routing("MiniMax-M3", "ollama", "http://127.0.0.1:11434") == (
        "ollama", "http://127.0.0.1:11434")
    assert llm_service.normalize_minimax_chat_routing("MiniMax-M3", "local", "")[0] == "minimax"


def test_hosted_provider_without_key_fails_clearly(monkeypatch):
    monkeypatch.setattr(llm_service, "_provider", "minimax")
    monkeypatch.setattr(llm_service, "_api_key", "")
    with pytest.raises(RuntimeError, match="Settings → Services"):
        llm_service._api_headers()
    monkeypatch.setattr(llm_service, "_provider", "ollama")
    assert "Authorization" not in llm_service._api_headers()


def test_text_only_ollama_model_does_not_receive_images(monkeypatch):
    class Response:
        def __init__(self, capabilities):
            self.capabilities = capabilities

        def raise_for_status(self):
            pass

        def json(self):
            return {"capabilities": self.capabilities}

    llm_service._OLLAMA_VISION_CACHE.clear()
    monkeypatch.setattr(llm_service, "_provider", "ollama")
    monkeypatch.setattr(llm_service, "_remote_url", "http://127.0.0.1:11434")
    monkeypatch.setattr(llm_service, "_model_id", "qwen3:14b")
    monkeypatch.setattr(llm_service.requests, "post", lambda *a, **k: Response(["completion", "thinking"]))
    assert llm_service._remote_accepts_images() is False
    llm_service._OLLAMA_VISION_CACHE.clear()
    monkeypatch.setattr(llm_service, "_model_id", "qwen2.5vl:7b")
    monkeypatch.setattr(llm_service.requests, "post", lambda *a, **k: Response(["completion", "vision"]))
    assert llm_service._remote_accepts_images() is True
    monkeypatch.setattr(llm_service, "_provider", "minimax")
    assert llm_service._remote_accepts_images() is True
