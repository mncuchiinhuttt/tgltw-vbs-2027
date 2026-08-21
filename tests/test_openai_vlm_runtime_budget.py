"""Tests for the bounded, OpenAI-compatible VLM request contract."""

import os
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "inference-code"))
sys.path.append(REPO_ROOT)

import models.openai_vlm as openai_vlm  # noqa: E402


class RecordingCompletions:
    def __init__(self, response=None, error=None):
        self.calls = []
        self.response = response or SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        self.error = error

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None and len(self.calls) == 1:
            raise self.error
        return self.response


def _vlm(completions):
    vlm = openai_vlm.OpenAIVLM.__new__(openai_vlm.OpenAIVLM)
    vlm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    vlm.model_name = "test-model"
    return vlm


def test_text_generation_uses_bounded_completion_budget(monkeypatch):
    monkeypatch.setattr(openai_vlm, "OPENAI_VLM_MAX_COMPLETION_TOKENS", 1234)
    completions = RecordingCompletions()
    result = _vlm(completions).generate(None, "answer briefly")

    assert result == "ok"
    assert completions.calls[0]["max_completion_tokens"] == 1234
    assert "max_tokens" not in completions.calls[0]


def test_old_compatible_endpoint_gets_one_targeted_fallback(monkeypatch):
    monkeypatch.setattr(openai_vlm, "OPENAI_VLM_MAX_COMPLETION_TOKENS", 2048)
    completions = RecordingCompletions(error=TypeError("unexpected max_completion_tokens"))
    result = _vlm(completions).generate(None, "answer briefly")

    assert result == "ok"
    assert len(completions.calls) == 2
    assert completions.calls[1]["max_tokens"] == 2048
    assert "max_completion_tokens" not in completions.calls[1]


def test_unrelated_api_error_is_not_retried():
    completions = RecordingCompletions(error=RuntimeError("rate limit"))
    with pytest.raises(RuntimeError, match="rate limit"):
        _vlm(completions).generate(None, "answer briefly")
    assert len(completions.calls) == 1


def test_constructor_passes_timeout_to_openai_client(monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(openai_vlm, "OpenAI", fake_openai)
    monkeypatch.setattr(openai_vlm, "OPENAI_VLM_TIMEOUT_SEC", 17.5)
    openai_vlm.OpenAIVLM(model_name="test-model", api_key="key")

    assert captured["timeout"] == 17.5
