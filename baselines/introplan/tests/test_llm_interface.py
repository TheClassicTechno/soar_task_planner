"""
Unit tests for llm_interface.py — Anthropic client fully mocked. No real API calls.
"""

import base64
import json
from io import BytesIO
from unittest.mock import MagicMock, patch, call

import pytest
from PIL import Image

from baselines.introplan.llm_interface import (
    LLMInterface,
    _extract_json,
    _pil_to_base64,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_llm(api_key="test-key"):
    """Build an LLMInterface with a mocked Anthropic client."""
    with patch("baselines.introplan.llm_interface.anthropic.Anthropic") as MockClient:
        llm = LLMInterface(api_key=api_key)
        llm._client = MockClient.return_value
    return llm


def _mock_message(text: str):
    """Create a mock Anthropic message response with given text content."""
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _rgb_image(w=32, h=32):
    return Image.new("RGB", (w, h), color=(100, 150, 200))


# ── LLMInterface.__init__ ─────────────────────────────────────────────────────

def test_init_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        LLMInterface()


def test_init_uses_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    with patch("baselines.introplan.llm_interface.anthropic.Anthropic"):
        llm = LLMInterface()
    assert llm._call_count == 0


def test_init_uses_provided_api_key():
    with patch("baselines.introplan.llm_interface.anthropic.Anthropic") as MockClient:
        LLMInterface(api_key="my-key")
    MockClient.assert_called_once_with(api_key="my-key")


def test_init_stores_model_and_params():
    with patch("baselines.introplan.llm_interface.anthropic.Anthropic"):
        llm = LLMInterface(api_key="k", model="claude-haiku-4-5-20251001", max_tokens=512, temperature=0.5)
    assert llm.model == "claude-haiku-4-5-20251001"
    assert llm.max_tokens == 512
    assert llm.temperature == 0.5


# ── total_calls ───────────────────────────────────────────────────────────────

def test_total_calls_starts_at_zero():
    llm = _make_llm()
    assert llm.total_calls == 0


def test_total_calls_increments_after_text_call():
    llm = _make_llm()
    llm._client.messages.create.return_value = _mock_message('{"key": "val"}')
    llm.predict_json("prompt")
    assert llm.total_calls == 1


def test_total_calls_increments_after_image_call():
    llm = _make_llm()
    llm._client.messages.create.return_value = _mock_message("a grassy field")
    llm.describe_image(_rgb_image(), "Describe this terrain.")
    assert llm.total_calls == 1


# ── predict_json ──────────────────────────────────────────────────────────────

def test_predict_json_returns_parsed_dict():
    llm = _make_llm()
    llm._client.messages.create.return_value = _mock_message('{"answer": "B", "confidence": 0.9}')
    result = llm.predict_json("Which option?")
    assert result == {"answer": "B", "confidence": 0.9}


def test_predict_json_handles_markdown_fences():
    llm = _make_llm()
    wrapped = "```json\n{\"key\": \"val\"}\n```"
    llm._client.messages.create.return_value = _mock_message(wrapped)
    result = llm.predict_json("prompt")
    assert result["key"] == "val"


def test_predict_json_retries_on_parse_error():
    llm = _make_llm()
    llm._client.messages.create.side_effect = [
        _mock_message("not json at all"),
        _mock_message('{"fixed": true}'),
    ]
    result = llm.predict_json("prompt")
    assert result == {"fixed": True}
    assert llm._client.messages.create.call_count == 2


def test_predict_json_raises_on_double_failure():
    llm = _make_llm()
    llm._client.messages.create.side_effect = [
        _mock_message("bad 1"),
        _mock_message("bad 2"),
    ]
    with pytest.raises((ValueError, json.JSONDecodeError)):
        llm.predict_json("prompt")


def test_predict_json_no_retry_raises_immediately():
    llm = _make_llm()
    llm._client.messages.create.return_value = _mock_message("not json")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        llm.predict_json("prompt", retry_on_parse_error=False)
    assert llm._client.messages.create.call_count == 1


def test_predict_json_passes_system_prompt():
    llm = _make_llm()
    llm._client.messages.create.return_value = _mock_message('{"ok": 1}')
    llm.predict_json("user prompt", system="You are a navigator.")
    call_kwargs = llm._client.messages.create.call_args[1]
    assert call_kwargs.get("system") == "You are a navigator."


def test_predict_json_no_system_when_none():
    llm = _make_llm()
    llm._client.messages.create.return_value = _mock_message('{"ok": 1}')
    llm.predict_json("user prompt", system=None)
    call_kwargs = llm._client.messages.create.call_args[1]
    assert "system" not in call_kwargs


# ── describe_image ────────────────────────────────────────────────────────────

def test_describe_image_returns_text():
    llm = _make_llm()
    llm._client.messages.create.return_value = _mock_message("  gravel path ahead  ")
    result = llm.describe_image(_rgb_image(), "Describe the terrain.")
    assert result == "gravel path ahead"


def test_describe_image_sends_image_content():
    llm = _make_llm()
    llm._client.messages.create.return_value = _mock_message("terrain")
    llm.describe_image(_rgb_image(), "Describe.")
    create_call = llm._client.messages.create.call_args[1]
    content = create_call["messages"][0]["content"]
    types = [block["type"] for block in content]
    assert "image" in types
    assert "text" in types


def test_describe_image_uses_base64_source():
    llm = _make_llm()
    llm._client.messages.create.return_value = _mock_message("terrain")
    llm.describe_image(_rgb_image(), "Describe.")
    create_call = llm._client.messages.create.call_args[1]
    image_block = next(
        b for b in create_call["messages"][0]["content"] if b["type"] == "image"
    )
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/jpeg"


def test_describe_image_converts_rgba_to_rgb():
    llm = _make_llm()
    llm._client.messages.create.return_value = _mock_message("grass")
    rgba_img = Image.new("RGBA", (16, 16), (255, 0, 0, 128))
    # Should not raise — RGBA is converted to RGB internally
    llm.describe_image(rgba_img, "Describe.")


# ── _call_text rate-limit backoff ─────────────────────────────────────────────

def test_call_text_retries_on_rate_limit():
    llm = _make_llm()
    import anthropic as _anthropic
    llm._client.messages.create.side_effect = [
        _anthropic.RateLimitError("rate limit", response=MagicMock(), body={}),
        _mock_message("hello"),
    ]
    with patch("baselines.introplan.llm_interface.time.sleep"):
        result = llm._call_text("prompt")
    assert result == "hello"
    assert llm._client.messages.create.call_count == 2


def test_call_text_raises_after_three_rate_limits():
    llm = _make_llm()
    import anthropic as _anthropic
    llm._client.messages.create.side_effect = _anthropic.RateLimitError(
        "rate limit", response=MagicMock(), body={}
    )
    with patch("baselines.introplan.llm_interface.time.sleep"):
        with pytest.raises(_anthropic.RateLimitError):
            llm._call_text("prompt")
    assert llm._client.messages.create.call_count == 3


# ── _extract_json ─────────────────────────────────────────────────────────────

def test_extract_json_plain():
    result = _extract_json('{"a": 1, "b": 2}')
    assert result == {"a": 1, "b": 2}


def test_extract_json_with_fences():
    text = "```json\n{\"x\": \"y\"}\n```"
    assert _extract_json(text) == {"x": "y"}


def test_extract_json_fences_without_language_tag():
    text = "```\n{\"x\": 1}\n```"
    assert _extract_json(text) == {"x": 1}


def test_extract_json_with_surrounding_text():
    text = "Here is the result:\n{\"decision\": \"B\"}\nDone."
    assert _extract_json(text) == {"decision": "B"}


def test_extract_json_raises_on_no_braces():
    with pytest.raises(ValueError, match="No JSON"):
        _extract_json("just plain text")


def test_extract_json_raises_on_invalid_json():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        _extract_json("{bad json: here}")


# ── _pil_to_base64 ────────────────────────────────────────────────────────────

def test_pil_to_base64_returns_string():
    img = _rgb_image()
    result = _pil_to_base64(img)
    assert isinstance(result, str)


def test_pil_to_base64_is_valid_base64():
    img = _rgb_image()
    encoded = _pil_to_base64(img)
    decoded = base64.b64decode(encoded)
    # Decoded bytes should be a valid JPEG (starts with FF D8)
    assert decoded[:2] == b"\xff\xd8"


def test_pil_to_base64_converts_rgba():
    rgba_img = Image.new("RGBA", (8, 8), (100, 200, 50, 128))
    result = _pil_to_base64(rgba_img)
    assert isinstance(result, str)
    decoded = base64.b64decode(result)
    assert decoded[:2] == b"\xff\xd8"
