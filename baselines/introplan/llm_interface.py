"""
Claude / OpenAI API interface for the IntroPlan baseline.

Supports two providers selectable via api_type=:
  "anthropic" — Claude (claude-sonnet-4-6, etc.)  requires ANTHROPIC_API_KEY
  "openai"    — OpenAI (gpt-4o, etc.)             requires OPENAI_API_KEY

Why we support both:
  - IntroPlan's original paper uses OpenAI. Its logit_bias trick was deprecated in
    March 2024, but stated-confidence JSON output works fine with any provider.
  - Claude avoids the deprecation issue but doesn't expose token logprobs.
  - Both are equally valid for our stated-confidence conformal prediction variant.

All responses are expected in JSON format. JSON parsing errors are retried once.
"""

import base64
import json
import os
import time
from io import BytesIO
from typing import Dict, Optional

import anthropic
from PIL import Image

try:
    import openai as _openai_module
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0


class LLMInterface:
    """
    Thin wrapper around Claude or OpenAI for IntroPlan inference.

    Handles:
      - Text-only prompts (reasoning + prediction)
      - Vision prompts (image description for calibration data)
      - JSON parsing with one automatic retry on parse failure
      - Rate-limit backoff (up to 3 attempts with exponential delay)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        api_type: str = "anthropic",
    ):
        """
        Args:
            api_key:     API key for the chosen provider. Falls back to the
                         corresponding env var (ANTHROPIC_API_KEY or OPENAI_API_KEY).
            model:       Model ID (e.g. "gpt-4o" or "claude-sonnet-4-6").
            max_tokens:  Maximum tokens in the response.
            temperature: Sampling temperature (0.0 = deterministic).
            api_type:    "anthropic" or "openai".
        """
        self._api_type = api_type

        if api_type == "openai":
            if not _OPENAI_AVAILABLE:
                raise ImportError(
                    "Install openai to use api_type='openai': pip install openai"
                )
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY in your .env file "
                    "or pass api_key= to LLMInterface()."
                )
            self._client = _openai_module.OpenAI(api_key=key)
        else:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "Anthropic API key required. Set ANTHROPIC_API_KEY in your .env file "
                    "or pass api_key= to LLMInterface()."
                )
            self._client = anthropic.Anthropic(api_key=key)

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._call_count = 0

    def predict_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        retry_on_parse_error: bool = True,
    ) -> Dict:
        """
        Send a prompt and parse the response as JSON.

        Args:
            prompt:               User-turn prompt text.
            system:               Optional system prompt.
            retry_on_parse_error: If True, retry once with an explicit JSON reminder.

        Returns:
            Parsed JSON dict.

        Raises:
            ValueError: If JSON parsing fails after the optional retry.
        """
        response_text = self._call_text(prompt, system)
        try:
            return _extract_json(response_text)
        except (json.JSONDecodeError, ValueError):
            if not retry_on_parse_error:
                raise ValueError(f"LLM returned non-JSON response:\n{response_text}")
            retry_prompt = (
                f"{prompt}\n\n"
                "IMPORTANT: Your previous response was not valid JSON. "
                "Respond ONLY with a valid JSON object, no other text."
            )
            response_text = self._call_text(retry_prompt, system)
            return _extract_json(response_text)

    def describe_image(
        self,
        image: Image.Image,
        prompt: str,
    ) -> str:
        """
        Send an image + text prompt and return the text description.

        Args:
            image:  PIL Image (RGB).
            prompt: Text instruction for what to describe.

        Returns:
            Model's text description of the image.
        """
        if self._api_type == "openai":
            return self._describe_image_openai(image, prompt)
        return self._describe_image_anthropic(image, prompt)

    @property
    def total_calls(self) -> int:
        """Total number of API calls made by this instance."""
        return self._call_count

    def _call_text(self, prompt: str, system: Optional[str] = None) -> str:
        """Make a text-only API call with exponential rate-limit retry."""
        if self._api_type == "openai":
            return self._call_text_openai(prompt, system)
        return self._call_text_anthropic(prompt, system)

    # ── Anthropic implementation ───────────────────────────────────────────────

    def _call_text_anthropic(self, prompt: str, system: Optional[str]) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        for attempt in range(3):
            try:
                response = self._client.messages.create(**kwargs)
                self._call_count += 1
                return response.content[0].text.strip()
            except anthropic.RateLimitError:
                if attempt == 2:
                    raise
                time.sleep(5 * (2 ** attempt))

        raise RuntimeError("API call failed after 3 attempts")

    def _describe_image_anthropic(self, image: Image.Image, prompt: str) -> str:
        image_b64 = _pil_to_base64(image)
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        self._call_count += 1
        return message.content[0].text.strip()

    # ── OpenAI implementation ──────────────────────────────────────────────────

    def _call_text_openai(self, prompt: str, system: Optional[str]) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=messages,
                )
                self._call_count += 1
                return response.choices[0].message.content.strip()
            except _openai_module.RateLimitError:
                if attempt == 2:
                    raise
                time.sleep(5 * (2 ** attempt))

        raise RuntimeError("API call failed after 3 attempts")

    def _describe_image_openai(self, image: Image.Image, prompt: str) -> str:
        image_b64 = _pil_to_base64(image)
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        self._call_count += 1
        return response.choices[0].message.content.strip()


# ── Helper utilities ──────────────────────────────────────────────────────────

def _extract_json(text: str) -> Dict:
    """
    Extract a JSON object from the model's response.

    Handles both raw JSON and markdown-fenced JSON (```json ... ```).
    """
    text = text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        inner_lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner_lines).strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in text: {text[:200]}")

    return json.loads(text[start:end])


def _pil_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded JPEG string."""
    buf = BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
