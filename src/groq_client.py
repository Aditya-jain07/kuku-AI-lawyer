"""Groq chat and Whisper transcription with bounded, configurable requests."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator

from groq import Groq, APIConnectionError, APIStatusError, APITimeoutError, RateLimitError


MAX_AUDIO_BYTES = 25 * 1024 * 1024  # Application limit, not a provider-wide promise.
_QWEN_INSTRUCT_MODELS = frozenset({"qwen/qwen3.8-27b"})


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _temperature(value: Any) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 2):
        raise ValueError("temperature must be a finite number between 0 and 2")
    return float(value)


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class GroqClient:
    """Preserve GroqClient(key, llm, stt), chat() and transcribe()."""

    def __init__(
        self, api_key: str, llm_model: str, stt_model: str, *,
        timeout_seconds: float = 30.0, max_retries: int = 2,
        temperature: float = 0.1, max_tokens: int = 1024,
        reasoning_effort: str | None = None,
    ):
        if not isinstance(api_key, str) or not api_key.strip():
            raise RuntimeError("GROQ_API_KEY is missing. Configure it in your private .env file.")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be a positive finite number")
        self.llm_model = _nonempty(llm_model, "llm_model")
        self.stt_model = _nonempty(stt_model, "stt_model")
        self.temperature = _temperature(temperature)
        self.max_tokens = _positive_int(max_tokens, "max_tokens")
        if reasoning_effort not in {None, "none", "default", "low", "medium", "high"}:
            raise ValueError("Unsupported reasoning_effort")
        if reasoning_effort is not None and self.llm_model not in _QWEN_INSTRUCT_MODELS:
            raise ValueError("Reasoning overrides are configured only for the supported Qwen model")
        # Use Qwen's documented instruct mode for short RAG dialogue. Callers
        # can explicitly request a supported thinking effort when required.
        self.reasoning_effort = reasoning_effort
        self.client = Groq(
            api_key=api_key.strip(), timeout=float(timeout_seconds),
            max_retries=_positive_int(max_retries, "max_retries", minimum=0),
        )

    @contextmanager
    def _api_errors(self, operation: str) -> Iterator[None]:
        """Keep SDK retries bounded and avoid exposing request bodies or keys."""
        try:
            yield
        except RateLimitError as exc:
            raise RuntimeError(f"Groq {operation} rate limit reached. Please try again later.") from exc
        except APITimeoutError as exc:
            raise RuntimeError(f"Groq {operation} timed out. Please try again.") from exc
        except APIConnectionError as exc:
            raise RuntimeError("Could not connect to Groq. Please try again.") from exc
        except APIStatusError as exc:
            status = exc.status_code
            if status == 401:
                message = "Groq authentication failed. Check your private API-key configuration."
            elif status == 403:
                message = "Groq denied access. Check the account and selected model permissions."
            elif status == 404:
                message = "The configured Groq model or endpoint is unavailable. Check the model setting."
            elif status == 413:
                message = "Groq rejected the request size. Use a smaller input."
            elif status >= 500:
                message = "Groq is temporarily unavailable. Please try again later."
            else:
                message = f"Groq {operation} failed (HTTP {status}). Check the request and model settings."
            raise RuntimeError(message) from exc

    def chat(
        self, messages: list[dict[str, str]], temperature: float | None = None,
        *, max_tokens: int | None = None,
    ) -> str:
        """Generate final-answer text without returning a token-truncated draft.

        The token cap is separate from the service's 2–4 sentence prompt target.
        Extra JSON fields allow older SDKs to send the current API parameters.
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("Chat messages must be a nonempty list")
        prepared: list[dict[str, str]] = []
        for number, item in enumerate(messages, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Chat message {number} must be a dictionary")
            role, content = item.get("role"), item.get("content")
            if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
                raise ValueError(f"Chat message {number} has no valid role")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Chat message {number} has no text content")
            prepared.append({"role": role, "content": content})
        actual_temperature = self.temperature if temperature is None else _temperature(temperature)
        budget = self.max_tokens if max_tokens is None else _positive_int(max_tokens, "max_tokens")
        extra_body: dict[str, Any] = {"max_completion_tokens": budget}
        if self.llm_model in _QWEN_INSTRUCT_MODELS:
            extra_body.update({
                "reasoning_format": "hidden",
                "reasoning_effort": self.reasoning_effort or "none",
            })
        with self._api_errors("chat request"):
            response = self.client.chat.completions.create(
                model=self.llm_model, messages=prepared, temperature=actual_temperature,
                extra_body=extra_body,
            )
        if not response.choices:
            raise RuntimeError("Groq returned no completion choices")
        choice = response.choices[0]
        reason = getattr(choice, "finish_reason", None)
        if reason == "length":
            raise RuntimeError(
                "The answer reached the configured token limit and was withheld to avoid showing "
                "an incomplete legal response. Increase LLM_MAX_TOKENS or narrow the question."
            )
        if reason not in {None, "stop"}:
            raise RuntimeError("Groq did not return a completed text answer")
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Groq returned an empty chat response")
        return content.strip()

    def transcribe(
        self, audio_bytes: bytes, filename: str = "voice_input.wav",
        *, language: str | None = None,
    ) -> str:
        """Transcribe audio; leave language detection automatic unless specified."""
        if not isinstance(audio_bytes, (bytes, bytearray)):
            raise TypeError("Audio data must be bytes")
        if not audio_bytes:
            raise ValueError("Audio data cannot be empty")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise ValueError("Audio exceeds the 25 MiB limit used by this application")
        filename = _nonempty(filename, "filename").replace("\\", "/").rsplit("/", 1)[-1]
        if not filename or filename in {".", ".."} or "\x00" in filename:
            raise ValueError("Invalid audio filename")
        options: dict[str, Any] = {}
        if language is not None:
            if not isinstance(language, str) or not language.strip():
                raise ValueError("language must be a nonempty language code, such as en or hi")
            options["language"] = language.strip().lower()
        with self._api_errors("transcription"):
            response = self.client.audio.transcriptions.create(
                file=(filename, bytes(audio_bytes)), model=self.stt_model,
                response_format="json", **options,
            )
        transcript = getattr(response, "text", "")
        if not isinstance(transcript, str) or not transcript.strip():
            raise RuntimeError("No transcription text was returned; try clearer audio")
        return transcript.strip()

    def close(self) -> None:
        """Release HTTP connections when this client is no longer needed."""
        self.client.close()
