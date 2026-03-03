from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import BadRequestError, OpenAI


@dataclass(slots=True)
class SummaryResult:
    text: str
    model: str


class OpenAISummarizer:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        model: str,
        temperature: float,
    ):
        if not api_key:
            raise RuntimeError("SUMMARY_API_KEY (or OPENAI_API_KEY) is not set")

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature

    def summarize(self, *, text: str, language: str) -> SummaryResult:
        clean_text = (text or "").strip()
        if not clean_text:
            return SummaryResult(text="", model=self.model)

        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a careful transcription summarizer. "
                        "Keep factual consistency with the source and do not hallucinate."
                    ),
                },
                {
                    "role": "user",
                    "content": _build_user_prompt(clean_text, language),
                },
            ],
        }
        # Some newer models only accept the default temperature.
        request_payload["temperature"] = self.temperature

        try:
            response = self.client.chat.completions.create(**request_payload)
        except BadRequestError as exc:
            if not _is_temperature_unsupported_error(exc):
                raise
            request_payload.pop("temperature", None)
            response = self.client.chat.completions.create(**request_payload)

        summary_text = _extract_text(response).strip()
        model_name = _read_field(response, "model")
        if not isinstance(model_name, str) or not model_name.strip():
            model_name = self.model

        return SummaryResult(text=summary_text, model=model_name)


def _is_temperature_unsupported_error(exc: BadRequestError) -> bool:
    body = getattr(exc, "body", None)
    message = ""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            raw = error.get("message")
            if isinstance(raw, str):
                message = raw
    if not message:
        message = str(exc)

    lower = message.lower()
    return "temperature" in lower and ("only the default" in lower or "does not support" in lower)


def _build_user_prompt(text: str, language: str) -> str:
    if language.lower().startswith("en"):
        return (
            "Summarize the transcript below in English.\\n"
            "- Keep key decisions, action items, and open questions.\\n"
            "- Use concise bullet points.\\n"
            "- Add a short 1-line conclusion at the end.\\n\\n"
            f"Transcript:\\n{text}"
        )

    return (
        "以下の文字起こしを日本語で要約してください。\\n"
        "- 事実に忠実に、推測はしない\\n"
        "- 重要な決定事項・ToDo・未決事項を箇条書き\\n"
        "- 最後に1行で全体要旨\\n\\n"
        f"文字起こし:\\n{text}"
    )


def _extract_text(response: Any) -> str:
    choices = _read_field(response, "choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first = choices[0]
    message = _read_field(first, "message")
    content = _read_field(message, "content")

    if isinstance(content, str):
        return content

    # SDK variant: list of content blocks.
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            value = _read_field(item, "text")
            if isinstance(value, str) and value.strip():
                parts.append(value)
        return "\\n".join(parts)

    return ""


def _read_field(obj: Any, field: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)
