from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import BadRequestError, OpenAI


@dataclass(slots=True)
class SummaryResult:
    text: str
    model: str


@dataclass(slots=True)
class ProofreadResult:
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

        response = self._create_chat_completion(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful transcription summarizer. "
                        "Keep factual consistency with the source and do not hallucinate."
                    ),
                },
                {
                    "role": "user",
                    "content": _build_summary_prompt(clean_text, language),
                },
            ],
        )

        summary_text = _extract_text(response).strip()
        model_name = _extract_model_name(response, self.model)
        return SummaryResult(text=summary_text, model=model_name)

    def proofread(self, *, text: str, language: str) -> ProofreadResult:
        clean_text = (text or "").strip()
        if not clean_text:
            return ProofreadResult(text="", model=self.model)

        response = self._create_chat_completion(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict transcript proofreader. "
                        "Do not add new facts. Preserve meaning, speaker intent, and uncertainty."
                    ),
                },
                {
                    "role": "user",
                    "content": _build_proofread_prompt(clean_text, language),
                },
            ],
        )

        proofread_text = _extract_text(response).strip()
        model_name = _extract_model_name(response, self.model)
        return ProofreadResult(text=proofread_text, model=model_name)

    def _create_chat_completion(
        self,
        *,
        model: str,
        temperature: float,
        messages: list[dict[str, str]],
    ) -> Any:
        request_payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        try:
            return self.client.chat.completions.create(**request_payload)
        except BadRequestError as exc:
            if not _is_temperature_unsupported_error(exc):
                raise
            request_payload.pop("temperature", None)
            return self.client.chat.completions.create(**request_payload)


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


def _build_summary_prompt(text: str, language: str) -> str:
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


def _build_proofread_prompt(text: str, language: str) -> str:
    if language.lower().startswith("en"):
        return (
            "Proofread the following ASR transcript in English.\\n"
            "Rules:\\n"
            "- Keep the original meaning and uncertainty.\\n"
            "- Do not add new facts or inferred content.\\n"
            "- Fix obvious repetition noise and punctuation/readability.\\n"
            "- Keep proper nouns as-is when uncertain.\\n"
            "- Output only the corrected transcript text.\\n\\n"
            f"Transcript:\\n{text}"
        )

    return (
        "以下は音声認識の文字起こしです。日本語として校正してください。\\n"
        "ルール:\\n"
        "- 意味は変えない。新しい情報を追加しない。\\n"
        "- 不自然な繰り返し・誤字・句読点を修正する。\\n"
        "- 固有名詞は不確実なら無理に変えない。\\n"
        "- 出力は校正済み本文のみ。説明は不要。\\n\\n"
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

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            value = _read_field(item, "text")
            if isinstance(value, str) and value.strip():
                parts.append(value)
        return "\\n".join(parts)

    return ""


def _extract_model_name(response: Any, fallback: str) -> str:
    model_name = _read_field(response, "model")
    if isinstance(model_name, str) and model_name.strip():
        return model_name
    return fallback


def _read_field(obj: Any, field: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)
