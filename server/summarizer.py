from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any

from openai import BadRequestError, OpenAI
from .core.logging import emit_container_log
from .langfuse_observer import LangfuseObserver


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SummaryResult:
    text: str
    model: str
    chunk_count: int = 1
    reduced: bool = False


@dataclass(slots=True)
class ProofreadResult:
    text: str
    model: str
    chunk_count: int = 1
    reduced: bool = False


class OpenAISummarizer:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        model: str,
        temperature: float,
        summary_system_prompt: str = "",
        summary_prompt_template: str = "",
        proofread_system_prompt: str = "",
        proofread_prompt_template: str = "",
        observer: LangfuseObserver | None = None,
    ):
        if not api_key:
            raise RuntimeError("SUMMARY_API_KEY (or ASR_API_KEY / OPENAI_API_KEY) is not set")

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.summary_system_prompt = summary_system_prompt.strip()
        self.summary_prompt_template = summary_prompt_template.strip()
        self.proofread_system_prompt = proofread_system_prompt.strip()
        self.proofread_prompt_template = proofread_prompt_template.strip()
        self.observer = observer

    def summarize(
        self,
        *,
        text: str,
        language: str,
        custom_template: str = "",
        trace_context: dict[str, str] | None = None,
    ) -> SummaryResult:
        clean_text = (text or "").strip()
        if not clean_text:
            return SummaryResult(text="", model=self.model)

        return self._summarize_once(clean_text, language, custom_template=custom_template, trace_context=trace_context)

    def summarize_long(
        self,
        *,
        text: str,
        language: str,
        max_chars: int,
        custom_template: str = "",
        trace_context: dict[str, str] | None = None,
    ) -> SummaryResult:
        clean_text = (text or "").strip()
        if not clean_text:
            return SummaryResult(text="", model=self.model)

        chunks = _split_text_into_chunks(clean_text, max_chars=max_chars)
        if len(chunks) == 1:
            return self._summarize_once(
                clean_text,
                language,
                custom_template=custom_template,
                trace_context=trace_context,
            )

        partials = [
            self._summarize_once(
                chunk,
                language,
                custom_template=custom_template,
                trace_context=trace_context,
            ).text
            for chunk in chunks
        ]
        reduced = False

        while len(partials) > 1:
            reduced = True
            groups = _group_texts_for_reduce(partials, max_chars=max_chars)
            partials = [self._reduce_summaries(group, language, trace_context=trace_context).text for group in groups]

        return SummaryResult(
            text=partials[0].strip(),
            model=self.model,
            chunk_count=len(chunks),
            reduced=reduced,
        )

    def proofread(
        self,
        *,
        text: str,
        language: str,
        mode: str = "proofread",
        trace_context: dict[str, str] | None = None,
    ) -> ProofreadResult:
        clean_text = _normalize_proofread_source_text((text or "").strip())
        if not clean_text:
            return ProofreadResult(text="", model=self.model)

        return self._proofread_once(clean_text, language, mode=mode, trace_context=trace_context)

    def proofread_long(
        self,
        *,
        text: str,
        language: str,
        max_chars: int,
        mode: str = "proofread",
        trace_context: dict[str, str] | None = None,
    ) -> ProofreadResult:
        clean_text = _normalize_proofread_source_text((text or "").strip())
        if not clean_text:
            return ProofreadResult(text="", model=self.model)

        chunk_chars = _effective_proofread_chunk_chars(max_chars)
        chunks = _split_text_into_chunks(clean_text, max_chars=chunk_chars)
        if len(chunks) == 1:
            return self._proofread_once(clean_text, language, mode=mode, trace_context=trace_context)

        corrected_chunks: list[str] = []
        trailing_context = ""
        for chunk in chunks:
            corrected = self._proofread_once(
                chunk,
                language,
                mode=mode,
                trailing_context=trailing_context,
                trace_context=trace_context,
            ).text.strip()
            if corrected:
                corrected_chunks.append(corrected)
                trailing_context = corrected[-800:]

        merged = _normalize_proofread_source_text("\n\n".join(part for part in corrected_chunks if part).strip())
        reduced = False
        if merged and len(merged) <= chunk_chars:
            merged = self._proofread_consistency_pass(
                merged,
                language,
                mode=mode,
                trace_context=trace_context,
            ).text.strip()
            reduced = True

        return ProofreadResult(
            text=merged,
            model=self.model,
            chunk_count=len(chunks),
            reduced=reduced,
        )

    def proofread_stream_long(
        self,
        *,
        text: str,
        language: str,
        max_chars: int,
        mode: str = "proofread",
        trace_context: dict[str, str] | None = None,
    ):
        clean_text = _normalize_proofread_source_text((text or "").strip())
        if not clean_text:
            yield {"type": "done", "model": self.model, "chunkCount": 0}
            return

        chunk_chars = _effective_proofread_chunk_chars(max_chars)
        chunks = _split_text_into_chunks(clean_text, max_chars=chunk_chars)
        yield {"type": "start", "model": self.model, "chunkCount": len(chunks)}

        trailing_context = ""
        for index, chunk in enumerate(chunks, start=1):
            yield {"type": "chunk_start", "chunkIndex": index, "chunkCount": len(chunks)}
            if index > 1:
                yield {"type": "delta", "delta": "\n\n", "chunkIndex": index, "chunkCount": len(chunks)}

            assembled_parts: list[str] = []
            for delta in self._proofread_stream_chunk(
                chunk,
                language,
                mode=mode,
                trailing_context=trailing_context,
                trace_context=trace_context,
            ):
                if delta:
                    assembled_parts.append(delta)
                    yield {"type": "delta", "delta": delta, "chunkIndex": index, "chunkCount": len(chunks)}

            corrected = "".join(assembled_parts).strip()
            if corrected:
                trailing_context = corrected[-800:]
            yield {"type": "chunk_done", "chunkIndex": index, "chunkCount": len(chunks)}

        yield {"type": "done", "model": self.model, "chunkCount": len(chunks)}

    def _summarize_once(
        self,
        text: str,
        language: str,
        *,
        custom_template: str = "",
        trace_context: dict[str, str] | None = None,
    ) -> SummaryResult:
        messages = [
            {
                "role": "system",
                "content": (
                    self.summary_system_prompt
                    or "You are a careful transcription summarizer. "
                    "Keep factual consistency with the source and do not hallucinate."
                ),
            },
            {
                "role": "user",
                "content": _build_summary_prompt(
                    text,
                    language,
                    custom_template=custom_template or self.summary_prompt_template,
                ),
            },
        ]
        with self._generation(
            name="summary.generate",
            input={"language": language, "text": text, "customTemplate": bool(custom_template)},
            model_parameters={"temperature": self.temperature},
            trace_context=trace_context,
        ) as generation:
            response = self._create_chat_completion(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
            )
            result_text = _extract_text(response).strip()
            generation.update(
                output={"text": result_text, "chars": len(result_text)},
                usage_details=_extract_usage_details(response),
            )
            return SummaryResult(
                text=result_text,
                model=_extract_model_name(response, self.model),
            )

    def _reduce_summaries(
        self,
        summaries: list[str],
        language: str,
        trace_context: dict[str, str] | None = None,
    ) -> SummaryResult:
        messages = [
            {
                "role": "system",
                "content": (
                    self.summary_system_prompt
                    or "You are a careful transcription summarizer. "
                    "Keep factual consistency with the source and do not hallucinate."
                ),
            },
            {
                "role": "user",
                "content": _build_summary_reduce_prompt(summaries, language),
            },
        ]
        with self._generation(
            name="summary.reduce",
            input={"language": language, "parts": summaries},
            model_parameters={"temperature": self.temperature},
            trace_context=trace_context,
        ) as generation:
            response = self._create_chat_completion(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
            )
            result_text = _extract_text(response).strip()
            generation.update(
                output={"text": result_text, "chars": len(result_text)},
                usage_details=_extract_usage_details(response),
            )
            return SummaryResult(
                text=result_text,
                model=_extract_model_name(response, self.model),
            )

    def _proofread_once(
        self,
        text: str,
        language: str,
        *,
        mode: str = "proofread",
        trailing_context: str = "",
        trace_context: dict[str, str] | None = None,
    ) -> ProofreadResult:
        messages = [
            {
                "role": "system",
                "content": (
                    _build_proofread_system_prompt(
                        mode=mode,
                        custom_system_prompt=self.proofread_system_prompt,
                    )
                ),
            },
            {
                "role": "user",
                "content": _build_proofread_prompt(
                    text,
                    language,
                    mode=mode,
                    custom_template=self.proofread_prompt_template,
                    trailing_context=trailing_context,
                ),
            },
        ]

        with self._generation(
            name=f"proofread.{mode}",
            input={"language": language, "text": text, "hasContext": bool(trailing_context)},
            model_parameters={"temperature": self.temperature},
            trace_context=trace_context,
        ) as generation:
            response = self._create_chat_completion(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
            )
            result_text = _extract_text(response).strip()
            generation.update(
                output={"text": result_text, "chars": len(result_text)},
                usage_details=_extract_usage_details(response),
            )

            return ProofreadResult(
                text=result_text,
                model=_extract_model_name(response, self.model),
            )

    def _proofread_stream_chunk(
        self,
        text: str,
        language: str,
        *,
        mode: str = "proofread",
        trailing_context: str = "",
        trace_context: dict[str, str] | None = None,
    ):
        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        _build_proofread_system_prompt(
                            mode=mode,
                            custom_system_prompt=self.proofread_system_prompt,
                        )
                    ),
                },
                {
                    "role": "user",
                    "content": _build_proofread_prompt(
                        text,
                        language,
                        mode=mode,
                        custom_template=self.proofread_prompt_template,
                        trailing_context=trailing_context,
                    ),
                },
            ],
            "temperature": self.temperature,
            "stream": True,
        }

        parts: list[str] = []
        for event in self._create_chat_completion_stream(request_payload):
            delta = _extract_stream_delta_text(event)
            if delta:
                parts.append(delta)
                yield delta

        final_text = "".join(parts).strip()
        with self._generation(
            name=f"proofread.stream.{mode}",
            input={"language": language, "text": text, "hasContext": bool(trailing_context)},
            output={"text": final_text, "chars": len(final_text)},
            model_parameters={"temperature": self.temperature, "stream": True},
            trace_context=trace_context,
        ):
            pass

    def _proofread_consistency_pass(
        self,
        text: str,
        language: str,
        *,
        mode: str = "proofread",
        trace_context: dict[str, str] | None = None,
    ) -> ProofreadResult:
        messages = [
            {
                "role": "system",
                "content": (
                    _build_proofread_system_prompt(
                        mode=mode,
                        custom_system_prompt=self.proofread_system_prompt,
                    )
                ),
            },
            {
                "role": "user",
                "content": _build_proofread_consistency_prompt(text, language, mode=mode),
            },
        ]
        with self._generation(
            name=f"proofread.consistency.{mode}",
            input={"language": language, "text": text},
            model_parameters={"temperature": self.temperature},
            trace_context=trace_context,
        ) as generation:
            response = self._create_chat_completion(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
            )
            result_text = _extract_text(response).strip()
            generation.update(
                output={"text": result_text, "chars": len(result_text)},
                usage_details=_extract_usage_details(response),
            )
            return ProofreadResult(
                text=result_text,
                model=_extract_model_name(response, self.model),
            )

    def _generation(
        self,
        *,
        name: str,
        input: Any,
        output: Any = None,
        model_parameters: dict[str, Any] | None = None,
        trace_context: dict[str, str] | None = None,
    ):
        if self.observer is None:
            return _noop_generation()
        return self.observer.generation(
            name=name,
            model=self.model,
            input=input,
            output=output,
            model_parameters=model_parameters or {},
            trace_context=trace_context,
        )

    def _create_chat_completion(
        self,
        *,
        model: str,
        temperature: float,
        messages: list[dict[str, str]],
    ) -> Any:
        prompt_chars = sum(len(str(message.get("content") or "")) for message in messages)
        request_payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        try:
            emit_container_log(__name__, "info", "LLM POST /v1/chat/completions: model=%s messages=%d chars=%d temp=%s stream=false", model, len(messages), prompt_chars, temperature)
            logger.info(
                "LLM POST /v1/chat/completions: model=%s messages=%d chars=%d temp=%s stream=false",
                model,
                len(messages),
                prompt_chars,
                temperature,
            )
            return self.client.chat.completions.create(**request_payload)
        except BadRequestError as exc:
            if not _is_temperature_unsupported_error(exc):
                raise
            request_payload.pop("temperature", None)
            emit_container_log(__name__, "info", "LLM POST /v1/chat/completions retry-without-temperature: model=%s messages=%d chars=%d stream=false", model, len(messages), prompt_chars)
            logger.info(
                "LLM POST /v1/chat/completions retry-without-temperature: model=%s messages=%d chars=%d stream=false",
                model,
                len(messages),
                prompt_chars,
            )
            return self.client.chat.completions.create(**request_payload)

    def _create_chat_completion_stream(self, request_payload: dict[str, Any]):
        messages = request_payload.get("messages") or []
        prompt_chars = sum(len(str(message.get("content") or "")) for message in messages if isinstance(message, dict))
        model = str(request_payload.get("model") or self.model)
        try:
            emit_container_log(__name__, "info", "LLM POST /v1/chat/completions: model=%s messages=%d chars=%d temp=%s stream=true", model, len(messages), prompt_chars, request_payload.get("temperature"))
            logger.info(
                "LLM POST /v1/chat/completions: model=%s messages=%d chars=%d temp=%s stream=true",
                model,
                len(messages),
                prompt_chars,
                request_payload.get("temperature"),
            )
            return self.client.chat.completions.create(**request_payload)
        except BadRequestError as exc:
            if not _is_temperature_unsupported_error(exc):
                raise
            request_payload = dict(request_payload)
            request_payload.pop("temperature", None)
            emit_container_log(__name__, "info", "LLM POST /v1/chat/completions retry-without-temperature: model=%s messages=%d chars=%d stream=true", model, len(messages), prompt_chars)
            logger.info(
                "LLM POST /v1/chat/completions retry-without-temperature: model=%s messages=%d chars=%d stream=true",
                model,
                len(messages),
                prompt_chars,
            )
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


def _render_prompt_template(template: str, *, text: str, language: str) -> str:
    if not template:
        return ""
    try:
        return template.format(text=text, language=language)
    except Exception:
        # Fallback: keep template robust even if braces are not escaped.
        return template.replace("{text}", text).replace("{language}", language)


def _build_summary_prompt(text: str, language: str, *, custom_template: str = "") -> str:
    rendered = _render_prompt_template(custom_template, text=text, language=language)
    if rendered.strip():
        return rendered

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


def _build_summary_reduce_prompt(summaries: list[str], language: str) -> str:
    body = "\n\n".join(
        f"[Part {index}]\n{summary.strip()}" for index, summary in enumerate(summaries, start=1) if summary.strip()
    ).strip()
    if language.lower().startswith("en"):
        return (
            "Merge the partial summaries below into one consistent English summary.\n"
            "- Remove duplication and preserve important facts.\n"
            "- Keep key decisions, action items, and open questions.\n"
            "- Use concise bullet points.\n"
            "- Add a short 1-line conclusion at the end.\n\n"
            f"Partial summaries:\n{body}"
        )

    return (
        "以下は分割された要約です。重複を除き、内容の整合を保ちながら1つの日本語要約に統合してください。\n"
        "- 事実に忠実に、推測はしない\n"
        "- 重要な決定事項・ToDo・未決事項を箇条書き\n"
        "- 最後に1行で全体要旨\n\n"
        f"分割要約:\n{body}"
    )


def _build_proofread_system_prompt(*, mode: str, custom_system_prompt: str = "") -> str:
    if mode == "proofread" and custom_system_prompt.strip():
        return custom_system_prompt.strip()

    if mode == "translate_ja":
        return (
            "You are a careful transcript translator. "
            "Translate the source text into natural Japanese without omitting facts. "
            "Preserve meaning, uncertainty, structure, and proper nouns whenever possible."
        )

    if mode == "translate_en":
        return (
            "You are a careful transcript translator. "
            "Translate the source text into natural English without omitting facts. "
            "Preserve meaning, uncertainty, structure, and proper nouns whenever possible."
        )

    return (
        "You are a strict transcript proofreader. "
        "Do not add new facts. Preserve meaning, speaker intent, and uncertainty."
    )


def _build_proofread_prompt(
    text: str,
    language: str,
    *,
    mode: str = "proofread",
    custom_template: str = "",
    trailing_context: str = "",
) -> str:
    rendered = ""
    if mode == "proofread":
        rendered = _render_prompt_template(custom_template, text=text, language=language)
    if rendered.strip():
        return rendered

    if mode == "translate_ja":
        context_prefix = ""
        if trailing_context.strip():
            context_prefix = (
                "以下は直前チャンクの翻訳文脈です。用語や文体の一貫性の参照にのみ使ってください。\n"
                "出力に再掲しないでください。\n\n"
                f"直前文脈:\n{trailing_context.strip()}\n\n"
            )
        return (
            f"{context_prefix}"
            "以下の文字起こしを自然な日本語に翻訳してください。\n"
            "- 事実や不確実性は保持する\n"
            "- 新しい情報を追加しない\n"
            "- 固有名詞は不確実なら無理に意訳しない\n"
            "- 段落や箇条書きの構造はできるだけ維持する\n"
            "- 出力は翻訳本文のみ\n\n"
            f"文字起こし:\n{text}"
        )

    if mode == "translate_en":
        context_prefix = ""
        if trailing_context.strip():
            context_prefix = (
                "Reference context from the immediately preceding translated chunk.\n"
                "Use it only to keep terminology and style consistent.\n"
                "Do not repeat it in the output.\n\n"
                f"Previous context:\n{trailing_context.strip()}\n\n"
            )
        return (
            f"{context_prefix}"
            "Translate the following transcript into natural English.\n"
            "- Preserve facts and uncertainty\n"
            "- Do not add inferred content\n"
            "- Keep proper nouns as-is when uncertain\n"
            "- Preserve paragraph or list structure when practical\n"
            "- Output only the translated text\n\n"
            f"Transcript:\n{text}"
        )

    if language.lower().startswith("en"):
        context_prefix = ""
        if trailing_context.strip():
            context_prefix = (
                "Reference context from the immediately preceding corrected chunk.\n"
                "Use it only to keep terminology and style consistent.\n"
                "Do not repeat it in the output.\n\n"
                f"Previous context:\n{trailing_context.strip()}\n\n"
            )
        return (
            f"{context_prefix}"
            "Proofread the following ASR transcript in English.\\n"
            "Rules:\\n"
            "- Keep the original meaning and uncertainty.\\n"
            "- Do not add new facts or inferred content.\\n"
            "- If adjacent lines repeat because of chunk overlap, merge them into one natural sentence.\\n"
            "- Fix obvious repetition noise and punctuation/readability.\\n"
            "- Keep proper nouns as-is when uncertain.\\n"
            "- Output only the corrected transcript text.\\n\\n"
            f"Transcript:\\n{text}"
        )

    context_prefix = ""
    if trailing_context.strip():
        context_prefix = (
            "以下は直前チャンクの校正文脈です。用語や表記の一貫性のための参照にのみ使ってください。\n"
            "出力に再掲しないでください。\n\n"
            f"直前文脈:\n{trailing_context.strip()}\n\n"
        )
    return (
        f"{context_prefix}"
        "以下は音声認識の文字起こしです。日本語として校正してください。\\n"
        "ルール:\\n"
        "- 意味は変えない。新しい情報を追加しない。\\n"
        "- チャンク overlap 由来で前後に同じ文や句がまたがっていたら、重複を1回に統合して自然につなぐ。\\n"
        "- 不自然な繰り返し・誤字・句読点を修正する。\\n"
        "- 固有名詞は不確実なら無理に変えない。\\n"
        "- 出力は校正済み本文のみ。説明は不要。\\n\\n"
        f"文字起こし:\\n{text}"
    )


def _build_proofread_consistency_prompt(text: str, language: str, *, mode: str = "proofread") -> str:
    if mode == "translate_ja":
        return (
            "以下の翻訳済み本文全体の整合を取ってください。\n"
            "- 意味は変えない\n"
            "- 用語、表記、段落の一貫性を整える\n"
            "- 不要な重複や空白のみを必要最小限で直す\n"
            "- 出力は最終本文のみ\n\n"
            f"翻訳済み本文:\n{text}"
        )

    if mode == "translate_en":
        return (
            "Unify the translated transcript below.\n"
            "- Keep the meaning unchanged\n"
            "- Standardize terminology, punctuation, and paragraph consistency\n"
            "- Remove only obvious duplication or spacing noise\n"
            "- Output only the finalized translation\n\n"
            f"Translated transcript:\n{text}"
        )

    if language.lower().startswith("en"):
        return (
            "Unify the formatting of the corrected transcript below.\n"
            "- Keep the meaning unchanged.\n"
            "- Preserve paragraph order.\n"
            "- Merge any overlap-boundary duplication into one natural sentence.\n"
            "- Standardize punctuation and spacing only where needed.\n"
            "- Output only the finalized transcript.\n\n"
            f"Corrected transcript:\n{text}"
        )

    return (
        "以下の校正済み文字起こし全体の整合を取ってください。\n"
        "- 意味は変えない\n"
        "- 段落順は維持する\n"
        "- overlap 境界由来の重複文や重複句があれば、1回に統合して自然な文章にする\n"
        "- 句読点、表記ゆれ、不要な空白だけを必要最小限で整える\n"
        "- 出力は最終本文のみ\n\n"
        f"校正済み本文:\n{text}"
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


def _extract_usage_details(response: Any) -> dict[str, int] | None:
    usage = _read_field(response, "usage")
    if usage is None:
        return None
    prompt_tokens = _read_field(usage, "prompt_tokens")
    completion_tokens = _read_field(usage, "completion_tokens")
    total_tokens = _read_field(usage, "total_tokens")
    payload: dict[str, int] = {}
    for key, value in (
        ("input", prompt_tokens),
        ("output", completion_tokens),
        ("total", total_tokens),
    ):
        try:
            if value is not None:
                payload[key] = int(value)
        except (TypeError, ValueError):
            continue
    return payload or None


def _extract_stream_delta_text(response: Any) -> str:
    choices = _read_field(response, "choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first = choices[0]
    delta = _read_field(first, "delta")
    content = _read_field(delta, "content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            value = _read_field(item, "text")
            if isinstance(value, str) and value:
                parts.append(value)
        return "".join(parts)

    return ""


def _split_text_into_chunks(text: str, *, max_chars: int) -> list[str]:
    clean = (text or "").strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    blocks = [part.strip() for part in re.split(r"\n{2,}", clean) if part.strip()]
    if not blocks:
        blocks = [clean]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for block in blocks:
        pieces = _split_block(block, max_chars=max_chars)
        for piece in pieces:
            separator_len = 2 if current_parts else 0
            if current_parts and current_len + separator_len + len(piece) > max_chars:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts = [piece]
                current_len = len(piece)
            else:
                current_parts.append(piece)
                current_len += separator_len + len(piece)

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [chunk for chunk in chunks if chunk]


def _split_block(block: str, *, max_chars: int) -> list[str]:
    clean = block.strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    sentences = [
        part.strip() for part in re.split(r"(?<=[。！？!?\.])(?:\s+|\n+)", clean) if part.strip()
    ]
    if len(sentences) <= 1:
        return _hard_wrap_text(clean, max_chars=max_chars)

    parts: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            parts.append(current.strip())
            current = sentence
            continue
        if not current and len(sentence) > max_chars:
            parts.extend(_hard_wrap_text(sentence, max_chars=max_chars))
            current = ""
            continue
        current = candidate

    if current.strip():
        parts.append(current.strip())
    return parts


def _hard_wrap_text(text: str, *, max_chars: int) -> list[str]:
    parts: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = max(window.rfind("\n"), window.rfind(" "), window.rfind("。"), window.rfind("、"), window.rfind("."))
        if cut < max_chars // 2:
            cut = max_chars
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _effective_proofread_chunk_chars(max_chars: int) -> int:
    safe_max = max(2_000, int(max_chars or 0))
    return min(safe_max, 8_000)


PROOFREAD_SPLIT_SENTENCE_RE = re.compile(r"(?<=[。！？!?\.])(?:\s+|\n+)")
PROOFREAD_NORMALIZE_DROP_RE = re.compile(r"[\s、。，．・：；！？!?,.:;()\[\]{}<>\"'「」『』]+")
PROOFREAD_MIN_OVERLAP_CHARS = 10
PROOFREAD_MAX_OVERLAP_CHARS = 72


def _normalize_proofread_source_text(text: str) -> str:
    clean = (text or "").strip()
    if not clean:
        return ""

    blocks = [part.strip() for part in re.split(r"\n{2,}", clean) if part.strip()]
    normalized_blocks = [_merge_adjacent_overlap_sentences(block) for block in blocks]
    return "\n\n".join(part for part in normalized_blocks if part).strip()


def _merge_adjacent_overlap_sentences(text: str) -> str:
    sentences = [part.strip() for part in PROOFREAD_SPLIT_SENTENCE_RE.split((text or "").strip()) if part.strip()]
    if len(sentences) < 2:
        return (text or "").strip()

    merged: list[str] = [sentences[0]]
    for sentence in sentences[1:]:
        previous = merged[-1]
        trimmed_sentence = _trim_overlap_sentence_prefix(sentence, previous)
        if not trimmed_sentence:
            continue
        merged.append(trimmed_sentence)
    return " ".join(merged).strip()


def _trim_overlap_sentence_prefix(current: str, previous: str) -> str:
    prev_normalized, _ = _normalize_proofread_compare_text(previous)
    curr_normalized, curr_index_map = _normalize_proofread_compare_text(current)
    if not prev_normalized or not curr_normalized:
        return current.strip()

    max_overlap = min(len(prev_normalized), len(curr_normalized), PROOFREAD_MAX_OVERLAP_CHARS)
    if max_overlap < PROOFREAD_MIN_OVERLAP_CHARS:
        return current.strip()

    best_overlap = 0
    for overlap_len in range(max_overlap, PROOFREAD_MIN_OVERLAP_CHARS - 1, -1):
        if prev_normalized[-overlap_len:] == curr_normalized[:overlap_len]:
            best_overlap = overlap_len
            break

    if best_overlap <= 0:
        return current.strip()

    cut_index = curr_index_map[best_overlap - 1]
    trimmed = current[cut_index:].lstrip()
    return trimmed.strip()


def _normalize_proofread_compare_text(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    index_map: list[int] = []
    for index, char in enumerate((text or "").strip()):
        if PROOFREAD_NORMALIZE_DROP_RE.fullmatch(char):
            continue
        normalized_chars.append(char)
        index_map.append(index + 1)
    return "".join(normalized_chars), index_map


def _group_texts_for_reduce(texts: list[str], *, max_chars: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for text in texts:
        clean = text.strip()
        if not clean:
            continue
        separator_len = 2 if current else 0
        if current and current_len + separator_len + len(clean) > max_chars:
            groups.append(current)
            current = [clean]
            current_len = len(clean)
            continue
        current.append(clean)
        current_len += separator_len + len(clean)

    if current:
        groups.append(current)
    return groups


def _read_field(obj: Any, field: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


@contextmanager
def _noop_generation():
    class _Noop:
        def update(self, **_: Any) -> None:
            return None

    yield _Noop()
