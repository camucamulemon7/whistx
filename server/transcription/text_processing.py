from __future__ import annotations

import difflib
import logging
import re

from ..asr import ASRChunkResult
from ..core.blocking import blocking_work_pool
from ..core.config import settings
from .session import LiveSession

logger = logging.getLogger(__name__)


def _build_prompt(session: LiveSession) -> str | None:
    parts: list[str] = []
    shared_vocabulary = str(getattr(session, "shared_vocabulary", "") or "").strip()
    language = (session.language or "").lower()
    operator_prompt = str(session.base_prompt or "").strip()
    recent_history = list(session.context_history or [])
    recent_terms = list(session.context_terms or [])

    if shared_vocabulary:
        if language.startswith("en"):
            parts.append("Shared glossary:\n" + shared_vocabulary)
        else:
            parts.append("共有用語辞典:\n" + shared_vocabulary)
    if operator_prompt:
        if language.startswith("en"):
            parts.append("Operator prompt:\n" + operator_prompt)
        else:
            parts.append("利用者プロンプト:\n" + operator_prompt)

    if session.context_prompt_enabled and (recent_history or recent_terms):
        if language.startswith("en"):
            header = "Recent transcript context:"
            terms_header = "Key terms:"
        elif not session.language:
            header = (
                "Recent transcript context. Keep the same spoken language as the audio:"
            )
            terms_header = "Key terms from recent transcript:"
        else:
            header = "直前の文字起こし文脈:"
            terms_header = "直前の重要語:"
        if recent_history:
            parts.append(f"{header}\n" + "\n".join(recent_history))
        if recent_terms:
            parts.append(f"{terms_header}\n" + ", ".join(recent_terms))

    merged = "\n\n".join(part for part in parts if part).strip()
    if session.context_max_chars > 0 and len(merged) > session.context_max_chars:
        merged = merged[-session.context_max_chars :].lstrip()
    return merged or None


def _append_context(session: LiveSession, text: str) -> None:
    if not session.context_prompt_enabled:
        return
    if session.context_max_chars <= 0:
        return

    cleaned = " ".join(text.split()).strip()
    cleaned = _sanitize_transcript_text(cleaned, language=session.language)
    if not cleaned:
        return

    session.context_history.append(cleaned)
    session.context_history = session.context_history[-session.context_recent_lines :]

    merged_terms = _merge_context_terms(
        existing=session.context_terms,
        new_terms=_extract_context_terms(cleaned),
        limit=session.context_term_limit,
    )
    session.context_terms = _trim_context_terms_to_budget(
        terms=merged_terms,
        max_chars=session.context_max_chars,
        history=session.context_history,
    )


CONTEXT_LATIN_TERM_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9.+/_-]{1,31}\b")
CONTEXT_KATAKANA_TERM_RE = re.compile(r"[ァ-ヶー]{3,}")
CONTEXT_CJK_TERM_RE = re.compile(r"[\u4e00-\u9fff]{2,12}")


def _extract_context_terms(text: str) -> list[str]:
    tokens: list[str] = []
    for pattern in (
        CONTEXT_LATIN_TERM_RE,
        CONTEXT_KATAKANA_TERM_RE,
        CONTEXT_CJK_TERM_RE,
    ):
        for match in pattern.finditer(text):
            token = match.group(0).strip(".,:;()[]{}<>\"'")
            if len(token) < 2:
                continue
            if token.isdigit():
                continue
            if token.lower() in {"recent", "transcript", "context"}:
                continue
            tokens.append(token)

    return _rank_context_terms(tokens)


def _rank_context_terms(tokens: list[str]) -> list[str]:
    ranked = sorted(
        set(tokens),
        key=lambda item: (
            0 if re.search(r"[A-Z0-9]", item) else 1,
            -len(item),
            item.lower(),
        ),
    )
    return ranked


def _merge_context_terms(
    *, existing: list[str], new_terms: list[str], limit: int
) -> list[str]:
    merged = list(existing)
    for term in new_terms:
        merged = [item for item in merged if item != term]
        merged.append(term)
    return merged[-limit:]


def _trim_context_terms_to_budget(
    *,
    terms: list[str],
    max_chars: int,
    history: list[str],
) -> list[str]:
    if max_chars <= 0:
        return terms

    history_text = "\n".join(history)
    budget = max(160, max_chars // 2) - len(history_text)
    if budget <= 0:
        return []

    kept: list[str] = []
    used = 0
    for term in reversed(terms):
        add = len(term) + (2 if kept else 0)
        if used + add > budget:
            continue
        kept.append(term)
        used += add
    kept.reverse()
    return kept


REPEAT_COLLAPSE_RE = re.compile(r"(.{2,24}?)\1{2,}")
REPEAT_DETECT_RE = re.compile(r"(.{2,24}?)\1{4,}")
PHRASE_TOKEN_RE = re.compile(r"[^。！？!?]+[。！？!?]?")
FILLER_REPEAT_RE = re.compile(
    r"(えーと|えっと|えー|あのー|あの|そのー|その)(?:[\s、,。]*\1)+"
)
MULTISPACE_NUMBER_RE = re.compile(r"(?<=\d)\s+(?=\d)")
JP_CHAR_CLASS = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
JP_SPACE_BEFORE_RE = re.compile(rf"(?<=[{JP_CHAR_CLASS}])\s+(?=[{JP_CHAR_CLASS}])")
JP_PUNCT_SPACE_RE = re.compile(r"\s+([、。，．・：；！？）］】」』])|([（［【「『])\s+")
OVERLAP_COMPARE_DROP_RE = re.compile(
    r"[\s、。，．・：；！？!?,.:;()\[\]{}<>\"'「」『』]+"
)
MIN_OVERLAP_MATCH_CHARS = 12
MAX_OVERLAP_MATCH_CHARS = 96
MIN_OVERLAP_MATCH_RATIO = 0.5
FUZZY_OVERLAP_MIN_RATIO = 0.82
LEADING_CONNECTOR_MARKERS = ("次に", "また", "なお", "では", "そして")
BROKEN_BOUNDARY_RE = re.compile(r"(.)\1{2,}")


def _sanitize_transcript_text(text: str, *, language: str | None = None) -> str:
    value = " ".join((text or "").split()).strip()
    if not value:
        return ""

    value = _normalize_transcript_spacing(value, language=language)

    # 連続反復を縮約し、意味の薄い暴走出力を抑える。
    for _ in range(3):
        collapsed = REPEAT_COLLAPSE_RE.sub(lambda m: m.group(1), value)
        if collapsed == value:
            break
        value = collapsed

    value = _collapse_long_repeated_char_loops(value)
    value = _collapse_repeated_phrase_loops(value)

    if _is_repetition_noise(value):
        return ""
    return value


def _light_proofread(text: str, *, language: str | None = None) -> str:
    value = (text or "").strip()
    if not value:
        return ""

    value = FILLER_REPEAT_RE.sub(lambda m: m.group(1), value)
    value = value.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    value = value.replace('"', "”").replace("'", "’")
    value = MULTISPACE_NUMBER_RE.sub("", value)
    return _sanitize_transcript_text(value, language=language)


def _should_drop_boundary_fragment(
    current: str,
    previous: str,
    *,
    source_mode: str | None = None,
    suspicious: bool = False,
) -> bool:
    clean = (current or "").strip()
    normalized = _normalize_compare_text(clean)
    if not clean or not normalized:
        return False

    if "\ufffd" in clean and len(normalized) <= 32:
        return True

    overlap_prefix = _has_previous_suffix_overlap(current, previous, min_chars=5)
    repeated_tail = bool(BROKEN_BOUNDARY_RE.search(normalized))
    is_display = (source_mode or "").strip().lower() == "display"

    if suspicious and len(normalized) <= 28 and overlap_prefix:
        return True
    if is_display and len(normalized) <= 24 and overlap_prefix and repeated_tail:
        return True
    return False


def _has_previous_suffix_overlap(
    current: str, previous: str, *, min_chars: int = 5
) -> bool:
    a = _normalize_compare_text(current)
    b = _normalize_compare_text(previous)
    if not a or not b:
        return False

    max_overlap = min(len(a), len(b), 16)
    if max_overlap < min_chars:
        return False

    for overlap_len in range(max_overlap, min_chars - 1, -1):
        if b.endswith(a[:overlap_len]):
            return True
    return False


def _accumulate_asr_usage(session: LiveSession, result: ASRChunkResult) -> None:
    usage = result.usage_details or {}
    session.asr_input_tokens += max(0, int(usage.get("input", 0) or 0))
    session.asr_output_tokens += max(0, int(usage.get("output", 0) or 0))
    session.asr_total_tokens += max(0, int(usage.get("total", 0) or 0))
    session.asr_estimated_tokens += max(0, int(result.estimated_tokens or 0))


def _should_retry_weird_transcription(
    text: str,
    previous_text: str,
    *,
    source_mode: str | None = None,
    suspicious: bool = False,
) -> bool:
    clean = (text or "").strip()
    normalized = _normalize_compare_text(clean)
    if not clean or not normalized:
        return False
    if _should_drop_boundary_fragment(
        clean, previous_text, source_mode=source_mode, suspicious=suspicious
    ):
        return True
    if suspicious and len(normalized) <= 40:
        return True
    if "\ufffd" in clean:
        return True
    return False


async def _retry_weird_transcription_if_needed(
    *,
    session: LiveSession,
    prepared,
    trace_context: dict[str, str] | None,
    audio_bytes: bytes,
    previous_text: str,
    result: ASRChunkResult,
) -> ASRChunkResult:
    if not bool(getattr(settings, "asr_rescue_retry_enabled", True)):
        return result
    if not _should_retry_weird_transcription(
        result.text,
        previous_text,
        source_mode=session.audio_source,
        suspicious=bool(result.suspicious),
    ):
        return result

    retry_temperature = float(
        getattr(settings, "asr_rescue_retry_temperature", 0.25) or 0.25
    )
    retry_prompt = _build_rescue_prompt(session)
    logger.info(
        "Retrying weird transcription: session=%s chars=%d suspicious=%s source=%s",
        session.session_id,
        len((result.text or "").strip()),
        bool(result.suspicious),
        session.audio_source,
    )
    rescue_result = await blocking_work_pool.run(
        "asr",
        session.transcriber.transcribe_chunk,
        audio_bytes,
        mime_type=prepared.mime_type,
        language=session.language,
        prompt=retry_prompt,
        temperature=retry_temperature,
        trace_context=trace_context,
    )
    _accumulate_asr_usage(session, rescue_result)
    if _prefer_rescue_transcription_result(
        original=result,
        retry=rescue_result,
        previous_text=previous_text,
        source_mode=session.audio_source,
    ):
        logger.info(
            "Using rescued transcription: session=%s chars=%d",
            session.session_id,
            len((rescue_result.text or "").strip()),
        )
        return rescue_result
    return result


def _build_rescue_prompt(session: LiveSession) -> str | None:
    parts: list[str] = []
    shared_vocabulary = str(getattr(session, "shared_vocabulary", "") or "").strip()
    base_prompt = str(getattr(session, "base_prompt", "") or "").strip()
    if shared_vocabulary:
        parts.append(shared_vocabulary)
    if base_prompt:
        parts.append(base_prompt)
    parts.append(
        "断片的な音声でも無理に補完せず、聞こえた範囲だけをそのまま文字起こししてください。"
    )
    prompt = "\n".join(part for part in parts if part).strip()
    return prompt or None


def _prefer_rescue_transcription_result(
    *,
    original: ASRChunkResult,
    retry: ASRChunkResult,
    previous_text: str,
    source_mode: str | None = None,
) -> bool:
    original_score = _transcription_weirdness_score(
        original.text,
        previous_text,
        source_mode=source_mode,
        suspicious=bool(original.suspicious),
    )
    retry_score = _transcription_weirdness_score(
        retry.text,
        previous_text,
        source_mode=source_mode,
        suspicious=bool(retry.suspicious),
    )
    if retry_score != original_score:
        return retry_score < original_score
    return len(_normalize_compare_text(retry.text)) > len(
        _normalize_compare_text(original.text)
    )


def _transcription_weirdness_score(
    text: str,
    previous_text: str,
    *,
    source_mode: str | None = None,
    suspicious: bool = False,
) -> int:
    clean = (text or "").strip()
    normalized = _normalize_compare_text(clean)
    score = 0
    if not clean:
        return 100
    if suspicious:
        score += 4
    if "\ufffd" in clean:
        score += 8
    if _should_drop_boundary_fragment(
        clean, previous_text, source_mode=source_mode, suspicious=suspicious
    ):
        score += 10
    if _is_repetition_noise(clean):
        score += 6
    if len(normalized) <= 12:
        score += 2
    return score


def _trim_overlap_prefix(current: str, previous: str) -> str:
    current = (current or "").strip()
    previous = (previous or "").strip()
    if not current or not previous:
        return current

    previous_normalized, _ = _normalize_overlap_compare_text(previous)
    current_normalized, current_index_map = _normalize_overlap_compare_text(current)
    if not previous_normalized or not current_normalized:
        return current

    max_overlap = min(
        len(previous_normalized), len(current_normalized), MAX_OVERLAP_MATCH_CHARS
    )
    min_required_overlap = max(
        MIN_OVERLAP_MATCH_CHARS,
        int(
            min(len(previous_normalized), len(current_normalized))
            * MIN_OVERLAP_MATCH_RATIO
        ),
    )
    if max_overlap < min_required_overlap:
        return current

    best_overlap = 0
    for overlap_len in range(max_overlap, min_required_overlap - 1, -1):
        if previous_normalized[-overlap_len:] == current_normalized[:overlap_len]:
            best_overlap = overlap_len
            break

    if best_overlap <= 0:
        best_overlap = _find_fuzzy_overlap(previous_normalized, current_normalized)
        if best_overlap <= 0:
            return current

    cut_index = current_index_map[best_overlap - 1]
    trimmed = current[cut_index:].lstrip()
    trimmed = trimmed.lstrip("、。，．・：；！？!?,.:;）］】」』")
    prefix = current[:cut_index].rstrip()
    for marker in LEADING_CONNECTOR_MARKERS:
        if prefix.endswith(marker) and not trimmed.startswith(marker):
            trimmed = marker + trimmed
            break
    return trimmed or current


def _coerce_monotonic_bounds(
    *, ts_start: int, ts_end: int, previous_end_ms: int
) -> tuple[int, int]:
    start = max(0, int(ts_start))
    end = max(start, int(ts_end))
    previous_end = max(0, int(previous_end_ms))
    if start < previous_end:
        start = previous_end
    if end < start:
        end = start
    return start, end


def _normalize_transcript_spacing(text: str, *, language: str | None) -> str:
    lowered = (language or "").strip().lower()
    if lowered and not lowered.startswith("ja"):
        return text

    value = JP_SPACE_BEFORE_RE.sub("", text)

    def _punct_repl(match: re.Match[str]) -> str:
        if match.group(1):
            return match.group(1)
        return match.group(2)

    value = JP_PUNCT_SPACE_RE.sub(_punct_repl, value)
    return value


def _normalize_overlap_compare_text(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    index_map: list[int] = []
    for index, char in enumerate((text or "").strip()):
        if OVERLAP_COMPARE_DROP_RE.fullmatch(char):
            continue
        normalized_chars.append(char)
        index_map.append(index + 1)
    return "".join(normalized_chars), index_map


def _normalize_compare_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


def _find_fuzzy_overlap(previous_normalized: str, current_normalized: str) -> int:
    max_overlap = min(
        len(previous_normalized), len(current_normalized), MAX_OVERLAP_MATCH_CHARS
    )
    min_required_overlap = max(
        MIN_OVERLAP_MATCH_CHARS,
        int(
            min(len(previous_normalized), len(current_normalized))
            * MIN_OVERLAP_MATCH_RATIO
        ),
    )
    if max_overlap < min_required_overlap:
        return 0

    for overlap_len in range(max_overlap, min_required_overlap - 1, -1):
        left = previous_normalized[-overlap_len:]
        right = current_normalized[:overlap_len]
        if (
            difflib.SequenceMatcher(None, left, right).ratio()
            >= FUZZY_OVERLAP_MIN_RATIO
        ):
            return overlap_len
    return 0


def _is_repetition_noise(text: str) -> bool:
    normalized = _normalize_compare_text(text)
    if len(normalized) < 32:
        return False

    matched = REPEAT_DETECT_RE.search(normalized)
    if not matched:
        return False

    run_len = len(matched.group(0))
    # 1箇所の反復だけで大半を占める場合はノイズ扱い。
    return run_len >= max(36, int(len(normalized) * 0.45))


def _collapse_repeated_phrase_loops(text: str) -> str:
    tokens = [
        token.strip() for token in PHRASE_TOKEN_RE.findall(text or "") if token.strip()
    ]
    if len(tokens) < 4:
        return text

    out: list[str] = []
    i = 0
    while i < len(tokens):
        collapsed = False
        max_unit = min(3, (len(tokens) - i) // 2)
        for unit_size in range(max_unit, 0, -1):
            unit = tokens[i : i + unit_size]
            if len(unit) < unit_size:
                continue

            repeats = 1
            cursor = i + unit_size
            while (
                cursor + unit_size <= len(tokens)
                and tokens[cursor : cursor + unit_size] == unit
            ):
                repeats += 1
                cursor += unit_size

            if repeats >= 3:
                out.extend(unit[:unit_size])
                i = cursor
                collapsed = True
                break

        if not collapsed:
            out.append(tokens[i])
            i += 1

    collapsed_text = " ".join(out).strip()
    return collapsed_text or text


def _collapse_long_repeated_char_loops(text: str) -> str:
    value = (text or "").strip()
    if len(value) < 48:
        return value

    out: list[str] = []
    i = 0
    text_len = len(value)
    while i < text_len:
        collapsed = False
        max_unit = min(64, (text_len - i) // 3)
        for unit_size in range(max_unit, 8, -1):
            unit = value[i : i + unit_size]
            if len(unit) < unit_size or unit.strip() != unit:
                continue

            repeats = 1
            cursor = i + unit_size
            while (
                cursor + unit_size <= text_len
                and value[cursor : cursor + unit_size] == unit
            ):
                repeats += 1
                cursor += unit_size

            if repeats >= 3:
                out.append(unit)
                i = cursor
                collapsed = True
                break

        if not collapsed:
            out.append(value[i])
            i += 1

    collapsed_text = "".join(out).strip()
    return collapsed_text or value


def _is_near_duplicate(
    current: str,
    previous: str,
    *,
    current_start_ms: int | None = None,
    previous_end_ms: int | None = None,
) -> bool:
    a = _normalize_compare_text(current)
    b = _normalize_compare_text(previous)
    if not a or not b:
        return False

    shorter = min(len(a), len(b))
    longer = max(len(a), len(b))
    gap_ms: int | None = None
    if current_start_ms is not None and previous_end_ms is not None:
        gap_ms = max(0, int(current_start_ms) - int(previous_end_ms))
    if a == b:
        return gap_ms is None or gap_ms <= 2_000
    if shorter >= 24 and shorter / longer >= 0.92 and (a in b or b in a):
        return gap_ms is None or gap_ms <= 2_000

    ratio_threshold = 0.9 if gap_ms is not None and gap_ms <= 2_000 else 0.97
    return (
        shorter >= 24 and difflib.SequenceMatcher(None, a, b).ratio() >= ratio_threshold
    )
