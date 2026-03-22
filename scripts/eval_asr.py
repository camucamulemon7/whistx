#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REFERENCE_FILENAME = "reference.txt"
HYPOTHESIS_FILENAME = "hypothesis.txt"
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class SampleCase:
    sample_id: str
    sample_dir: Path
    reference_text: str
    hypothesis_text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SampleResult:
    sample_id: str
    sample_dir: str
    category: str
    language: str
    ref_words: int
    hyp_words: int
    word_edits: int
    wer: float
    ref_chars: int
    hyp_chars: int
    char_edits: int
    cer: float
    exact_match: bool
    metadata: dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ASR output against reference transcripts.")
    parser.add_argument("--dataset", type=Path, help="Dataset root containing sample directories.")
    parser.add_argument("--reference", type=Path, help="Reference transcript file for a single pair run.")
    parser.add_argument("--hypothesis", type=Path, help="Hypothesis transcript file for a single pair run.")
    parser.add_argument("--json-out", type=Path, help="Write JSON report to this path.")
    parser.add_argument("--csv-out", type=Path, help="Write per-sample CSV rows to this path.")
    parser.add_argument("--strict", action="store_true", help="Fail when no sample cases are found.")
    args = parser.parse_args(argv)

    cases = load_cases(args)
    if not cases:
        message = "no evaluation cases found"
        if args.strict:
            raise SystemExit(message)
        print(message, file=sys.stderr)
        return 1

    results = [evaluate_case(case) for case in cases]
    summary = build_summary(results)

    report = {
        "summary": summary,
        "cases": [result_to_dict(result) for result in results],
    }

    print_summary(summary)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv_out, results)

    return 0


def load_cases(args: argparse.Namespace) -> list[SampleCase]:
    if args.reference or args.hypothesis:
        if not args.reference or not args.hypothesis:
            raise SystemExit("--reference and --hypothesis must be provided together")
        return [load_pair_case(args.reference, args.hypothesis)]

    dataset_root = args.dataset or Path("tests/fixtures/asr_eval")
    if not dataset_root.exists():
        raise SystemExit(f"dataset root not found: {dataset_root}")

    return discover_cases(dataset_root)


def discover_cases(dataset_root: Path) -> list[SampleCase]:
    cases: list[SampleCase] = []
    for reference_path in sorted(dataset_root.rglob(REFERENCE_FILENAME)):
        sample_dir = reference_path.parent
        hypothesis_path = sample_dir / HYPOTHESIS_FILENAME
        if not hypothesis_path.exists():
            continue
        cases.append(load_case(sample_dir, reference_path, hypothesis_path))
    return cases


def load_pair_case(reference_path: Path, hypothesis_path: Path) -> SampleCase:
    sample_dir = reference_path.parent
    return load_case(sample_dir, reference_path, hypothesis_path, sample_id_override=sample_dir.name or "pair")


def load_case(
    sample_dir: Path,
    reference_path: Path,
    hypothesis_path: Path,
    *,
    sample_id_override: str | None = None,
) -> SampleCase:
    reference_text = read_text(reference_path)
    hypothesis_text = read_text(hypothesis_path)
    metadata = read_metadata(sample_dir / METADATA_FILENAME)

    sample_id = sample_id_override or metadata.get("id") or sample_dir.name
    return SampleCase(
        sample_id=str(sample_id),
        sample_dir=sample_dir,
        reference_text=reference_text,
        hypothesis_text=hypothesis_text,
        metadata=metadata,
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid metadata JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"metadata must be an object: {path}")
    return payload


def evaluate_case(case: SampleCase) -> SampleResult:
    ref_words = tokenize_words(case.reference_text)
    hyp_words = tokenize_words(case.hypothesis_text)
    ref_chars = tokenize_chars(case.reference_text)
    hyp_chars = tokenize_chars(case.hypothesis_text)

    word_edits = levenshtein_distance(ref_words, hyp_words)
    char_edits = levenshtein_distance(ref_chars, hyp_chars)
    ref_word_count = len(ref_words)
    ref_char_count = len(ref_chars)

    return SampleResult(
        sample_id=case.sample_id,
        sample_dir=str(case.sample_dir),
        category=str(case.metadata.get("category", "")),
        language=str(case.metadata.get("language", "")),
        ref_words=ref_word_count,
        hyp_words=len(hyp_words),
        word_edits=word_edits,
        wer=rate(word_edits, ref_word_count),
        ref_chars=ref_char_count,
        hyp_chars=len(hyp_chars),
        char_edits=char_edits,
        cer=rate(char_edits, ref_char_count),
        exact_match=normalize_text(case.reference_text) == normalize_text(case.hypothesis_text),
        metadata=case.metadata,
    )


def tokenize_words(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    if re.search(r"\s", normalized):
        return [token for token in normalized.split(" ") if token]
    return tokenize_chars(normalized)


def tokenize_chars(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [char for char in normalized if not char.isspace()]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def levenshtein_distance(left: list[str], right: list[str]) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    current = [0] * (len(right) + 1)

    for i, left_item in enumerate(left, start=1):
        current[0] = i
        for j, right_item in enumerate(right, start=1):
            substitution_cost = 0 if left_item == right_item else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + substitution_cost,
            )
        previous, current = current, previous

    return previous[-1]


def rate(errors: int, ref_count: int) -> float:
    if ref_count <= 0:
        return 0.0 if errors <= 0 else 1.0
    return errors / ref_count


def build_summary(results: list[SampleResult]) -> dict[str, Any]:
    sample_count = len(results)
    word_ref_total = sum(result.ref_words for result in results)
    char_ref_total = sum(result.ref_chars for result in results)
    word_edits_total = sum(result.word_edits for result in results)
    char_edits_total = sum(result.char_edits for result in results)

    macro_wer = sum(result.wer for result in results) / sample_count if sample_count else 0.0
    macro_cer = sum(result.cer for result in results) / sample_count if sample_count else 0.0

    return {
        "sampleCount": sample_count,
        "wordRefTotal": word_ref_total,
        "charRefTotal": char_ref_total,
        "wordEditsTotal": word_edits_total,
        "charEditsTotal": char_edits_total,
        "wer": rate(word_edits_total, word_ref_total),
        "cer": rate(char_edits_total, char_ref_total),
        "macroWer": macro_wer,
        "macroCer": macro_cer,
    }


def result_to_dict(result: SampleResult) -> dict[str, Any]:
    return {
        "sampleId": result.sample_id,
        "sampleDir": result.sample_dir,
        "category": result.category,
        "language": result.language,
        "refWords": result.ref_words,
        "hypWords": result.hyp_words,
        "wordEdits": result.word_edits,
        "wer": result.wer,
        "refChars": result.ref_chars,
        "hypChars": result.hyp_chars,
        "charEdits": result.char_edits,
        "cer": result.cer,
        "exactMatch": result.exact_match,
        "metadata": result.metadata,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(
        "samples={sampleCount} wer={wer:.4f} cer={cer:.4f} macro_wer={macroWer:.4f} macro_cer={macroCer:.4f}".format(
            **summary
        )
    )


def write_csv(path: Path, results: list[SampleResult]) -> None:
    fieldnames = [
        "sampleId",
        "sampleDir",
        "category",
        "language",
        "refWords",
        "hypWords",
        "wordEdits",
        "wer",
        "refChars",
        "hypChars",
        "charEdits",
        "cer",
        "exactMatch",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = result_to_dict(result)
            writer.writerow({key: row.get(key) for key in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
