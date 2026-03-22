# ASR Evaluation Fixtures

This directory contains small evaluation fixtures for `scripts/eval_asr.py`.

## Layout

Each sample lives in its own directory and should contain:

- `reference.txt`
- `hypothesis.txt`
- `metadata.json` when metadata is available

The evaluator discovers sample directories recursively under this root.

## Categories

- `one_on_one`
- `multi_speaker`
- `screen_share`
- `mic_only`
- `domain_terms`
- `noisy`

## Dummy sample

`one_on_one/sample_001` is a minimal example that can be used to verify the evaluator locally.
