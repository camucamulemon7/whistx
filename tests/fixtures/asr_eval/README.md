# ASR Evaluation Fixtures

This directory contains small evaluation fixtures for `scripts/eval_asr.py`.

## Layout

Each sample lives in its own directory and should contain:

- `reference.txt`
- `hypothesis.txt`
- `metadata.json` when metadata is available

The evaluator discovers sample directories recursively under this root.

Metadata can record `namedEntities`, speaker label sequences, boundary counts, latency, API requests, estimated cost, timestamp validity, model/profile, audio source, and experiment toggles (VAD, chunk length, overlap, prompt, retry, and silence drop). Raw audio is intentionally excluded from this repository.

## Categories

- `one_on_one`
- `multi_speaker`
- `screen_share`
- `mic_only`
- `domain_terms`
- `noisy`

## Dummy sample

`one_on_one/sample_001` is a minimal example that can be used to verify the evaluator locally.

## Data handling

- Use only consented recordings and remove names, customer data, credentials, and screen contents before evaluation.
- Keep sensitive audio in an access-controlled external dataset; commit only anonymized transcripts and aggregate metadata.
- Use opaque sample IDs and document retention/deletion ownership outside the repository.
- Treat reference transcripts as sensitive even when audio is not stored.

## Required coverage

Production baselines should cover Japanese one-to-one and multi-speaker meetings, English, mixed Japanese/English, technical terms, silence/backchannels, microphone and screen-share audio, noisy input, and cross-chunk sentences. The checked-in sample is deliberately synthetic and validates evaluator behavior only.
