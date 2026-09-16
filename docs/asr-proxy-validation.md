# ASR through LiteLLM: verified configuration

Validated on 2026-09-15 against `http://localhost:4000/v1`, using the advertised model names `Qwen3-ASR-1.7B` and `whisper-large-v3-turbo`.

```dotenv
ASR_BASE_URL=http://localhost:4000/v1
ASR_API_KEY=<your LiteLLM key>
ASR_BACKEND=whisper
ASR_MODEL=Qwen3-ASR-1.7B
ASR_DEFAULT_LANGUAGE=auto
```

In this configuration, `whisper` selects the shared OpenAI-compatible `/audio/transcriptions` adapter; it does not restrict the remote model to Whisper. Switch `ASR_MODEL` to `whisper-large-v3-turbo` and restart to use the other model. When running the application in a container, use a host address reachable from that container instead of `localhost`.

Automatic language selection omits the language field upstream. Sending the literal `auto` caused the deployed Whisper provider to reject inference. The adapter first requests `verbose_json` to retain Whisper confidence/timing metadata. If the provider explicitly rejects that format with HTTP 400, it retries once with `json` and remembers that format for that adapter instance. Qwen accepted `json` on this deployment. Unrelated HTTP 400 errors are not converted into format retries. JSON-only results do not supply segment confidence or word timing.

## Real audio results

Fixture: the public 11-second English [JFK sample from OpenAI Whisper's tests](https://github.com/openai/whisper/blob/main/tests/jfk.flac), converted to 16 kHz mono PCM WAV. The real live adapter was fed audio at capture speed through `scripts/bench_streaming_asr.py`; upstream inference was not mocked. Temporary storage and quota isolation avoid touching application history.

| Model | First partial | Stop-to-finalization | Total, including 11s input | Result |
|---|---:|---:|---:|---|
| Qwen3-ASR-1.7B | 2.349s | 0.355s | 11.372s | Correct speech, finalized, no error events |
| whisper-large-v3-turbo | 2.393s | 0.501s | 11.519s | Correct speech, finalized, no error events |

Both produced the expected “ask not what your country can do for you” passage, differing in punctuation. Automatic and explicit English language settings passed. These are single-run functional checks, not capacity or quality benchmarks. They do not establish Japanese accuracy, mixed-language accuracy, long-meeting stability, microphone latency, or end-to-end WebSocket transport performance. Japanese fixture generation was unavailable, so Japanese recognition is not claimed as tested.

The application/browser recording lifecycle test passed separately with its synthetic server fixtures. The final Python suite passed 149 tests with one PostgreSQL-only test skipped (150 total). Ruff also passed.

To repeat with a PCM WAV and configured `.env`:

```bash
ASR_BACKEND=whisper ASR_MODEL=Qwen3-ASR-1.7B \
  .venv/bin/python scripts/bench_streaming_asr.py sample.wav \
  --language auto --realtime --output artifacts/qwen-check.json
ASR_BACKEND=whisper ASR_MODEL=whisper-large-v3-turbo \
  .venv/bin/python scripts/bench_streaming_asr.py sample.wav \
  --language auto --realtime --output artifacts/whisper-check.json
```

FFmpeg must be installed or supplied through `FFMPEG_BIN` and `DIARIZATION_FFMPEG_BIN`. Do not commit `.env`, API keys, or private recording fixtures.

## Qwen native Realtime through LiteLLM (2026-09-16)

Native WebSocket transcription was verified against the actual LiteLLM 1.97.0
proxy on port 4000 after its Qwen provider was changed from `hosted_vllm` to
`openai` and a nonempty upstream `api_key` was configured. The original
`hosted_vllm` route failed with `Unsupported model`; the OpenAI route without an
upstream key failed with `api_key is required for OpenAI realtime calls`.

The working deployment used:

```yaml
model_list:
  - model_name: Qwen3-ASR-1.7B
    litellm_params:
      model: openai/Qwen3-ASR-1.7B
      api_base: http://host.docker.internal:8004/v1
      api_key: unused  # Only when the upstream vLLM does not require authentication.
    model_info:
      mode: audio_transcription
```

The existing `audio_transcription` mode was retained and worked for this
WebSocket route. LiteLLM's documentation shows `mode: realtime` for dedicated
Realtime deployments; changing mode alone does not add provider support.
The upstream key above is separate from whistx's `ASR_API_KEY`, which authenticates
to LiteLLM. Use the real upstream key when vLLM requires authentication.

whistx now connects to `/v1/realtime?model=Qwen3-ASR-1.7B` and sends the same
model in vLLM's `session.update` event. The model name is URL-encoded. Use the
same proxy model name and upstream model name for this integration; separate
routing aliases are not implemented.

```dotenv
ASR_BACKEND=qwen3_vllm
ASR_BASE_URL=http://localhost:4000/v1
ASR_MODEL=Qwen3-ASR-1.7B
ASR_DEFAULT_LANGUAGE=auto
ASR_REALTIME_WINDOW_SECONDS=5
ASR_HIGH_ACCURACY_ENABLED=1
```

Choose **自動検出** in the recording language selector to use native Realtime.
Explicit Japanese/English selections use the HTTP transcription endpoint to
honor the requested language. High-accuracy recognition also uses HTTP.
The browser's existing language selection takes precedence over the environment
default. A containerized whistx needs a host address reachable from its container
instead of `localhost`.

### Application adapter verification

Replayed the same public 11-second English sample at capture speed through
`QwenLiveMeeting`, including its normal 5-second Realtime windows and final
high-accuracy revision:

- First nonempty partial: 5.143 seconds after replay start.
- Finalization after capture: 1.735 seconds; total: 12.755 seconds.
- Three Realtime records were replaced by one high-accuracy record.
- No error events; the final transcript matched the spoken sample without
  repeated sentences or protocol language headers.

A single long diagnostic WebSocket previously returned repeated phrases and
language headers on both direct and proxy connections. This was not reproduced
with the application's normal 5-second windows on this sample. No heuristic
text deduplication was added: it could remove deliberate repetitions in speech.
This is a single-sample adapter check, not a Japanese accuracy, browser microphone,
long-session, or load test.

```bash
ASR_BACKEND=qwen3_vllm ASR_MODEL=Qwen3-ASR-1.7B \
  python scripts/bench_streaming_asr.py path/to/16k-mono.wav \
  --language auto --realtime --output artifacts/qwen-realtime-check.json
```

Whisper's current Xinference upstream on port 9997 returns 404 for
`/v1/realtime`. Keep `ASR_BACKEND=whisper` for `whisper-large-v3-turbo`;
the Qwen native backend must not be selected for that model.
