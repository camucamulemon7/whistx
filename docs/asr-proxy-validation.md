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

## Native Realtime is a separate integration

The shared adapter above displays progressive transcripts using successive HTTP requests and local agreement. It does not use LiteLLM's native `/realtime` WebSocket.

[LiteLLM's Realtime documentation](https://docs.litellm.ai/docs/realtime) shows a realtime model deployment (`model_info.mode: realtime`) and a model query parameter on the WebSocket URL. This deployment advertised both ASR models as `hosted_vllm` providers in `audio_transcription` mode. The current Qwen-specific client omits the query model; its proxy handshake returned 403. A diagnostic connection with `?model=Qwen3-ASR-1.7B` passed the handshake but closed before a usable session arrived.

Changing mode alone has therefore not been validated as a solution. The Qwen-specific adapter uses vLLM's model/commit/append/final-commit protocol; the proxy's documented Realtime examples target other providers/protocols. Keep the verified HTTP adapter for this proxy. A separate compatible realtime deployment, routing, and protocol check are needed before enabling `ASR_BACKEND=qwen3_vllm` through it. The direct vLLM endpoint at port 8004 passed separate Qwen tests, but that is not evidence of native Realtime support through port 4000.
