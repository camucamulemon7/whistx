# speech.cpp / Qwen3-ASR routes

The HTTP request profile is checked against [nyosegawa/speech.cpp v0.8.3](https://github.com/nyosegawa/speech.cpp/tree/7ac99697cd680c78beef482955279ef15d4421c7), C API 3.1. The installed Mac release reports 0.8.3. Qwen3-ASR-1.7B Q8_0 runs on Metal MTL0. Whistx loads no ASR weights. [The versioned server API](https://github.com/nyosegawa/speech.cpp/blob/7ac99697cd680c78beef482955279ef15d4421c7/docs/server.md) implements a subset of OpenAI's audio API.

## One HTTP recognition route

| Entry point | Capture / conversion | Recognition |
|---|---|---|
| Live microphone/shared audio/both | AudioWorklet PCM16 mono 16 kHz → Whistx `/ws/transcribe/live` | Rolling HTTP/LocalAgreement: new samples every 1.25 seconds for one track, 2.5 seconds per track for mic+display; bounded final windows default to five seconds |
| Automatic high accuracy | The same retained PCM journal; complete 30-second intervals and stop tail | A separate HTTP client, using the same ASR URL, model and key |
| Video/audio import | FFmpeg extraction to mono PCM16 WAV; file-only pause strategy, max 90 s for speech.cpp, source-anchored temporary preview | Same gateway batch transcriber; one file request globally, independent timeout and media worker lane |
| Manual audio re-recognition | Retained source WAV; old compressed clips decoded only when needed | The same configured batch transcriber over HTTP |
| Legacy capture fallback | MediaRecorder → Whistx `/ws/transcribe` → existing preprocessing | The same configured batch transcriber; AudioWorklet is preferred but not required for speech.cpp |

The browser-to-Whistx WebSocket carries capture and UI events. It does not require an upstream OpenAI Realtime connection. Whisper keeps its existing HTTP lane and confidence-aware adapter. vLLM keeps its separate Realtime protocol. Future OpenAI Realtime support can be added as a transport capability at the gateway; it is not required for current recording, import or refinement. The unused direct-native WebSocket implementation, Host workaround and 16→24 kHz conversion have been removed from Whistx.

## Settings and the current Mac gateway

Select `ASR_BACKEND=speech_cpp` for speech.cpp's HTTP parameter profile. All recognition uses `ASR_BASE_URL`, `ASR_MODEL`, and `ASR_API_KEY`, including recording. `ASR_API_TIMEOUT_SECONDS` bounds rolling requests; HQ/manual recognition uses `ASR_HIGH_ACCURACY_TIMEOUT_SECONDS`; files use `MEDIA_ASR_TIMEOUT_SECONDS`. [File recognition](media-import.md) has its own segmentation and temporary preview, without RT/HQ passes or LocalAgreement. The administrator screen saves shared connection settings in the mounted data directory. Blank key fields preserve the existing credential. Saving requires an application restart.

Current intended registration: `ASR_BASE_URL=http://host.docker.internal:4000/v1`, `ASR_MODEL=Qwen3-ASR-1.7B-native`, using the existing LiteLLM key. LiteLLM maps this alias to `openai/Qwen3-ASR-1.7B` at `http://host.docker.internal:8080/v1`, mode `audio_transcription`. Existing `Qwen3-ASR-1.7B` and Irodori TTS registrations keep their adapter endpoint for shared OpenWebUI compatibility. No existing model registration or authentication policy is changed.

The adapter is a separately added local FastAPI service, not part of upstream speech.cpp. It converts compressed uploads for clients such as OpenWebUI. It is unnecessary for Whistx's normal WAV requests. Native registration must not copy the adapter's `additional_drop_params` option: installed LiteLLM 1.104.0 forwards it as `additional_drop_params[]` on the audio endpoint, and native rejects it.

Retired `ASR_REALTIME_BASE_URL`, `ASR_REALTIME_MODEL`, and `ASR_REALTIME_API_KEY` values are ignored. Existing saved values remain on disk for rollback, but are neither loaded nor exposed in the administrator API. They cannot be newly submitted through that API. Saved meeting model identities remain unchanged; resuming a recording after selecting a different model alias is rejected, while its saved audio and transcript remain available for manual re-recognition.

`OPENWEBUI_BASE_URL` configures LLM summary, QA and translation; it does not silently route ASR through OpenWebUI. Translation remains a separate LLM operation, preserving the recognized source. Automatic language recognition omits the language field; selected languages are passed to the ASR endpoint.

## Audio, errors and truncation

Requests use multipart WAV, model, response_format=json, optional prompt and selected language. They omit temperature, vLLM priority, streaming and Whisper-only metrics. Native accepts WAV PCM/float at arbitrary rates and averages/resamples internally. Compressed old clips use a pure FFmpeg WAV conversion without highpass, lowpass, volume, noise reduction or overlap. Normal WAV input is not decoded again. Existing silence gates and sample-anchored source retention remain.

HTTP non-success and timeouts fail recognition instead of committing a successful result. Failed rolling recognition keeps source audio and reports a buffered error. HQ retains the earlier transcript after failed attempts. For a disconnected live client, the server keeps the worker and source lock until in-flight inference returns, preserving acknowledged audio and durable results. Cancelling an awaiting task prevents it from committing a late result, but an already dispatched synchronous HTTP request may continue until its timeout. Existing media cancellation kills decoders and retains the worker until inference returns.

Native `X-Speech-Stop=model_limit` means HTTP 200 contains incomplete text. LiteLLM 1.104.0 forwards it as `llm_provider-x-speech-stop`. Whistx checks both names before returning an ASR result, including recording, HQ, import and manual refinement. A missing header remains compatible with existing gateways; therefore a gateway that removes both forms cannot prove completion. The actual gateway route and stop header must be verified after deployment.

## Verification

Run `scripts/verify_speech_cpp.py --help` from the repository root. It creates an isolated SQLite DB and synthetic user/session, exercises authenticated capture, automatic HQ, real FFmpeg MP4 import and manual refinement, and records the HTTP target, gateway provider base and stop header. Use synthetic PCM16/16 kHz mono input. It never uses the running application's DB or creates production accounts.

Mock tests cover HTTP status/timeout failures, native and LiteLLM stop headers, pure format conversion, common URL/model/key selection, cancellation with retained source, retired-secret masking and existing Whisper behavior. Real-model tests and simulated model-limit tests are reported separately. Physical microphone capture, production PostgreSQL and future OpenAI Realtime relay are not covered by this synthetic workflow.

After the resource-lifecycle and worker-cancellation fixes, the same isolated workflow passed on 2026-10-16 through the running LiteLLM at `http://127.0.0.1:4000/v1` with `Qwen3-ASR-1.7B-native`. Recording emitted partial/final results and an automatic HQ revision; MP4 import and manual re-recognition completed. All 14 transcription requests returned HTTP 200 and `complete`, with the gateway reporting the native provider at `http://host.docker.internal:8080/v1`. The machine-readable local report is `artifacts/architecture-fixes/speech-cpp-integration.json`. The running application and its settings/data were not changed.

## Recording display and cadence

Recording uses the main transcript below the status controls. The separate green interim card, its HTML container, background/padding CSS and old renderer remain removed. The HTTP recognizer's LocalAgreement-2 `stableText` now extends a lightweight text node within the lower paragraph immediately. Volatile hypotheses are omitted; repeated updates and regressing agreed prefixes do not erase/retype visible progress. No typewriter queue or text timer is added. Final records replace the provisional span with the complete authoritative result. Earlier HQ replacement preserves current progress; ready/reconnect and successful finalization clear temporary spans.

Agreed text is still provisional: it uses muted text and an accessible label/tooltip, separate from `.log-row` and `state.segments`, and is excluded from copy, exports and history. There is no saved draft ID or audio action. Final characters render synchronously, including the stop tail. Unchanged committed nodes are neither rewritten nor reparented on paragraph updates; HQ moves/replaces only necessary rows. Following readers scroll after text changes, while readers inspecting earlier text keep their position. Connection/errors, HQ status, stop retry, pending-audio download and the separate file-import preview remain.

`共有音声` is a capture-source identifier. Main paragraph headings omit default source names for a single input and retain source distinction for mic/display dual capture or history. Actual diarized speaker names and timestamps remain visible; stored `track`, `speaker`, text, copies and exports are unchanged. There is no separate recording-preview card in either input mode.

Speech.cpp now requests updates after 20,000 new samples (1.25 s) for one track, or 40,000 samples (2.5 s) for two tracks. Whisper retains its 2 s cadence. Capture packets remain 250 ms, final RT windows remain 5 s, LocalAgreement-2 and silence gates are unchanged, and HQ still uses the retained 30 s journal and stop tail. The silence endpoint threshold remains 700 ms by default. Existing polling coalesces incoming audio while a request is running: at most one RT request per meeting and one independent HQ request, without a request queue for every capture packet.

The live backend gate uses 20 ms frames, DC-offset removal and RMS floor 0.003, with the existing quiet stationary broadband-noise rejection and 300/100 ms onset/tail padding. This is a conservative signal heuristic, not neural VAD. Capture still sends silence and retains every source sample; only decoding is skipped. The browser's legacy chunk-length presets affect MediaRecorder fallback, not this live decoder cadence.

With continuous speech and aligned default 5 s windows this targets 48 RT calls/minute in either capture mode, plus ordinary HQ calls (2 per track/minute), leaving room under the default 60/minute ASR budget. A 1 s interval would consume the entire budget on RT alone. Shorter configured windows, extra language-recovery HQ calls, simultaneous meetings and a lower custom quota can still hit the existing limiter; no quota or security setting is relaxed. Finer updates increase total ASR work even though concurrency remains bounded.

`python -m scripts.compare_speech_cpp_cadence --audio <synthetic.wav> --report <report.json>` compares the previous 2 s cadence with the current one. Supply existing gateway URL/model/key through the ASR environment. The script creates isolated journals, paces PCM in real time, exercises actual HTTP RT/HQ, records timing and provider headers, and verifies retained audio and finalization. Its fixture bypasses only the isolated app's database-backed quota counter; the real gateway retains its own limits. This tests backend synthetic PCM ingestion, not browser screen/audio capture.

Mac Qwen3-ASR-1.7B-native comparison on 2026-10-11 through the running LiteLLM (same model, 250 ms packets and 5 s windows):

| Synthetic clip | First partial, before → after | Median partial gap | Nonempty partials | RT / HQ requests |
|---|---|---|---|---|
| Japanese, 10.02 s including silence tail | 2.398 → 1.417 s | 2.073 → 1.309 s | 4 → 6 | 6/1 → 8/1 |
| Japanese/English/Japanese, 33.28 s including silence tail | 2.190 → 1.445 s | 2.090 → 1.306 s | 12 → 17 | 18/3 → 23/3 |

Final HQ text was identical in both comparisons, including English and sentence endings. All 63 HTTP requests used the configured LiteLLM transcription endpoint and reported native provider/complete stop headers. RT/HQ concurrency stayed 1 each, original PCM was retained, and observed partial backlog stayed within one 250 ms packet. These short serial comparisons do not establish multi-user load capacity. No running service was restarted or configured during the comparison itself.

The first final RT record still arrived at 5.25–5.34 s, and median final-record intervals stayed 4.96–5.00 s. The lower UI now uses agreed prefixes before those final records, with the same HTTP cadence, final context and sample boundaries. Dual input retains its 2.5 s budget: with the default 5 s window there may be no nonempty agreed prefix before a final result, so the earlier display benefit is limited in that mode. No ASR request frequency, quota, service setting or upstream transport is changed.

A fresh 10.02 s synthetic Japanese run on 2026-10-11 through the existing LiteLLM/native model measured median RT inference 178 ms, first partial 1.496 s, first agreed text 2.745 s and first final 5.351 s. Replaying exactly those real ASR events through the deployed `9bbcad1` and updated frontend controllers changed first lower-text visibility from 5.351 to 2.745 s and visible updates from 2 to 6. Prefix lengths were 6 characters at 2.745 s and 13 at 4.042 s. The same trace used 8 RT requests and 1 HQ request; final records and copied text matched exactly. The updated controller's maximum event-handler work was 0.364 ms in a Node DOM fixture. That measures JavaScript execution, not real-browser layout/paint or WebSocket transit. Physical capture, natural long speech and dual-input latency improvement remain unverified. DOM tests cover dual-source separation, repeat/correction handling, earlier HQ with current progress, reconnect, stopped-tail rendering and mocked history saving.

The Mac application was updated through `6631c1b` on 2026-10-11 after an authorized idle restart, including agreed-prefix display and the [CSS interaction fixes](css-interactions.md). Existing configuration, credentials, database content/schema and artifacts were preserved. Already-open pages require a refresh while recording is stopped. Repository merges do not automatically redeploy or restart the Mac application.

Shared-audio recording was confirmed by the user on 2026-10-11. The later automated Chrome suite uses synthetic capture/API fixtures and checks recording lifecycle, HQ/reconnect updates, media import, source labels, main-transcript structure and CSS interaction. It also passes against UI files copied from the running container. These checks do not establish real microphone/screen-sharing behavior, physical touch operation, or Safari/Firefox behavior. Temporary browser test profiles do not modify the user's normal browser preferences.
