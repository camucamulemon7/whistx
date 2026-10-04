# whistx architecture

この文書は現在の実装に対する規範的な案内である。移行メモではない。新しい責務や設定を追加するときは、ここに示す一方向の依存とsource of truthを維持する。

## システム境界

```text
Browser
  ├─ HTTP ───────> api/routes ──> services ──> repositories ──> database
  │                                  └────────> artifact storage ──> filesystem
  └─ WebSocket ─> api/ws ──────> runtime ────> transcription/*
                                             ├─> audio_pipeline
                                             ├─> openai_whisper
                                             ├─> transcript_store
                                             └─> diarizer

server.app:app
  └─ core.application.create_app
       ├─ middleware
       ├─ api routers
       ├─ one static mount
       └─ one lifespan ──> runtime startup/shutdown
```

フロントエンドは次の依存方向を取る。

```text
main.js
  └─ src/bootstrap.js + src/app.js (orchestration)
       ├─ api/*, auth/api.js, history/api.js, capabilities/api.js
       ├─ audio/vad.js, transcription/websocket.js
       ├─ auth/session.js, history/state.js, state/storage.js
       └─ ui/format.js, ui/theme.js
```

## Source of truth

| 領域 | 唯一のsource of truth | 補助表現 |
|---|---|---|
| アプリ設定 | `server/core/config/` と `registry.py` | `.env.example`、Compose、READMEは台帳から同期する |
| ASGI app | `server/core/application.create_app()` | `server/app.py` はインスタンス公開だけ |
| HTTP routes | `server/api/routes/` | runtimeにはデコレータを置かない |
| WebSocket入口 | `server/api/ws/transcribe.py` | 認証・guest制限を所有 |
| Live session | `server/transcription/session.py` | runtimeは生成・終了を調停 |
| chunk protocol | `server/transcription/messages.py` | client側は `web/src/transcription/websocket.js` |
| ASR worker | `server/transcription/worker.py` | 依存は `WorkerDependencies` で明示 |
| 要約・校正処理 | `services/summary_service.py` | routeがruntimeのmodel/observerを明示引数で渡す。request schemaは`schemas.py` |
| OIDC HTTPフロー | `services/oidc_flow.py` | cookie/redirect/認証フローはruntimeから独立 |
| 話者ラベル | `services/diarization_service.py` | runtimeがdiarizer/send/configを明示引数で渡す |
| OIDC transport | `server/services/oidc_service.py` | routeはcookieとHTTPフローを調停 |
| 認証済み履歴metadata | Database | repositoryがDB accessを所有 |
| runtime transcript | `TranscriptStore`配下のartifact | 保存時にhistory serviceが検証・コピー |
| 保存済みartifact | history directory | DBには検索・認可・参照用metadataを置く |
| frontend session state | `web/src/app.js` の単一state | 純粋な変換・永続化は小モジュールへ委譲 |
| 依存バージョン | `requirements.lock`, `requirements-dev.lock` | 入力は`requirements*.txt` |

## Appとライフサイクル

`server.app:app` が唯一の公開エントリーポイントである。`create_app()` はmiddleware、router、static mountを一度だけ登録する。startup/shutdownはFastAPI lifespanからのみ実行する。`runtime.py` は外部クライアントと長寿命リソースの構築・破棄を担当するが、FastAPI appやrouteを生成しない。

プロセス内の可変状態は現在、ASR/LLM client、active WebSocket、cleanup task、OIDC discovery cache、rate-limit bucketに限る。LiveSessionの状態は接続ごとのdataclassへ閉じ、ASR workerの依存は明示的な`WorkerDependencies`で渡す。複数worker間で共有すべき制限は将来外部storeへ移す必要がある。

## 履歴とartifactの整合性境界

1. 録音中は`TranscriptStore`がruntime txt/jsonl/audio/screenshotを所有する。
2. stop処理がmetadataをfinalizedにする。
3. history serviceはfinalized、所有token、パスがroot配下であることを検証する。
4. artifactを一時ディレクトリへコピーし、DB metadata/segmentをflushする。
5. 最終ディレクトリへのrename後にcommitする。失敗時は一時・最終artifactを削除してrollbackする。

DBは一覧・検索・認可・正規化segmentのsource of truth、filesystemはダウンロード可能なbyte artifactのsource of truthである。どちらか片方だけを「履歴全体のsource of truth」とは扱わない。

## 維持・統合・削除

| 判定 | モジュール | 理由 |
|---|---|---|
| 維持 | `core/application.py` | 単一app factory/lifespan |
| 維持 | `core/config/` | 唯一の設定実装 |
| 維持 | `api/routes`, `api/ws` | framework境界 |
| 維持 | `services`, `repositories` | 業務処理と永続化の明確な境界 |
| 維持 | `transcription/*` | session/protocol/workerのテスト可能な境界 |
| 維持 | `runtime.py` | 長寿命resourceとリアルタイムフローのcomposition root |
| 削除済み | `legacy_app.py` | 重複app・重複routeを持つ移行集約層 |
| 削除済み | `server/config.py` | `core/config`への薄い互換export |
| 削除済み | `server.app`の互換関数/export | ASGI entrypointの責務外 |
| 削除済み | runtimeのFastAPI decorators/static mount | route source of truthとの重複 |

## 変更の置き場所

- endpointのvalidation/HTTP表現: `api/routes`
- 認証・履歴などの業務規則: `services`
- SQLAlchemy query: `repositories`
- 設定値: 該当する`core/config` dataclass、loader、`registry.py`
- WebSocket message形式: `transcription/messages.py`
- ASR実行順序・retry buffer: `transcription/worker.py`
- model provider固有処理: `openai_whisper.py`
- frontendの副作用なし処理: 該当する`web/src` module
- DOMと機能間の調停: `web/src/app.js`

薄いwrapperを追加する前に、既存の所有者へ直接置けない理由を説明する。移行aliasを追加する場合は削除条件と期限を同じPRに記録する。

## 検証

`tests/test_architecture.py` が単一app factory、route一意性、runtimeからのFastAPI登録排除、設定互換moduleの不在を検証する。通常のCIはさらにPython/JavaScriptテスト、SQLite/PostgreSQL migrationと起動、Docker healthを検証する。

## Runtime extraction boundaries (#71)

- Summary/proofread/SSE: `services/summary_service.py`; routes pass model and observer.
- OIDC HTTP flow: `services/oidc_flow.py`; transport and auth business rules remain in their existing services.
- Speaker labels: `services/diarization_service.py`; diarizer, sender and config are explicit dependencies.
- Chunk WebSocket, ASR worker composition and audio preparation: `transcription/coordinator.py`.
- Live/Qwen capture: `transcription/live.py` and `qwen_live.py`; resources come from the WebSocket route.
- Cleanup: `services/runtime_cleanup.py`; outbox processing starts after the retention transaction commits.
- Capabilities: `services/health_service.py`, used by the readiness route.
- Resource construction/lifecycle: `runtime.py`; its WebSocket function injects resources into the coordinator.

`core/runtime_resources.RuntimeResources` owns clients, active sockets and lifecycle tasks.
The composition root exposes the resource object in `app.state.runtime_resources` and passes it
explicitly to the chunk and live coordinators. Service/transcription modules cannot import runtime;
architecture tests also prohibit reintroducing extracted HTTP handlers and legacy modules.
Extracted modules are explicit mypy targets. No compatibility exports remain.
Docker/PostgreSQL CI remains required before integration.

OIDC discovery caching belongs to the resource-owned `OIDCFlow` instance; services do not retain a separate process-global discovery cache.

## Frontend controller boundaries

`web/src/app.js` is the composition and event-wiring entry. It creates a fresh
`state/store.js` store and supplies live dependencies to the feature controllers
in `web/src/controllers/`: recording, media capture, screenshot viewer,
transcription, transcript, history, auth, AI, layout, settings, glossary,
capabilities, telemetry and workspace protection. Cross-feature callbacks resolve
through the composition context so they observe current resources and state.
Controllers do not import the app and constructors do not register events or read
DOM/state; the app wires listeners once after all controllers exist. The module
entry's line budget and inert constructors are regression-tested.

`ui/notifications.js` owns banner dismissal and toast rendering.
`ui/modals.js` owns stack order, focus restoration and body scroll locking.
Browser tests instantiate these controllers independently and exercise their real
DOM behavior, in addition to existing screenshot and recording flows. Node tests
cover isolated stores, settings boundaries, recording handles and workspace locks.

`web/style.css` fixes the cascade order: tokens, base, layout, components,
utilities, workspace, responsive, then meeting refinements. The last two component
collections preserve historical theme/workspace overrides in their original
positions; moving them before responsive rules would alter the current UI.
Only identical top-level selector/declaration duplicates were removed. The runtime
stylesheet still loads after this entry, and administration styles after runtime.
No cascade layers or new design rules are introduced.

Issue #72's prerequisite is satisfied: closed Issue #82 identifies Claude's
completed refresh, commit `d50bac6` is included by merged PR #83 and is an ancestor
of main. PR #84 and later UI changes are also integrated. This separation retains
the existing state schema and UI behavior; narrower per-feature state interfaces
can be developed later without changing the composition entry.
