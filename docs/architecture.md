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
| OIDC transport | `server/services/oidc_service.py` | route/runtimeはcookieとHTTPフローを調停 |
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
