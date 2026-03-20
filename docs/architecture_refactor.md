# whistx アーキテクチャ・リファクタ診断メモ

## 1. 現状診断サマリ

### 責務マップ

| 領域 | 現在の主要配置 | 備考 |
| --- | --- | --- |
| HTTP API | `server/api/routes/*.py` | `auth/admin/history` は service 直結に移行。Keycloak callback など一部のみ `server/legacy_app.py` を経由。 |
| WebSocket セッション管理 | `server/api/ws/transcribe.py` | 入口のみ分離。セッションワーカー本体は `server/legacy_app.py`。 |
| ASR 呼び出し | `server/openai_whisper.py`, `server/asr.py` | 現行ビルドは Whisper 互換のみ使用。 |
| 音声前処理 | `server/audio_pipeline.py` | `ffmpeg` ベース。 |
| 話者分離 | `server/diarizer.py` | `pyannote.audio` を利用。 |
| 履歴保存 | `server/services/history_service.py`, `server/repositories/history_repository.py`, `server/transcript_store.py` | DB とファイルの二層。 |
| 認証/認可 | `server/services/auth_service.py`, `server/services/admin_service.py`, `server/auth.py`, `server/deps.py` | cookie/session・認可・admin 操作を整理。 |
| UI 状態管理 | `web/src/app.js`, `web/src/auth/session.js`, `web/src/history/state.js`, `web/src/ui/theme.js` | エントリー薄化後、auth/history/theme helper を追加分離。 |
| UI API 呼び出し | `web/src/api/client.js`, `web/src/auth/api.js`, `web/src/history/api.js`, `web/src/capabilities/api.js` | API 呼び出しを ES modules に分離。 |
| 外部依存 | OpenAI 互換 ASR / LLM / Keycloak / Langfuse / SQLite / filesystem | 境界定義は README と本メモに追記。 |

### 主要ボトルネック

- `server/app.py` は薄いエントリーのまま維持し、ルーティングは `server/core/application.py` に集約。
- `web/main.js` は薄いエントリーのまま維持し、実装本体は `web/src/app.js` 配下に寄せた。
- `server/config.py` は `server/core/config/` に分割済み。互換 import は維持。
- `server/legacy_app.py` は WS / ASR / Keycloak callback など未移行領域に限定しつつ縮小中。

## 2. 死んだコード・不要コード候補

### 判定済み

| 対象 | 判定 | 根拠 |
| --- | --- | --- |
| `server/voxtral_realtime.py` | 削除済み | `_build_transcriber_factory()` が realtime ASR を常に拒否し、README も非対応。参照経路なし。 |
| `server/openai_whisper.py` | 使用中 | 現行 transcriber factory がこちらのみ利用。 |
| `server/api/legacy_app.py` | 削除済み | 中間生成物だったため削除。 |
| `APP_ENTRYPOINT` | 互換維持中 | 起動スクリプトからのみ参照。 |
| `WHISPER_MODEL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL` | 互換維持中 | config / 起動スクリプトのフォールバック用途。 |
| `HOST`, `PORT`, `WS_PATH`, `DATA_DIR` | 互換維持中 | 旧 env 別名。 |
| `UI_BANNERS`, `WEBUI_BANNERS`, `APP_SOC_PROMPT_TEMPLATE` | 互換維持中 | UI 設定の旧形式入力。 |

### README と実装のズレ

- README の「Realtime ASR models are not supported」は実装と整合している。
- 依存 source of truth は `requirements.txt` / `requirements-diarization.txt` であり、`pyproject.toml` は uv tool metadata として明記した。

## 3. 環境変数台帳

### 使用中

- `ASR_*`
- `SUMMARY_*`
- `PROOFREAD_*`
- `APP_HOST`, `APP_PORT`, `APP_WS_PATH`, `APP_DATA_DIR`, `APP_TRANSCRIPTS_DIR`, `APP_DB_URL`, `APP_SESSION_SECRET`, `APP_SESSION_DAYS`, `APP_HISTORY_DIR`
- `ENABLE_SELF_SIGNUP`
- `KEYCLOAK_*`
- `DIARIZATION_*`
- `LANGFUSE_*`
- `APP_BRAND_*`, `APP_UI_BANNERS_TEXT`, `APP_UI_BANNERS`, `APP_PROMPT_TEMPLATES`

### 非推奨だが互換維持中

- `OPENAI_API_KEY`, `OPENAI_BASE_URL`
- `WHISPER_MODEL`
- `HOST`, `PORT`, `WS_PATH`, `DATA_DIR`
- `DEFAULT_LANGUAGE`, `DEFAULT_PROMPT`, `DEFAULT_TEMPERATURE`
- `CONTEXT_PROMPT_ENABLED`, `CONTEXT_MAX_CHARS`, `MAX_QUEUE_SIZE`, `MAX_CHUNK_BYTES`
- `UI_BANNERS`, `WEBUI_BANNERS`
- `APP_SOC_PROMPT_TEMPLATE`
- `LANGFUSE_ENV`

### 未使用または削除候補

- `server/voxtral_realtime.py` は今回削除済み。
- `pyproject.toml` を依存 source of truth とみなす運用は未採用と明文化した。

## 4. 履歴保存の責務境界

### Source of truth

- DB: 検索・一覧・権限制御・履歴メタデータ・セグメント正規化済み表現。
- ファイル: 元の transcript artifact (`txt`, `jsonl`, `zip`, screenshot, metadata)。

### 保存時の境界

- `server/transcript_store.py`: ランタイム中の一時 transcript と screenshot を保持。
- `server/services/history_service.py`: finalized runtime transcript を検証し、artifact を履歴領域へコピーする。
- `server/repositories/history_repository.py`: `TranscriptHistory` / `TranscriptSegment` の DB access を担当。

### 失敗時

- `history_service.save_history()` は temp artifact directory を使い、途中失敗時は temp と artifact の両方を削除する。
- DB flush 後でも rename 前に失敗した場合に備え、例外時 cleanup を実施する。

### path traversal 対策

- ランタイム transcript は `resolve_transcript_path()` / `resolve_screenshot_path()` で root 配下を検証。
- 履歴 artifact は `history_service.get_history_file_path()` と `resolve_history_screenshot_path()` で root 相対パスのみ受け入れる。

## 5. 新しいディレクトリ構成

```text
server/
  app.py
  legacy_app.py
  auth.py
  repositories/
    history_repository.py
    session_repository.py
    user_repository.py
  services/
    admin_service.py
    auth_service.py
    history_service.py
  core/
    application.py
    security.py
    config/
      __init__.py
      app.py
      asr.py
      auth.py
      base.py
      diarization.py
      observability.py
      ui.py
  api/
    routes/
      admin.py
      auth.py
      health.py
      history.py
      summary.py
      transcript.py
    ws/
      transcribe.py
web/
  main.js
  admin.js
  src/
    app.js
    bootstrap.js
    api/
      client.js
    auth/
      api.js
      session.js
    capabilities/
      api.js
    history/
      api.js
      state.js
    state/
      storage.js
    ui/
      theme.js
```

## 6. 削除したコード一覧と理由

- `server/api/legacy_app.py`: 一時的な中間ファイルだったため削除。
- `server/voxtral_realtime.py`: 現行ビルドで到達不能かつ README でも非対応のため削除。

## 7. 主要な責務分割の説明

- `server/app.py`: FastAPI アプリ生成と互換公開のみ。
- `server/core/application.py`: startup/shutdown と router/static 登録を一元化。
- `server/api/routes/auth.py`: login/bootstrap/register/logout の HTTP 入口。Keycloak のみ移行互換で `legacy_app` 利用。
- `server/api/routes/admin.py`: pending user / user role 管理の HTTP 入口。
- `server/api/routes/history.py`: 履歴保存・一覧・詳細・download の HTTP 入口。
- `server/services/auth_service.py`: 認証状態、login rate limit、register/bootstrap/logout、Keycloak user upsert。
- `server/services/admin_service.py`: admin 操作の業務ロジック。
- `server/services/history_service.py`: runtime artifact 検証、history 保存、download path 解決。
- `server/repositories/*.py`: SQLAlchemy access の境界。
- `web/main.js`: エントリーのみ。
- `web/src/auth/session.js`: guest mode と auth label helper。
- `web/src/history/state.js`: history selection/list state helper。
- `web/src/ui/theme.js`: theme 初期化用 helper。

## 8. 移行リスク

- `server/legacy_app.py` には WS / ASR / Keycloak callback / summarize / proofread がまだ残る。
- `web/src/app.js` はなお巨大で、audio/ws/dom rendering の分割余地がある。
- 起動スクリプトの共通化は実施済みだが、CI や packaging を含む完全統一は今後の余地がある。

## 9. README / 開発者向け運用メモ更新案

- バックエンド入口は `server/app.py`、移行中の残存ロジックは `server/legacy_app.py` とする。
- 認証や履歴の DB access は `server/repositories/` に追加する。
- 依存追加時は `requirements.txt` または `requirements-diarization.txt` を更新し、`pyproject.toml` は tool metadata に留める。
- 新機能追加時は route handler にロジックを増やさず、service / repository / frontend helper へ寄せる。
