# whistx アーキテクチャ・リファクタ診断メモ

## 1. 現状診断サマリ

### 責務マップ

| 領域 | 現在の主要配置 | 備考 |
| --- | --- | --- |
| HTTP API | `server/api/routes/*.py` | 入口をルート単位に分離。実処理は移行期間中 `server/legacy_app.py` に委譲。 |
| WebSocket セッション管理 | `server/api/ws/transcribe.py` | 入口のみ分離。セッションワーカー本体は `server/legacy_app.py`。 |
| ASR 呼び出し | `server/openai_whisper.py`, `server/asr.py` | 現行ビルドは Whisper 互換のみ使用。 |
| 音声前処理 | `server/audio_pipeline.py` | `ffmpeg` ベース。 |
| 話者分離 | `server/diarizer.py` | `pyannote.audio` を利用。 |
| 履歴保存 | `server/services/history_service.py`, `server/transcript_store.py` | DB とファイルの二層。 |
| 認証/認可 | `server/auth.py`, `server/deps.py`, `server/api/routes/auth.py`, `server/api/routes/admin.py` | HTTP 層と認証ドメインの入口は分離済み。 |
| UI 状態管理 | `web/src/app.js` | 旧巨大実装を移送。`web/main.js` は薄いエントリー化。 |
| UI API 呼び出し | `web/src/api/client.js`, `web/admin.js` | 管理画面は共通クライアント利用に変更。 |
| 外部依存 | OpenAI 互換 ASR / LLM / Keycloak / Langfuse / SQLite / filesystem | 境界定義は README と本メモに追記。 |

### 主要ボトルネック

- `server/app.py` は 2275 行から薄いエントリーへ変更し、初期化と互換公開だけに縮小。
- `web/main.js` は 4268 行から薄いエントリーへ変更し、実装本体を `web/src/app.js` へ移送。
- `server/config.py` は `server/core/config/` に分割し、互換ラッパー化。

## 2. 死んだコード・不要コード候補

### 参照実態を確認した候補

| 対象 | 判定 | 根拠 |
| --- | --- | --- |
| `server/voxtral_realtime.py` | 隔離候補 | `server/legacy_app.py::_build_transcriber_factory()` では realtime ASR を常に拒否し、README も非対応と明記。現行コードパスでは未使用。 |
| `server/openai_whisper.py` | 使用中 | 現行 transcriber factory が常にこちらを返す。 |
| `APP_ENTRYPOINT` | 互換維持中 | アプリ本体では未使用。`run.sh`/`start.sh`/`podman-run.sh`/`entrypoint.sh` からのみ参照。 |
| `WHISPER_MODEL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL` | 互換維持中 | `server/core/config/asr.py` と起動スクリプトでフォールバック用途に使用。 |
| `HOST`, `PORT`, `WS_PATH`, `DATA_DIR` | 互換維持中 | スクリプト/設定の旧別名としてのみ使用。 |
| `UI_BANNERS`, `WEBUI_BANNERS`, `APP_SOC_PROMPT_TEMPLATE` | 互換維持中 | UI 設定の旧形式入力としてのみ使用。 |
| `server/api/legacy_app.py` | 未使用 | 途中生成物。現状の import 参照先ではないため削除候補。 |

### README と実装のズレ

- README の「Realtime ASR models are not supported」は実装と整合している。
- `pyproject.toml` は依存 source of truth ではなく、現状はメタデータ用途に留まる。
- 依存の実 source of truth は `requirements.txt` と `requirements-diarization.txt`。

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

- `server/api/legacy_app.py` に対応する設定はなし。ファイル自体が削除候補。
- `pyproject.toml` 内の依存 source of truth 想定は実装上未使用。

## 4. 履歴保存の責務境界

### Source of truth

- DB: 検索・一覧・権限制御・履歴メタデータ・セグメント正規化済み表現。
- ファイル: 元の transcript artifact (`txt`, `jsonl`, `zip`, screenshot, metadata)。

### 保存時の境界

- `server/transcript_store.py`: ランタイム中の一時 transcript と screenshot を保持。
- `server/services/history_service.py`: finalized runtime transcript を検証し、ユーザー履歴領域へコピーし、DB レコードを生成。

### 失敗時

- `history_service.save_history()` は temp artifact directory を使い、途中失敗時は temp と artifact の両方を削除する。
- DB flush 後でも rename 前に失敗した場合に備え、例外時 cleanup を実施する。

### path traversal 対策

- ランタイム transcript は `resolve_transcript_path()` / `resolve_screenshot_path()` で root 配下を検証。
- 履歴 artifact は `history_service` 側の path 解決で root 相対パスのみ受け入れる。

## 5. 新しいディレクトリ構成

```text
server/
  app.py
  legacy_app.py
  core/
    application.py
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
    state/
      storage.js
```

## 6. 削除したコード一覧と理由

- 既存コードの本削除は未実施。
- 理由: まずは外部仕様を壊さず責務を分離し、未使用判定が確実なものだけ次段で削除する方針。
- 隔離したコード: `server/legacy_app.py`, `web/src/app.js`。
- 削除候補: `server/voxtral_realtime.py`, `server/api/legacy_app.py`。

## 7. 主要な責務分割の説明

- `server/app.py`: FastAPI アプリ生成と後方互換 API 公開のみ。
- `server/core/application.py`: startup/shutdown と router/static 登録を一元化。
- `server/api/routes/*.py`: ルート単位の薄い HTTP 入口。
- `server/core/config/*.py`: 設定責務をドメイン別に分割。
- `web/main.js`: エントリーのみ。
- `web/admin.js`: API 共通クライアントを使う独立ページ。

## 8. 移行リスク

- `server/legacy_app.py` への委譲が残っているため、完全な service 分割は未完了。
- `web/src/app.js` はまだ巨大で、audio/ws/auth/history/ui の内部分割余地が大きい。
- 起動スクリプトは重複が残っており、source of truth の一本化は文書化先行。
- `server/api/legacy_app.py` が未削除のため、混乱防止の cleanup が別途必要。

## 9. README / 開発者向け運用メモ更新案

- バックエンド入口は `server/app.py`、移行期間中の旧実装集約は `server/legacy_app.py` と明記する。
- 設定の追加先は `server/core/config/` とする。
- 依存追加時は `requirements.txt` または `requirements-diarization.txt` を source of truth とし、`pyproject.toml` はメタデータ扱いにする。
- 新機能追加時は route handler にロジックを増やさず、`legacy_app.py` から service へ切り出す。
