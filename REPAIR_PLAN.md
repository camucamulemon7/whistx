# whistx 修正計画

## 目的

現状の機能を維持しながら、以下を達成する。

- PostgreSQLを使った本番構成で確実に起動できる
- ローカル、Docker、Podmanで設定が一貫する
- 自動テストと静的解析がすべて成功する
- ゲスト文字起こしの仕様と利用制限を明確化する
- 依存関係と対応Pythonバージョンを再現可能にする
- セキュリティ境界を明確にする
- `legacy_app.py` と巨大なフロントエンド実装を段階的に分割する

## 基本方針

- 既存の未コミット変更を保護し、上書きしない。
- リリースブロッカーを先に直し、リファクタリングはその後に行う。
- 挙動変更と構造変更を同じPRに混在させない。
- 各フェーズでテストを追加し、後続作業の安全網にする。
- SQLiteは開発用、PostgreSQLは本番用として検証する。

## フェーズ1: リリースブロッカー修正

### 1. PostgreSQL依存の追加

対象:

- `requirements.txt`
- 必要に応じて `Dockerfile`
- `README.md`

作業:

- `postgresql+psycopg://` に必要な `psycopg` をランタイム依存へ追加する。
- クリーン環境でPostgreSQL用URLを指定し、アプリをimportできることを確認する。
- PostgreSQLに対してAlembic migrationを実行する。
- アプリを起動し、`/api/health` が成功することを確認する。

完了条件:

- `APP_ENV=production` と `APP_DB_URL=postgresql+psycopg://...` の構成で起動できる。
- PostgreSQL上でAlembicがheadまで適用される。
- DockerイメージにもPostgreSQLドライバが含まれる。

### 2. コンテナ環境変数の伝播修正

対象:

- `scripts/container_common.sh`
- `start.sh`
- `podman-run.sh`
- `.env.example`

最低限伝播させる設定:

```text
APP_ENV
HISTORY_RETENTION_DAYS
RUNTIME_TRANSCRIPT_RETENTION_HOURS
DEBUG_CHUNKS_RETENTION_HOURS
UNSAVED_RUNTIME_RETENTION_HOURS
ASR_API_TIMEOUT_SECONDS
SUMMARY_API_TIMEOUT_SECONDS
PROOFREAD_API_TIMEOUT_SECONDS
FFMPEG_TIMEOUT_SECONDS
ASR_PREPROCESS_ENABLED
ASR_PREPROCESS_SAMPLE_RATE
ASR_OVERLAP_MS
ASR_VAD_DROP_ENABLED
ASR_VAD_SPEECH_RATIO_MIN
ASR_RETRY_MAX_ATTEMPTS
ASR_RETRY_BASE_DELAY_MS
ASR_MULTI_PASS_ENABLED
ASR_RESCUE_RETRY_ENABLED
ASR_RESCUE_RETRY_TEMPERATURE
ASR_LIGHT_PROOFREAD_ENABLED
ASR_CONTEXT_RECENT_LINES
ASR_CONTEXT_TERM_LIMIT
APP_PROMPT_TEMPLATES
HF_HUB_DISABLE_XET
```

作業:

- コードが参照する設定とコンテナへ渡す設定を照合する。
- 個別列挙を維持する場合は、設定台帳との不一致を検出するテストを追加する。
- 可能であれば、検証済みenvファイルをコンテナへ渡す方式に整理する。

完了条件:

- ホスト側 `.env` の主要設定とコンテナ内の設定が一致する。
- コンテナ内でも `APP_ENV=production` が保持される。
- 本番モードでSQLiteを指定すると起動が拒否される。

## フェーズ2: テストの完全成功

### 3. WebSocketゲスト利用の仕様確定

推奨仕様:

- `ALLOW_GUEST_TRANSCRIPTION` を追加する。
- 開発環境ではゲスト利用を許可可能にする。
- 本番環境のデフォルトは無効にする。
- 認証済みの有効ユーザーは利用可能にする。
- ゲスト無効時の未認証接続はWebSocketコード `4401` で拒否する。

追加するテスト:

- 認証済みユーザーの接続成功
- ゲスト有効時の未認証接続成功
- ゲスト無効時の未認証接続拒否
- 無効ユーザーの接続拒否
- 期限切れセッションの接続拒否

公開ゲスト利用を許可する場合の追加対策:

- IP単位の接続数制限
- 全体の同時WebSocket数制限
- セッション最大継続時間
- セッション最大音声バイト数
- ASRリクエスト回数制限
- キュー超過時の明確な切断・再試行方針

完了条件:

- WebSocketの実装、UI、README、テストが同じ仕様を表現している。
- 未認証利用による無制限なASRコスト発生を防止できる。

### 4. 音声パイプラインテストの修正

対象:

- `tests/test_asr_pipeline.py`
- 必要に応じて音声デバッグ保存処理

作業:

- テスト用設定に不足している `debug_chunks_dir` などを一時ディレクトリで提供する。
- グローバルな `settings` の全面置換を減らす。
- 音声リトライ、結合、デバッグ保存、URL解決を個別テストへ分ける。

完了条件:

```text
python -m unittest discover -s tests -v
OK
```

## フェーズ3: 静的解析とCI

### 5. Ruffエラーの解消

修正順:

1. `F821` 未定義名
2. 実行経路に残る古いimportと到達不能コード
3. テストの `E402`
4. 互換再エクスポートの明示化またはRuff設定
5. その他の未使用importと軽微な違反

重点対象:

- `legacy_app.py` の未定義 `history_service` 参照
- `ASRChunkResult` のimport整理
- legacy側に残る重複HTTPルート
- `server/config.py` などの互換再エクスポート

完了条件:

```bash
ruff check server tests scripts
python -m compileall -q server tests scripts
```

が成功する。

### 6. CIの追加

推奨ジョブ:

1. Python 3.12でのRuffと構文チェック
2. 単体テスト
3. SQLite統合テスト
4. PostgreSQL統合テスト
5. Alembic upgrade
6. JavaScript構文チェック
7. シェル構文チェック
8. Docker build
9. 起動後のhealth check

必須チェック:

- Ruff
- 単体テスト
- PostgreSQL migration
- Docker build
- health check

## フェーズ4: 依存と設定の一貫性

### 7. Python対応バージョンの統一

推奨:

- Python 3.12を正式サポート版とする。
- README、`pyproject.toml`、Docker、CIをPython 3.12へ統一する。

Python 3.10対応を維持する場合:

- CIで3.10と3.12の両方を実行する。
- 全依存が3.10に対応することを検証する。

### 8. 依存関係の固定

作業:

- `uv.lock` またはハッシュ付きrequirementsを導入する。
- ランタイム依存と開発依存を分離する。
- Ruffなどの品質ツールを開発依存へ追加する。
- FastAPIなどの検証済み範囲を明確にする。
- 定期的な依存更新手順を用意する。

完了条件:

- ローカル、CI、コンテナが同じ依存バージョンを使用する。
- クリーン環境で同じテスト結果を再現できる。

### 9. 設定デフォルトの統一

優先して統一する設定:

- `ASR_CONTEXT_MAX_CHARS`
- `ASR_CONTEXT_RECENT_LINES`
- `ASR_CONTEXT_TERM_LIMIT`
- `LANGFUSE_ENABLED`
- `ENABLE_SELF_SIGNUP`

照合対象:

- Pythonコード
- `.env.example`
- `README.md`
- コンテナスクリプト

完了条件:

- どの起動方法でも、明示指定しない場合の値が一致する。
- 設定一覧を自動検証できる。

## フェーズ5: セキュリティ強化

### 10. 信頼プロキシ設定

追加候補:

```text
APP_PUBLIC_URL
APP_TRUST_PROXY_HEADERS
APP_TRUSTED_PROXY_IPS
APP_ALLOWED_HOSTS
```

作業:

- 信頼済みプロキシ経由の場合だけForwarded系ヘッダーを使用する。
- Keycloak callback URLは固定の `APP_PUBLIC_URL` を優先する。
- ログインレート制限で、クライアント指定のIPを無条件に信用しない。
- Hostヘッダーの許可リストを追加する。

追加するテスト:

- 未信頼クライアントの `X-Forwarded-For` が無視される。
- 未信頼クライアントの `X-Forwarded-Host` がOIDC URLへ反映されない。
- 信頼済みプロキシ経由ではHTTPSと外部ホストが正しく復元される。

### 11. セッションとAPI保護

候補:

- 状態変更APIのOrigin検証
- セッションIDのDB保存時ハッシュ化
- 全セッション失効機能
- パスワード変更時の既存セッション失効
- 管理者降格処理のトランザクション保護
- ASR、要約、校正のユーザー単位利用制限

完了条件:

- セッション窃取時の影響範囲を縮小できる。
- 管理操作の競合で管理者がゼロにならない。
- 高コストAPIを無制限に呼び出せない。

## フェーズ6: アーキテクチャ改善

### 12. `legacy_app.py` の段階的分割

推奨順:

1. WebSocket接続認証
2. `LiveSession` とセッション生成
3. チャンク解析と順序検証
4. ASRワーカー
5. diarization終了処理
6. Keycloak/OIDC
7. 古いHTTPルートの削除
8. startup/shutdown処理の分離

目標構成:

```text
server/
  api/ws/transcribe.py
  services/transcription_service.py
  services/diarization_service.py
  services/oidc_service.py
  transcription/session.py
  transcription/worker.py
  transcription/messages.py
```

ルール:

- 1回の変更で1責務だけを移動する。
- 移動前に対象の回帰テストを追加する。
- import互換層には削除期限を記録する。

### 13. `web/src/app.js` の段階的分割

分離候補:

```text
web/src/audio/capture.js
web/src/audio/vad.js
web/src/audio/mixer.js
web/src/transcription/websocket.js
web/src/transcription/session.js
web/src/transcription/render.js
web/src/summary/controller.js
web/src/proofread/controller.js
web/src/screenshots/capture.js
web/src/ui/layout.js
web/src/ui/toast.js
```

推奨順:

1. 副作用のない純粋関数
2. API/WebSocket通信
3. 音声処理
4. DOM描画
5. UI状態管理

完了条件:

- `app.js` が初期化と機能間の調停を中心とする。
- 音声、通信、描画を独立してテストできる。

## 推奨PR構成

1. PostgreSQL依存と本番起動確認
2. コンテナ環境変数の修正
3. ゲストWebSocket仕様と利用制限
4. 残りのテストエラー修正
5. Ruffクリーン化
6. CIと依存ロック
7. プロキシ・セッション・レート制限強化
8. `legacy_app.py` 分割
9. `web/src/app.js` 分割

## 最初のマイルストーン

PR 1から6までを完了し、以下を満たす。

```text
PostgreSQL起動成功
Alembic migration成功
全テスト成功
Ruff成功
JavaScript構文チェック成功
シェル構文チェック成功
Docker build成功
コンテナhealth check成功
設定値の一貫性確保
CI必須化
```

## 検証コマンド

```bash
ruff check server tests scripts
python -m compileall -q server tests scripts
python -m unittest discover -s tests -v

for file in $(rg --files web -g '*.js'); do
  node --check "$file"
done

for file in run.sh start.sh podman-run.sh entrypoint.sh scripts/*.sh; do
  bash -n "$file"
done

docker build -t whistx:repair-check .
```

PostgreSQL統合テストでは、別途テスト用PostgreSQLを起動し、次を確認する。

```bash
alembic upgrade head
python -c "import server.app"
```

最後にアプリを起動し、`/api/health` が成功することを確認する。
