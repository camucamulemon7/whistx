# whistx v2

OpenAI API (`whisper-1`) を使った、チャンク型のリアルタイム文字起こしアプリです。  
ブラウザで録音した音声を WebSocket でサーバーへ送信し、サーバーが `whisper-1` で文字起こしして結果を逐次返します。

## 1. 最短実行手順（ローカル）

### 前提
- Python 3.10+
- 最新版 Chrome / Edge
- OpenAI API キー

### 手順
```bash
cd /home/remon1129/ai/whistx
cp .env.example .env
# .env の OPENAI_API_KEY を設定
./run.sh
```

`run.sh` は `.venv` を自動作成し、依存インストール後に `uvicorn` を起動します。  
ブラウザで `http://localhost:8005` を開き、`録音開始` を押してください。

UI の既定は精度優先で、チャンク秒数は `20s`（調整範囲 `12〜30s`）です。  
`要約` ボタンで、画面上の文字起こし結果を LLM で要約できます（`SUMMARY_API_KEY` 未設定時は `OPENAI_API_KEY` を流用）。
音声ソースは `マイク / 画面共有音声 / マイク+画面共有音声` を選択できます。

## 2. Docker 実行

```bash
cd /home/remon1129/ai/whistx
cp .env.example .env
# .env の OPENAI_API_KEY を設定
./start.sh
```

`start.sh` は Docker イメージをビルドしてコンテナを起動します。

Podman の場合:

```bash
./podman-run.sh
```

## 3. 実行時設定（環境変数）

必須:

- `OPENAI_API_KEY`

設定ファイル:

- `/home/remon1129/ai/whistx/.env`
- `run.sh` / `start.sh` / `podman-run.sh` は起動時に `.env` を自動読込します

主な任意設定:

- `WHISPER_MODEL`（既定: `whisper-1`）
- `OPENAI_BASE_URL`（既定: OpenAI 公式エンドポイント）
- `SUMMARY_API_KEY`（既定: 空。未設定なら `OPENAI_API_KEY` を流用）
- `SUMMARY_BASE_URL`（既定: `OPENAI_BASE_URL` と同じ）
- `SUMMARY_MODEL`（既定: `gpt-4o-mini`）
- `SUMMARY_TEMPERATURE`（既定: `0.2`）
- `SUMMARY_INPUT_MAX_CHARS`（既定: `16000`）
- `DEFAULT_LANGUAGE`（既定: `ja`）
- `PORT`（既定: `8005`）
- `WS_PATH`（既定: `/ws/transcribe`）
- `DATA_DIR`（既定: `data/transcripts`）
- `CONTEXT_PROMPT_ENABLED`（既定: `1`。前チャンク文脈を次チャンクへ引き継ぐ）
- `CONTEXT_MAX_CHARS`（既定: `1000`。引き継ぐ文脈文字数の上限）
- `MAX_CHUNK_BYTES`（既定: `12582912`）

## 4. API / WebSocket

### Health Check

- `GET /api/health`

### 要約 API

- `POST /api/summarize`

```json
{
  "text": "文字起こし本文",
  "language": "ja"
}
```

### WebSocket

- `ws://<host>:<port>/ws/transcribe`（HTTPS の場合は `wss://`）

クライアント -> サーバー:

- `start`
```json
{"type":"start","sessionId":"sess-xxx","language":"ja","prompt":"会議用語"}
```

- `chunk`
```json
{
  "type": "chunk",
  "seq": 0,
  "offsetMs": 0,
  "durationMs": 20000,
  "mimeType": "audio/webm;codecs=opus",
  "audio": "<base64>"
}
```

- `stop`
```json
{"type":"stop"}
```

サーバー -> クライアント:

- `info`（`ready` / `stopping`）
- `final`（確定テキスト）
- `error`
- `conn`（接続数）

## 5. 出力ファイル

文字起こし結果は `data/transcripts` に保存されます。

- `*.txt`
- `*.jsonl`
- `*.srt`（JSONL から生成）

ダウンロード API:

- `/api/transcript/{session_id}.txt`
- `/api/transcript/{session_id}.jsonl`
- `/api/transcript/{session_id}.srt`

## 6. 注意事項

- `whisper-1` はネイティブな双方向ストリーミング API ではないため、短いチャンク連続送信でリアルタイム風に実現しています。
- チャンク秒数を短くすると体感遅延は減りますが、API 呼び出し回数とコストは増えます。
- デフォルトで、直前チャンクの転写末尾（`CONTEXT_MAX_CHARS` 以内）を次チャンクの `prompt` に自動連結します。
- ブラウザ側で簡易 VAD を行い、無音チャンクは送信スキップします。加えて、無音時に出やすい定型ハルシネーションはサーバー側で抑止しています。

## 7. Zoom / Webex 音声取得のコツ

- UIの「音声ソース」で `画面共有音声` か `マイク + 画面共有音声` を選びます。
- 共有ダイアログで必ず音声共有を有効にします（Chrome/Edge のタブ共有なら「タブの音声を共有」）。
- 画面共有を停止すると録音も自動停止します。
- 環境によっては会議デスクトップアプリより、ブラウザ版会議のほうが取得しやすい場合があります。
