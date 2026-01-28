# Whistx プロジェクト 改善仕様書

作成日: 2025年1月26日
バージョン: 1.0

---

## 1. 概要

本仕様書は、IMPROVEMENTS.mdに記載された改善計画と現在の実装状況のギャップ分析に基づき、次に実装すべき改善項目の優先順位、実装内容、期待される効果、見積もり工数を定義する。

**全体完了率: 70% (14/20)**

---

## 2. 優先度: 高（必須）

### 2.1 H200 GPU環境の構築 & vLLM検証

| 属性 | 値 |
|------|-----|
| **優先度** | 高 |
| **カテゴリ** | ローカルLLM統合 / パフォーマンス最適化 |
| **見積もり工数** | 3-4日 |

#### 現状
- H200 GPUへのアクセス権なし
- vLLMサーバーが起動されていない
- モデルが未ダウンロード

#### 改善内容

##### 重要: ハードウェアスペックの確認
まず、H200 GPUのVRAM容量を確認する必要があります。
- **H200標準スペック:** 141GB HBM3
- **MIG分割インスタンスの場合:** 28GBの可能性

**VRAM容量に応じたモデル選定:**

| VRAM容量 | 推奨モデル | 推論精度 | 量子化 |
|----------|-----------|----------|--------|
| 28GB | Qwen2.5 32B Int4 | 高 | Int4 |
| 28GB | Llama 3.1 8B | 中 | FP16 |
| 80GB以上 | Qwen2.5 32B FP16 | 最高 | FP16 |
| 141GB | Qwen2.5 72B Int4 | 最高 | Int4 |

1. **H200 GPU環境の構築**
   - H200 GPUへのアクセス権の取得
   - CUDA 12.4 + cuDNN 9.1の検証
   - GPUメモリ容量の確認（vLLMのvLLMエンジンで確認）
   - モデル選定の決定

2. **vLLMサーバーの起動**
   - vLLMのインストール
   - vLLMサーバーの起動スクリプト作成
   - OpenAI-compatible APIのエンドポイント確認
   - OOM（Out Of Memory）時の復旧メカニズムの実装

3. **モデルのダウンロード & 検証**
   - HuggingFaceからモデルのDL
   - モデルサイズの確認
   - 量子化の設定（Int4/Int8/FP16）
   - モデルの初期化テスト

4. **LLM分析機能のE2Eテスト**
   - vLLM + 選定モデルでの分析テスト
   - 要約、タスク、決定事項の抽出確認
   - レイテンシ計測（目標: <3秒）
   - VRAM使用量の計測

#### 実装詳細

##### 1. vLLMインストール

```bash
# vLLMのインストール
pip install vllm

# vLLMサーバーの起動
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --port 8000
```

##### 2. 起動スクリプトの作成

`scripts/start-vllm.sh`:

```bash
#!/bin/bash
# vLLMサーバー起動スクリプト

MODEL=${VLLM_MODEL:-"Qwen/Qwen2.5-72B-Instruct"}
DTYPE=${VLLM_DTYPE:-"bfloat16"}
GPU_MEMORY=${VLLM_GPU_MEMORY:-0.9}
MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}
PORT=${VLLM_PORT:-8000}

python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --dtype $DTYPE \
  --gpu-memory-utilization $GPU_MEMORY \
  --max-model-len $MAX_MODEL_LEN \
  --port $PORT
```

##### 3. 環境変数の設定

`backend/config.py`:

```python
# vLLMサーバー設定
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
VLLM_DTYPE = os.getenv("VLLM_DTYPE", "bfloat16")
VLLM_GPU_MEMORY = float(os.getenv("VLLM_GPU_MEMORY", "0.9"))
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "32768"))
```

##### 4. 健康チェックエンドポイント & OOMハンドリング

`backend/llm_service.py`:

```python
def is_available(self) -> bool:
    """vLLM APIが利用可能かチェック"""
    try:
        client = self._get_client()
        response = client.get(f"{self.api_url}/v1/models", timeout=5.0)
        available = response.status_code == 200
        if available:
            logger.info("vLLM API is available")
        else:
            logger.warning(f"vLLM API health check failed: {response.status_code}")
        return available
    except Exception as e:
        logger.warning(f"vLLM API is not available: {e}")
        return False

def analyze(self, transcripts: List[Dict]) -> Dict:
    """議事録を分析（vLLM HTTP APIを使用）"""
    if not transcripts:
        return {
            "summary": "",
            "tasks": [],
            "decisions": [],
            "next_actions": [],
        }

    if not self.is_available():
        logger.warning("vLLM API is not available, returning fallback analysis")
        return self._fallback_analysis(transcripts)

    try:
        client = self._get_client()
        messages = self._build_messages(transcripts)

        # OpenAI-compatible API形式でリクエスト
        request_body = {
            "model": self.model,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
            "response_format": {"type": "json_object"}  # JSON出力を強制
        }

        logger.info(f"Sending request to vLLM API: {len(transcripts)} transcripts")

        response = client.post(
            f"{self.api_url}/v1/chat/completions",
            json=request_body,
            timeout=REQUEST_TIMEOUT
        )

        # OOMチェック
        if response.status_code == 503:
            logger.error("vLLM API returned 503 (likely OOM)")
            return self._fallback_analysis(transcripts)

        response.raise_for_status()

        result = response.json()
        result_text = result["choices"][0]["message"]["content"]
        logger.info(f"vLLM API response received: {len(result_text)} chars")

        # JSONパース
        return self._parse_json_output(result_text)

    except httpx.HTTPStatusError as e:
        logger.error(f"vLLM API HTTP error: {e.response.status_code} {e.response.text}")
        return self._fallback_analysis(transcripts)
    except Exception as e:
        logger.error(f"vLLM API request failed: {e}")
        return self._fallback_analysis(transcripts)
```

#### 期待される効果
- LLM分析機能が利用可能になる
- 議事録作成時間が30分から2分に短縮
- レイテンシ目標 (<3秒) 達成
- 完全オンプレミス化（APIコスト $0）
- VRAM効率的な推論

#### 成功指標
- vLLMサーバーが正常に起動する
- 選定モデルが正常にロードされる
- LLM分析機能のE2Eテストが通る
- レイテンシが3秒以下である
- VRAM使用量が許容量内である
- OOM時に自動復旧する

#### バックアッププラン

| 条件 | 対応策 |
|------|--------|
| VRAM 28GB | Qwen2.5 32B Int4 (約19GB) を使用 |
| VRAM 80GB以上 | Qwen2.5 32B FP16 (約64GB) を使用 |
| VRAM 141GB | Qwen2.5 72B Int4 (約41GB) を使用 |
| 推論レイテンシ >3秒 | モデルを軽量化（32B→8B） |

---

### 2.2 レイテンシ計測 & 最適化

| 属性 | 値 |
|------|-----|
| **優先度** | 高 |
| **カテゴリ** | パフォーマンス最適化 |
| **見積もり工数** | 2日 |

#### 現状
- レイテンシが未測定
- ボトルネックが不明
- 目標値（<0.5秒）の達成状況が不明

#### 改善内容
1. **レイテンシ計測の実装**
   - 音声入力 → 表示までの各ステップの計測
   - ボトルネックの特定
   - Prometheusメトリクスの拡張

2. **最適化の実施**
   - バッファサイズの動的調整
   - WebSocket通信の圧縮
   - Whisper推論の最適化

#### 実装詳細

##### 1. レイテンシ計測の実装

`backend/transcribe_worker.py`:

```python
import time

class TranscribeWorker:
    def __init__(self, ...):
        # レイテンシ計測
        self.audio_received_ts = None
        self.vad_detected_ts = None
        self.inference_start_ts = None
        self.inference_end_ts = None
        self.sent_ts = None

    async def _run_infer_final(self, pcm16: bytes, ts: Tuple[int, int]):
        # 推論開始タイムスタンプ
        self.inference_start_ts = time.time()

        # 推論実行
        text = await loop.run_in_executor(...)

        # 推論終了タイムスタンプ
        self.inference_end_ts = time.time()

        # レイテンシ計算
        inference_latency = self.inference_end_ts - self.inference_start_ts

        logger.info(f"Inference latency: {inference_latency * 1000:.2f}ms")
```

##### 2. Prometheusメトリクスの拡張

`backend/metrics.py`:

```python
from prometheus_client import Histogram, Gauge

# レイテンシヒストグラム
audio_to_vad_latency_seconds = Histogram(
    'whistx_audio_to_vad_latency_seconds',
    'Audio to VAD detection latency'
)

vad_to_inference_latency_seconds = Histogram(
    'whistx_vad_to_inference_latency_seconds',
    'VAD to inference latency'
)

inference_to_sent_latency_seconds = Histogram(
    'whistx_inference_to_sent_latency_seconds',
    'Inference to sent latency'
)

total_latency_seconds = Histogram(
    'whistx_total_latency_seconds',
    'Total audio to display latency'
)
```

##### 3. バッファサイズの動的調整

`backend/transcribe_worker.py`:

```python
class AdaptiveBuffer:
    def __init__(self):
        self.min_buffer = 500   # ms
        self.max_buffer = 3000  # ms
        self.target_latency = 800  # ms
        self.current_buffer = 1000

    def adjust_buffer(self, processing_time: float):
        """処理時間に応じてバッファサイズを動調整"""
        if processing_time > self.target_latency:
            # 処理が遅い → バッファを増やす
            self.current_buffer = min(self.max_buffer, self.current_buffer * 1.1)
        else:
            # 処理が速い → バッファを減らす
            self.current_buffer = max(self.min_buffer, self.current_buffer * 0.95)

        logger.info(f"Adjusted buffer size to {self.current_buffer:.2f}ms")
```

##### 4. WebSocket通信の圧縮

`frontend/src/hooks/useWebSocket.js`:

```javascript
import pako from 'pako';

class OptimizedAudioSender {
  constructor(ws) {
    this.ws = ws;
    this.batch = [];
    this.batchTimer = null;
    this.batchSize = 100; // ms
  }

  send(audioData) {
    this.batch.push(audioData);

    if (!this.batchTimer) {
      this.batchTimer = setTimeout(() => {
        this.flush();
      }, this.batchSize);
    }
  }

  flush() {
    // Deflate圧縮を適用
    const compressed = pako.deflate(JSON.stringify(this.batch));
    this.ws.send(compressed);
    this.batch = [];
    this.batchTimer = null;
  }
}
```

#### 期待される効果
- レイテンシが可視化される
- ボトルネックが特定される
- レイテンシ目標（<0.5秒）の達成
- ネットワーク帯域の削減

#### 成功指標
- 全体レイテンシが0.5秒以下である
- 各ステップのレイテンシが計測されている
- ボトルネックが特定されている
- Prometheusメトリクスが収集されている

---

### 2.3 E2Eテストの実装

| 属性 | 値 |
|------|-----|
| **優先度** | 高 |
| **カテゴリ** | テスト |
| **見積もり工数** | 3日 |

#### 現状
- 自動化テストが実装されていない
- 回帰テストがない
- 重大なリグレッションを防ぐ仕組みがない

#### 改善内容
1. **バックエンドのE2Eテスト**
   - WebSocket接続テスト
   - 音声認識テスト
   - LLM分析テスト

2. **フロントエンドのE2Eテスト**
   - Reactコンポーネントのテスト
   - WebSocket通信のテスト
   - ユーザーインタラクションのテスト

3. **回帰テスト**
   - 既存機能のテストスイート
   - 定期的なテスト実行

#### 実装詳細

##### 1. バックエンドのE2Eテスト

`tests/test_e2e.py`:

```python
import pytest
import asyncio
import websockets
from backend.app import app

@pytest.mark.asyncio
async def test_websocket_connection():
    """WebSocket接続テスト"""
    # クライアント接続
    ws = await websockets.connect('ws://localhost:8005/ws/transcribe')

    # startコマンド送信
    await ws.send(json.dumps({"type": "start"}))

    # レスポンス受信
    response = await ws.recv()
    data = json.loads(response)

    # 検証
    assert data["type"] == "info"
    assert data["state"] == "ready"

    # 切断
    await ws.close()

@pytest.mark.asyncio
async def test_audio_transcription():
    """音声認識テスト"""
    # クライアント接続
    ws = await websockets.connect('ws://localhost:8005/ws/transcribe')

    # startコマンド送信
    await ws.send(json.dumps({"type": "start"}))

    # テスト音声送信
    test_audio = load_test_audio("test_ja.wav")
    await ws.send(test_audio)

    # レスポンス受信
    response = await ws.recv()
    data = json.loads(response)

    # 検証
    assert data["type"] in ["partial", "final"]

    # 切断
    await ws.close()
```

##### 2. フロントエンドのE2Eテスト

`tests/frontend/App.test.jsx`:

```jsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import App from '../../src/App'

describe('App', () => {
  test('レンダリングテスト', () => {
    render(<App />)
    expect(screen.getByText('Whistx v2')).toBeInTheDocument()
  })

  test('録音開始ボタンクリックテスト', async () => {
    render(<App />)
    const button = screen.getByText('録音開始')
    fireEvent.click(button)
    await waitFor(() => {
      expect(screen.getByText('停止')).toBeInTheDocument()
    })
  })
})
```

##### 3. 回帰テスト

`tests/test_regression.py`:

```python
import pytest

@pytest.mark.regression
def test_whisper_model_loading():
    """Whisperモデルのロードテスト"""
    from backend.transcription import get_transcriber
    transcriber = get_transcriber()
    assert transcriber is not None
    assert transcriber.model is not None

@pytest.mark.regression
def test_vad_initialization():
    """VAD初期化テスト"""
    from backend.adaptive_vad import get_vad
    vad = get_vad()
    assert vad is not None
    assert vad.model is not None

@pytest.mark.regression
def test_diarizer_initialization():
    """話者ダイアライザー初期化テスト"""
    from backend.diarizer import GlobalDiarizer
    diarizer = GlobalDiarizer.get()
    assert diarizer is not None
```

##### 4. テスト実行スクリプト

`scripts/run-tests.sh`:

```bash
#!/bin/bash
# テスト実行スクリプト

# バックエンドのテスト
echo "Running backend tests..."
pytest tests/ -v

# フロントエンドのテスト
echo "Running frontend tests..."
cd frontend
npm test

# E2Eテスト
echo "Running E2E tests..."
pytest tests/test_e2e.py -v
```

#### 期待される効果
- 自動化テストが実装される
- 回帰テストによりリグレッションを防げる
- コードの品質が向上する
- CI/CDパイプラインに統合できる

#### 成功指標
- 全てのテストが通る
- 回帰テストが定期実行されている
- CI/CDパイプラインに統合されている

---

## 3. 優先度: 中（推奨）

### 3.1 WebSocket自動再接続

| 属性 | 値 |
|------|-----|
| **優先度** | 中 |
| **カテゴリ** | UI/UX改善 |
| **見積もり工数** | 1日 |

#### 現状
- WebSocket切断時の自動再接続が実装されていない
- ネットワーク不安定時に問題が発生する

#### 改善内容
1. **自動再接続ロジックの実装**
   - 切断時の自動再接続
   - 再接続間隔の指数バックオフ
   - 状態の保持

#### 実装詳細

##### 1. 自動再接続ロジックの実装

`frontend/src/hooks/useWebSocket.js`:

```javascript
import { useCallback, useState, useEffect, useRef } from 'react'

export function useWebSocket(url, options = {}) {
  const [isConnected, setIsConnected] = useState(false)
  const [reconnectAttempts, setReconnectAttempts] = useState(0)
  const wsRef = useRef(null)
  const reconnectTimerRef = useRef(null)
  const maxReconnectAttempts = options.maxReconnectAttempts || 10
  const reconnectDelay = options.reconnectDelay || 1000 // ms

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    try {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        setReconnectAttempts(0)
      }

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason)
        setIsConnected(false)
        wsRef.current = null

        // 自動再接続
        if (reconnectAttempts < maxReconnectAttempts) {
          const delay = reconnectDelay * Math.pow(2, reconnectAttempts)
          console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts + 1}/${maxReconnectAttempts})`)
          reconnectTimerRef.current = setTimeout(() => {
            setReconnectAttempts(prev => prev + 1)
            connect()
          }, delay)
        } else {
          console.error('Max reconnect attempts reached')
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
      }

    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
    }
  }, [url, reconnectAttempts, maxReconnectAttempts, reconnectDelay])

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsConnected(false)
  }, [])

  return {
    isConnected,
    reconnectAttempts,
    connect,
    disconnect,
  }
}
```

#### 期待される効果
- ネットワーク不安定時に自動復旧する
- ユーザー体験が向上する
- システムの信頼性が向上する

#### 成功指標
- 切断時に自動再接続する
- 再接続回数が上限を超えない
- 再接続間隔が指数バックオフする

---

### 3.2 UI/UXの微調整

| 属性 | 値 |
|------|-----|
| **優先度** | 中 |
| **カテゴリ** | UI/UX改善 |
| **見積もり工数** | 2日 |

#### 現状
- 音声位置へのジャンプ機能が未実装
- スムーズなアニメーションが不完全

#### 改善内容
1. **音声位置へのジャンプ機能**
   - タイムスタンプクリックで音声位置へジャンプ
   - 録音再生機能

2. **スムーズなアニメーション**
   - CSSアニメーションの最適化
   - ローディングインジケーター

#### 実装詳細

##### 1. 音声位置へのジャンプ機能

`frontend/src/App.jsx`:

```jsx
function App() {
  const [currentPosition, setCurrentPosition] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  const handleSeekTo = (timestamp) => {
    // 音声位置へのジャンプ
    setCurrentPosition(timestamp)
    // 録音再生ロジックを追加
  }

  return (
    <div>
      <TranscriptList
        transcripts={transcripts}
        onSeekTo={handleSeekTo}
      />
      <AudioPlayer
        position={currentPosition}
        isPlaying={isPlaying}
        onPlayPause={() => setIsPlaying(!isPlaying)}
      />
    </div>
  )
}

function TranscriptItem({ transcript, onSeekTo }) {
  const handleClick = () => {
    onSeekTo(transcript.tsStart)
  }

  return (
    <div onClick={handleClick} className="cursor-pointer">
      <span>{formatTime(transcript.tsStart)}</span>
      <span>{transcript.text}</span>
    </div>
  )
}
```

##### 2. スムーズなアニメーション

`frontend/src/App.css`:

```css
/* フェードインアニメーション */
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-fade-in {
  animation: fadeIn 0.3s ease-in-out;
}

/* ローディングインジケーター */
@keyframes spin {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

.animate-spin {
  animation: spin 1s linear infinite;
}
```

#### 期待される効果
- 音声位置へのジャンプができる
- UIがスムーズに動作する
- ユーザー体験が向上する

#### 成功指標
- タイムスタンプクリックで音声位置へジャンプする
- アニメーションがスムーズに動作する
- ローディングインジケーターが表示される

---

## 4. 優先度: 低（将来実装）

### 4.1 WebSocket通信圧縮

| 属性 | 値 |
|------|-----|
| **優先度** | 低 |
| **カテゴリ** | パフォーマンス最適化 |
| **見積もり工数** | 1日 |

#### 改善内容
- Deflate圧縮の実装
- バッチ処理の最適化

#### 実装詳細
- `pako`ライブラリを使用してWebSocket通信を圧縮
- 詳細は2.2の「WebSocket通信の圧縮」を参照

#### 期待される効果
- ネットワーク帯域が60%削減
- レイテンシが100ms改善

---

### 4.2 ドキュメント作成

| 属性 | 値 |
|------|-----|
| **優先度** | 低 |
| **カテゴリ** | ドキュメント |
| **見積もり工数** | 2日 |

#### 改善内容
1. **ユーザーマニュアル**
   - インストール手順
   - 使用方法
   - トラブルシューティング

2. **デプロイガイド**
   - Docker/Podmanデプロイ
   - GPU環境の構築
   - 設定のカスタマイズ

#### 実装詳細
- Markdown形式でドキュメントを作成
- 画像やスクリーンショットを追加

#### 期待される効果
- ユーザーが簡単に使用できる
- 障害発生時に自己解決できる

---

### 4.3 リアクション、絵文字

| 属性 | 値 |
|------|-----|
| **優先度** | 低 |
| **カテゴリ** | UI/UX改善 |
| **見積もり工数** | 1日 |

#### 改善内容
- チャット形式でのリアクション追加
- 絵文字の使用

#### 実装詳細
- Reactコンポーネントとしてリアクションボタンを実装
- 絵文字ピッカーを追加

#### 期待される効果
- コミュニケーションが豊かになる
- ユーザー体験が向上する

---

## 5. まとめ

### 優先度別作業量

| 優先度 | 項目数 | 合計工数 |
|--------|--------|----------|
| 高 | 3 | 7日 |
| 中 | 2 | 3日 |
| 低 | 3 | 4日 |
| **合計** | **8** | **14日** |

### 実装ロードマップ

| 週 | タスク |
|----|--------|
| **第1週** | H200 GPU環境の構築 & vLLM検証 (2日) <br> レイテンシ計測 & 最適化 (2日) <br> WebSocket自動再接続 (1日) |
| **第2週** | E2Eテストの実装 (3日) <br> UI/UXの微調整 (2日) |
| **第3週** | WebSocket通信圧縮 (1日) <br> ドキュメント作成 (2日) <br> リアクション、絵文字 (1日) |

### 成功指標

| 指標 | 目標 |
|------|------|
| H200 GPU環境の構築 | H200 GPUでvLLMが動作 |
| レイテンシ | <0.5秒 |
| LLM分析レイテンシ | <3秒 |
| E2Eテスト | 全テストが通る |
| ドキュメント | ユーザーマニュアル完備 |

---

## 6. セキュリティ/認証（重要追加）**codexレビュー反映: 優先度「高」**

### 6.1 WebSocket認証

#### 現状
- WebSocketエンドポイントに認証がない
- だれでも接続可能

#### 改善内容
1. **トークンベース認証の実装**
   - WebSocket接続時にトークンを要求
   - トークンの検証
   - 期限切れのトークンの拒否

#### 実装詳細

##### 1. トークンベース認証の実装

`backend/app.py`:

```python
from fastapi import WebSocket, WebSocketDisconnect, status
from typing import Optional
import jwt

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"

def verify_websocket_token(token: str) -> Optional[dict]:
    """WebSocketトークンの検証"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

@app.websocket(config.WS_PATH)
async def ws_transcribe(ws: WebSocket):
    # トークン検証
    token = ws.query_params.get("token")
    if not token:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Token required")
        return

    payload = verify_websocket_token(token)
    if not payload:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
        return

    # 接続受け入れ
    await ws.accept()
    # ... 以下は既存の処理
```

##### 2. トークン生成エンドポイント

`backend/app.py`:

```python
from pydantic import BaseModel

class TokenRequest(BaseModel):
    username: str
    password: str

@app.post("/api/token")
async def create_token(request: TokenRequest):
    """トークン生成（簡易版）"""
    # TODO: 実際のユーザー認証ロジック
    if request.username == "admin" and request.password == "password":
        payload = {
            "sub": request.username,
            "exp": datetime.utcnow() + timedelta(hours=24),
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        return {"access_token": token, "token_type": "bearer"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")
```

#### 期待される効果
- 不正アクセスが防止される
- セキュリティが向上する

---

## 7. 補足

### 技術的負債

以下の技術的負債の解消も推奨されます：

1. **transcription.py と asr_backends.py の統合**
   - 現在パラレル実装になっている
   - どちらかを廃止するか統合する

2. **レガシーコードの削除**
   - **server/** と **web/** ディレクトリの削除
   - バックアップ推奨

3. **環境変数の一元化**
   - 設定ファイルの整理
   - ドキュメントの更新
