# backend/llm_service.py
"""
vLLM HTTP APIクライアント
会議の要約・タスク抽出・決定事項の分析

別プロセスで起動したvLLMサーバーのHTTP APIを使用
"""

import os
import json
import logging
from typing import List, Dict, Optional, AsyncGenerator
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)

# 環境変数で設定を上書き可能
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8000")  # vLLM APIのエンドポイント
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "120.0"))  # タイムアウト（秒）

# システムプロンプト
DEFAULT_SYSTEM_PROMPT = """あなたは優秀な議事録作成者です。
会議内容から以下を抽出してください：
1. 要約（3段落程度、箇条書き）
2. タスク一覧（担当者含む）
3. 決定事項
4. 次回のアクションアイテム

出力は以下のJSON形式で：
{
  "summary": "...",
  "tasks": [{"text": "...", "assignee": "..."}],
  "decisions": ["...", "..."],
  "next_actions": ["...", "..."]
}
"""


class LocalLLMService:
    """
    vLLM HTTP APIクライアント

    別プロセスで起動したvLLMサーバーのOpenAI-compatible APIを使用
    """

    def __init__(
        self,
        api_url: str = None,
        model: str = None,
        system_prompt: str = None,
    ):
        """
        LLMサービスの初期化

        Args:
            api_url: vLLM APIのエンドポイント (default: 環境変数 LLM_API_URL)
            model: モデル名 (default: 環境変数 LLM_MODEL)
            system_prompt: システムプロンプト (default: DEFAULT_SYSTEM_PROMPT)
        """
        self.api_url = (api_url or LLM_API_URL).rstrip('/')
        self.model = model or LLM_MODEL
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.client = None

        logger.info(f"LLM Service initialized: API={self.api_url}, model={self.model}")

    def _get_client(self) -> httpx.Client:
        """HTTPクライアントを取得（遅延初期化）"""
        if self.client is None:
            self.client = httpx.Client(timeout=REQUEST_TIMEOUT)
        return self.client

    def is_available(self) -> bool:
        """vLLM APIが利用可能かチェック"""
        try:
            client = self._get_client()
            # モデルリスト取得でヘルスチェック
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

    def _build_messages(self, transcripts: List[Dict]) -> List[Dict]:
        """OpenAIフォーマットのメッセージを構築"""
        # 直近の議事録をテキスト化
        text = "\n".join([
            f"[{t.get('timestamp', '')}] {t.get('speaker', '')}: {t.get('text', '')}"
            for t in transcripts
        ])

        # OpenAIフォーマット（システムプロンプト + ユーザーメッセージ）
        return [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": f"会議内容:\n{text}"
            }
        ]

    def analyze(self, transcripts: List[Dict]) -> Dict:
        """
        議事録を分析（vLLM HTTP APIを使用）

        Args:
            transcripts: トランスクリプトのリスト

        Returns:
            分析結果 (summary, tasks, decisions, next_actions)
        """
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

    def _parse_json_output(self, text: str) -> Dict:
        """LLM出力からJSONを抽出"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # JSONマーカーで抽出
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

            # パース失敗時はフォールバック
            logger.warning("Failed to parse JSON from LLM output")
            return {
                "summary": text,
                "tasks": [],
                "decisions": [],
                "next_actions": [],
            }

    def _fallback_analysis(self, transcripts: List[Dict]) -> Dict:
        """vLLM APIが使えない場合の簡易分析"""
        # 簡単な統計情報
        total_speakers = len(set(t.get('speaker', 'Unknown') for t in transcripts))
        total_words = sum(len(t.get('text', '').split()) for t in transcripts)

        # タイムスタンプでソート
        sorted_transcripts = sorted(
            transcripts,
            key=lambda x: x.get('timestamp', 0)
        )

        return {
            "summary": f"会議の記録があります（{len(transcripts)}発話、{total_speakers}名の発言者、計{total_words}語）",
            "tasks": [],
            "decisions": [],
            "next_actions": [],
            "_meta": {
                "total_transcripts": len(transcripts),
                "total_speakers": total_speakers,
                "total_words": total_words,
                "llm_unavailable": True,
            }
        }

    def close(self):
        """HTTPクライアントをクローズ"""
        if self.client:
            self.client.close()
            self.client = None


# グローバルインスタンス
_llm_service_instance: Optional[LocalLLMService] = None


def get_llm_service() -> LocalLLMService:
    """LLMサービスのシングルトンインスタンスを取得"""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LocalLLMService()
    return _llm_service_instance
