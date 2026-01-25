// frontend/src/hooks/useWebSocket.js
import { useRef, useCallback, useState, useEffect } from 'react'

/**
 * WebSocket通信カスタムフック
 * バイナリ音声データとJSONコマンドの送受信
 */
export function useWebSocket(url, options = {}) {
  const wsRef = useRef(null)
  const [isConnected, setIsConnected] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [status, setStatus] = useState('disconnected')

  // 受信メッセージのコールバック
  const onMessageCallbackRef = useRef(null)
  const onErrorCallbackRef = useRef(null)
  const onCloseCallbackRef = useRef(null)

  /**
   * WebSocket接続を開始
   */
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected')
      return
    }

    try {
      // プロトコルの判定
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const wsUrl = url || `${proto}://${window.location.host}/ws/transcribe`

      console.log('Connecting to WebSocket:', wsUrl)
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.binaryType = 'arraybuffer'

      ws.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        setStatus('connected')
      }

      ws.onmessage = (event) => {
        const data = event.data

        // テキストメッセージ (JSON)
        if (typeof data === 'string') {
          try {
            const message = JSON.parse(data)
            console.log('WebSocket received:', message)

            // セッションIDの保存
            if (message.type === 'info' && message.sessionId) {
              setSessionId(message.sessionId)
            }

            // コールバックを実行
            if (onMessageCallbackRef.current) {
              onMessageCallbackRef.current(message)
            }
          } catch (e) {
            console.error('Failed to parse WebSocket message:', e)
          }
        }
        // バイナリメッセージ
        else {
          console.log('Received binary data:', data.byteLength, 'bytes')
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setStatus('error')
        if (onErrorCallbackRef.current) {
          onErrorCallbackRef.current(error)
        }
      }

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason)
        setIsConnected(false)
        setStatus('disconnected')
        wsRef.current = null

        if (onCloseCallbackRef.current) {
          onCloseCallbackRef.current(event)
        }
      }

    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
      setStatus('error')
      throw error
    }
  }, [url])

  /**
   * WebSocket接続を閉じる
   */
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsConnected(false)
    setStatus('disconnected')
  }, [])

  /**
   * 録音開始コマンドを送信
   */
  const startRecording = useCallback((opts = {}) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected')
      return false
    }

    const message = {
      type: 'start',
      sessionId: sessionId || undefined,
      opts: {
        language: 'ja',
        asrBackend: 'faster-whisper',
        vadBackend: 'silero',
        ...opts,
      },
    }

    wsRef.current.send(JSON.stringify(message))
    console.log('Sent start command:', message)
    return true
  }, [sessionId])

  /**
   * 録音停止コマンドを送信
   */
  const stopRecording = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected')
      return false
    }

    const message = {
      type: 'stop',
    }

    wsRef.current.send(JSON.stringify(message))
    console.log('Sent stop command:', message)
    return true
  }, [])

  /**
   * バイナリ音声データを送信
   */
  const sendAudioData = useCallback((audioData) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected')
      return false
    }

    // Int16ArrayをArrayBufferに変換
    const buffer = audioData.buffer

    // シーケンス番号とタイムスタンプを付与
    const header = new ArrayBuffer(8)
    const view = new DataView(header)
    view.setUint32(0, 0, true) // seq (暫定0)
    view.setUint32(4, Date.now() >>> 0, true) // timestamp

    // ヘッダーと音声データを結合
    const out = new Uint8Array(header.byteLength + buffer.byteLength)
    out.set(new Uint8Array(header), 0)
    out.set(new Uint8Array(buffer), header.byteLength)

    wsRef.current.send(out)
    return true
  }, [])

  /**
   * LLM分析リクエストを送信
   */
  const requestAnalysis = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected')
      return false
    }

    if (!sessionId) {
      console.error('No session ID')
      return false
    }

    const message = {
      type: 'analyze',
      sessionId: sessionId,
    }

    wsRef.current.send(JSON.stringify(message))
    console.log('Sent analyze command:', message)
    return true
  }, [sessionId])

  /**
   * メッセージ受信コールバックの設定
   */
  const onMessage = useCallback((callback) => {
    onMessageCallbackRef.current = callback
  }, [])

  /**
   * エラーコールバックの設定
   */
  const onError = useCallback((callback) => {
    onErrorCallbackRef.current = callback
  }, [])

  /**
   * クローズコールバックの設定
   */
  const onClose = useCallback((callback) => {
    onCloseCallbackRef.current = callback
  }, [])

  // コンポーネントのアンマウント時に切断
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  return {
    isConnected,
    sessionId,
    status,
    connect,
    disconnect,
    startRecording,
    stopRecording,
    sendAudioData,
    requestAnalysis,
    onMessage,
    onError,
    onClose,
  }
}
