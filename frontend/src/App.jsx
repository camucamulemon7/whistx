import { useState, useEffect, useCallback } from 'react'
import { useAudioRecorder } from './hooks/useAudioRecorder'
import { useWebSocket } from './hooks/useWebSocket'

// ヘッダーコンポーネント
function Header({ status, children }) {
  return (
    <header className="sticky top-0 z-50 bg-white border-b shadow-sm">
      <div className="max-w-4xl mx-auto px-4 py-3 flex justify-between items-center">
        <h1 className="text-xl font-bold text-gray-900">Whistx v2</h1>
        <div className="flex items-center gap-4">
          {children}
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${status === '録音中' ? 'bg-red-500 animate-pulse' : 'bg-gray-300'}`} />
            <span className="text-sm text-gray-600">{status}</span>
          </div>
        </div>
      </div>
    </header>
  )
}

// オーディオソース選択コンポーネント
function AudioSourceSelector({ selectedSource, onSourceChange, disabled }) {
  return (
    <div className="flex items-center gap-2">
      <select
        value={selectedSource}
        onChange={(e) => onSourceChange(e.target.value)}
        disabled={disabled}
        className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg bg-white text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <option value="mic">🎤 マイク</option>
        <option value="system">🔊 システム音声 (Webex/Zoom)</option>
        <option value="both">🎤+🔊 マイク+システム音声</option>
      </select>
    </div>
  )
}

// トランスクリプトアイテムコンポーネント
function TranscriptItem({ transcript }) {
  const formatTime = (tsStart) => {
    const ts = Math.max(0, Math.round(tsStart / 1000))
    const mm = Math.floor(ts / 60).toString().padStart(2, '0')
    const ss = (ts % 60).toString().padStart(2, '0')
    return `${mm}:${ss}`
  }

  return (
    <div className="flex items-start gap-3 mb-2 group animate-in slide-in-from-bottom-2">
      <div className="flex-shrink-0 w-16 text-sm text-gray-500 font-mono">
        {formatTime(transcript.tsStart || 0)}
      </div>
      <div className="flex-1">
        {transcript.speaker && (
          <span className="inline-block px-2 py-0.5 text-xs font-semibold bg-blue-100 text-blue-700 rounded-full mb-1">
            {transcript.speaker}
          </span>
        )}
        <p className="text-gray-800 leading-relaxed">
          {transcript.text || ''}
        </p>
      </div>
    </div>
  )
}

// パーシャル（暫定）結果表示
function PartialResult({ text }) {
  if (!text) return null
  return (
    <div className="flex items-start gap-3 mb-2 opacity-70">
      <div className="flex-shrink-0 w-16 text-sm text-gray-400 font-mono">
        ...
      </div>
      <div className="flex-1">
        <p className="text-gray-600 italic">{text}</p>
      </div>
    </div>
  )
}

// トランスクリプトリストコンポーネント
function TranscriptList({ transcripts, partialText }) {
  return (
    <div className="space-y-2">
      {transcripts.length === 0 && !partialText ? (
        <div className="text-center text-gray-400 py-8">
          <p>録音を開始すると、ここに文字起こしが表示されます</p>
        </div>
      ) : (
        <>
          {transcripts.map((t, index) => (
            <TranscriptItem key={t.id || index} transcript={t} />
          ))}
          <PartialResult text={partialText} />
        </>
      )}
    </div>
  )
}

// オーディオビジュアライザー
function AudioVisualizer({ level, isRecording }) {
  if (!isRecording) return null

  return (
    <div className="fixed bottom-24 left-1/2 -translate-x-1/2">
      <div className="flex items-end gap-1 h-8">
        {[...Array(32)].map((_, i) => (
          <div
            key={i}
            className="w-1 bg-blue-500 rounded-full transition-all duration-75"
            style={{
              height: `${Math.max(4, Math.min(32, level * 32 * (1 + Math.sin(i * 0.5))))}px`,
              opacity: 0.6 + (level * 0.4),
            }}
          />
        ))}
      </div>
    </div>
  )
}

// フッターコンポーネント
function Footer({ isRecording, wsConnected, onToggle, onAnalyze, transcriptCount }) {
  return (
    <footer className="fixed bottom-0 left-0 right-0 bg-white border-t shadow-lg">
      <div className="max-w-4xl mx-auto px-4 py-4 flex flex-col items-center gap-3">
        {/* WebSocket接続状態 */}
        <div className="flex items-center gap-2 text-xs">
          <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-gray-300'}`} />
          <span className="text-gray-500">
            {wsConnected ? 'サーバーに接続中' : 'サーバー未接続'}
          </span>
          {transcriptCount > 0 && (
            <span className="ml-2 text-gray-400">
              ({transcriptCount} 件のトランスクリプト)
            </span>
          )}
        </div>

        {/* ボタン群 */}
        <div className="flex items-center gap-3">
          {/* LLM分析ボタン */}
          {transcriptCount > 0 && (
            <button
              onClick={onAnalyze}
              disabled={!wsConnected}
              className="px-6 py-3 rounded-full font-semibold text-purple-600 bg-purple-50 hover:bg-purple-100 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              🤖 分析
            </button>
          )}

          {/* 録音ボタン */}
          <button
            onClick={onToggle}
            disabled={!wsConnected}
            className={`px-8 py-3 rounded-full font-semibold text-white transition-all ${
              isRecording
                ? 'bg-red-500 hover:bg-red-600'
                : 'bg-blue-600 hover:bg-blue-700'
            } ${!wsConnected ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {isRecording ? '停止' : '録音開始'}
          </button>
        </div>
      </div>
    </footer>
  )
}

// 分析結果モーダルコンポーネント
function AnalysisModal({ analysis, onClose }) {
  if (!analysis) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-hidden">
        {/* ヘッダー */}
        <div className="sticky top-0 bg-white border-b px-6 py-4 flex justify-between items-center">
          <h2 className="text-xl font-bold text-gray-900">会議分析結果</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl leading-none"
          >
            ×
          </button>
        </div>

        {/* コンテンツ */}
        <div className="p-6 overflow-y-auto max-h-[calc(80vh-80px)] space-y-6">
          {/* 要約 */}
          {analysis.summary && (
            <section>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">要約</h3>
              <p className="text-gray-700 whitespace-pre-wrap">{analysis.summary}</p>
            </section>
          )}

          {/* タスク */}
          {analysis.tasks && analysis.tasks.length > 0 && (
            <section>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">タスク一覧</h3>
              <ul className="space-y-2">
                {analysis.tasks.map((task, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-blue-500 font-bold">•</span>
                    <span className="text-gray-700">{task.text}</span>
                    {task.assignee && (
                      <span className="ml-auto px-2 py-0.5 text-xs bg-blue-100 text-blue-700 rounded-full">
                        {task.assignee}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* 決定事項 */}
          {analysis.decisions && analysis.decisions.length > 0 && (
            <section>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">決定事項</h3>
              <ul className="space-y-2">
                {analysis.decisions.map((decision, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-green-500">✓</span>
                    <span className="text-gray-700">{decision}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* 次回のアクション */}
          {analysis.next_actions && analysis.next_actions.length > 0 && (
            <section>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">次回のアクション</h3>
              <ul className="space-y-2">
                {analysis.next_actions.map((action, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-orange-500">→</span>
                    <span className="text-gray-700">{action}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* メタデータ（LLM未使用時など） */}
          {analysis._meta && (
            <section className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-500">
                トランスクリプト数: {analysis._meta.total_transcripts} |
                発言者数: {analysis._meta.total_speakers} |
                総語数: {analysis._meta.total_words}
              </p>
            </section>
          )}
        </div>
      </div>
    </div>
  )
}

// メインアプリコンポーネント
function App() {
  const [transcripts, setTranscripts] = useState([])
  const [partialText, setPartialText] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [analysis, setAnalysis] = useState(null)
  const [showAnalysis, setShowAnalysis] = useState(false)

  // WebSocketフック
  const {
    isConnected: wsConnected,
    sessionId,
    status: wsStatus,
    connect: wsConnect,
    disconnect: wsDisconnect,
    startRecording: wsStartRecording,
    stopRecording: wsStopRecording,
    sendAudioData,
    requestAnalysis,
    onMessage,
  } = useWebSocket()

  // オーディオ録音フック
  const {
    audioLevel,
    startRecording: startAudioCapture,
    stopRecording: stopAudioCapture,
    selectedSource,
    setSelectedSource,
  } = useAudioRecorder()

  // セッションIDの状態管理
  useEffect(() => {
    if (sessionId) {
      console.log('Session ID:', sessionId)
    }
  }, [sessionId])

  // WebSocketメッセージハンドラー
  useEffect(() => {
    onMessage((message) => {
      console.log('Received message:', message)

      if (message.type === 'partial') {
        // パーシャル（暫定）結果
        setPartialText(message.text || '')
      } else if (message.type === 'final') {
        // 確定結果
        setPartialText('')
        setTranscripts((prev) => [
          ...prev,
          {
            id: message.segmentId || `${Date.now()}`,
            text: message.text || '',
            tsStart: message.tsStart || 0,
            tsEnd: message.tsEnd || 0,
            speaker: message.speaker || null,
          },
        ])
      } else if (message.type === 'analysis') {
        // LLM分析結果
        console.log('Analysis result:', message.result)
        setAnalysis(message.result)
        setShowAnalysis(true)
      } else if (message.type === 'info') {
        console.log('Info:', message)
      } else if (message.type === 'error') {
        console.error('Server error:', message.message)
      }
    })
  }, [onMessage])

  // LLM分析リクエスト
  const handleAnalyze = useCallback(() => {
    if (!wsConnected) {
      alert('サーバーに接続されていません')
      return
    }
    if (requestAnalysis()) {
      console.log('Analysis requested')
    } else {
      alert('分析リクエストに失敗しました')
    }
  }, [wsConnected, requestAnalysis])

  // オーディオデータハンドラー
  const handleAudioData = useCallback(({ audio, timestamp }) => {
    // WebSocketで音声データを送信
    if (!sendAudioData(audio)) {
      console.error('Failed to send audio data')
    }
  }, [sendAudioData])

  // 録音開始/停止ハンドラー
  const handleToggleRecording = async () => {
    if (isRecording) {
      // 停止処理
      stopAudioCapture()
      await wsStopRecording()
      setIsRecording(false)
      setPartialText('')
    } else {
      // 開始処理
      try {
        // WebSocket接続
        if (!wsConnected) {
          await wsConnect()
          // 接続待機
          await new Promise(resolve => setTimeout(resolve, 500))
        }

        // 録音開始コマンド送信
        if (wsStartRecording()) {
          // オーディオキャプチャ開始
          await startAudioCapture(handleAudioData, {
            gain: 1.0,
            audioSource: selectedSource,
          })
          setIsRecording(true)
        }
      } catch (error) {
        console.error('Failed to start recording:', error)
        alert('録音の開始に失敗しました: ' + error.message)
      }
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header status={isRecording ? '録音中' : '待機中'}>
        <AudioSourceSelector
          selectedSource={selectedSource}
          onSourceChange={setSelectedSource}
          disabled={isRecording}
        />
      </Header>
      <main className="max-w-4xl mx-auto px-4 py-8 pb-32">
        <TranscriptList
          transcripts={transcripts}
          partialText={partialText}
        />
      </main>
      <AudioVisualizer level={audioLevel} isRecording={isRecording} />
      <Footer
        isRecording={isRecording}
        wsConnected={wsConnected}
        onToggle={handleToggleRecording}
        onAnalyze={handleAnalyze}
        transcriptCount={transcripts.length}
      />
      {showAnalysis && (
        <AnalysisModal
          analysis={analysis}
          onClose={() => setShowAnalysis(false)}
        />
      )}
    </div>
  )
}

export default App
