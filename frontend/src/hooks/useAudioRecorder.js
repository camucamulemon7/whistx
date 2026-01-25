// frontend/src/hooks/useAudioRecorder.js
import { useRef, useCallback, useState } from 'react'

/**
 * オーディオ録音カスタムフック
 * マイク、システムオーディオ、仮想オーディオデバイスに対応
 */
export function useAudioRecorder() {
  const audioContextRef = useRef(null)
  const workletNodeRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const [isRecording, setIsRecording] = useState(false)
  const [audioLevel, setAudioLevel] = useState(0)
  const onAudioCallbackRef = useRef(null)
  const [selectedSource, setSelectedSource] = useState('mic') // 'mic' | 'system' | 'device'

  /**
   * 利用可能なオーディオデバイスを列挙
   */
  const getAudioDevices = useCallback(async () => {
    try {
      // デバイスの列挙許可を要求
      await navigator.mediaDevices.getUserMedia({ audio: true })
      const devices = await navigator.mediaDevices.enumerateDevices()

      return devices
        .filter(device => device.kind === 'audioinput')
        .map(device => ({
          deviceId: device.deviceId,
          label: device.label || `Microphone ${device.deviceId.slice(0, 5)}`,
        }))
    } catch (error) {
      console.error('Failed to enumerate audio devices:', error)
      return []
    }
  }, [])

  /**
   * システムオーディオを取得（画面共有経由）
   */
  const getSystemAudio = useCallback(async () => {
    try {
      // getDisplayMediaでシステムオーディオを取得
      // ブラウザのポップアップで「タブのオーディオ」または「システムオーディオ」を選択
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,  // videoは必須（ユーザーはキャプチャしないタブを選択可能）
        audio: true,  // システムオーディオを要求
      })

      // videoトラックは使用しないので削除
      const videoTrack = stream.getVideoTracks()[0]
      if (videoTrack) {
        videoTrack.stop()
        stream.removeTrack(videoTrack)
      }

      return stream
    } catch (error) {
      console.error('Failed to get system audio:', error)
      throw error
    }
  }, [])

  /**
   * 録音開始
   */
  const startRecording = useCallback(async (onAudioData, options = {}) => {
    try {
      let stream = null
      const source = options.audioSource || selectedSource

      if (source === 'both') {
        // マイク + システムオーディオの同時録音
        const micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: options.echoCancellation ?? true,
            noiseSuppression: options.noiseSuppression ?? true,
            autoGainControl: options.autoGainControl ?? true,
            sampleRate: 48000,
          },
        })
        const systemStream = await getSystemAudio()

        // 両方の audio track を統合して1つのストリームに
        const combinedTracks = [
          ...micStream.getAudioTracks(),
          ...systemStream.getAudioTracks(),
        ]
        stream = new MediaStream(combinedTracks)

        // 両方の元ストリームを保持（停止用）
        mediaStreamRef.current = { micStream, systemStream, combinedStream: stream }
      } else if (source === 'system') {
        // システムオーディオ（Webex、Zoom等）
        stream = await getSystemAudio()
        mediaStreamRef.current = stream
      } else if (options.deviceId) {
        // 特定のオーディオデバイス（仮想デバイス等）
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            deviceId: { exact: options.deviceId },
            echoCancellation: options.echoCancellation ?? true,
            noiseSuppression: options.noiseSuppression ?? true,
            autoGainControl: options.autoGainControl ?? true,
            sampleRate: 48000,
          },
        })
        mediaStreamRef.current = stream
      } else {
        // デフォルトマイク
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: options.echoCancellation ?? true,
            noiseSuppression: options.noiseSuppression ?? true,
            autoGainControl: options.autoGainControl ?? true,
            sampleRate: 48000, // 入力サンプリングレート
          },
        })
        mediaStreamRef.current = stream
      }

      // AudioContextの作成
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 48000,
      })
      audioContextRef.current = audioContext

      // AudioWorkletのロード
      await audioContext.audioWorklet.addModule('/audio-worklet-processor.js')

      // MediaStreamSourceの作成
      const mediaSource = audioContext.createMediaStreamSource(stream)

      // WorkletNodeの作成
      const workletNode = new AudioWorkletNode(audioContext, 'pcm16-downsampler')
      workletNodeRef.current = workletNode

      // ゲインの設定
      if (options.gain !== undefined) {
        workletNode.port.postMessage({ type: 'set', gain: options.gain })
      }

      // オーディオレベル計測用のAnalyserNode
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 256
      mediaSource.connect(analyser)

      // ダグラムの接続: Source -> Worklet -> Destination (mute)
      const destination = audioContext.createGain()
      destination.gain.value = 0 // ミュート（スピーカーから出さない）
      analyser.connect(workletNode).connect(destination).connect(audioContext.destination)

      // オーディオフレームを受信
      workletNode.port.onmessage = (e) => {
        const { type, ptsMs, payload } = e.data
        if (type === 'frame' && payload) {
          // Int16Arrayに変換
          const audioData = new Int16Array(payload)
          if (onAudioData) {
            onAudioData({
              audio: audioData,
              timestamp: ptsMs,
            })
          }
        }
      }

      // オーディオレベルの定期的な更新
      const dataArray = new Uint8Array(analyser.frequencyBinCount)
      const updateLevel = () => {
        if (!isRecording) return
        analyser.getByteFrequencyData(dataArray)
        // 平均レベルを計算（0-1）
        const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length
        setAudioLevel(average / 255)
        requestAnimationFrame(updateLevel)
      }
      updateLevel()

      setIsRecording(true)
      onAudioCallbackRef.current = onAudioData

      console.log('Recording started')
      return true

    } catch (error) {
      console.error('Failed to start recording:', error)
      setIsRecording(false)
      throw error
    }
  }, [])

  /**
   * 録音停止
   */
  const stopRecording = useCallback(() => {
    // WorkletNodeの切断
    if (workletNodeRef.current) {
      try {
        workletNodeRef.current.disconnect()
      } catch (e) {
        console.error('Failed to disconnect worklet node:', e)
      }
      workletNodeRef.current = null
    }

    // AudioContextの閉じる
    if (audioContextRef.current) {
      try {
        audioContextRef.current.close()
      } catch (e) {
        console.error('Failed to close audio context:', e)
      }
      audioContextRef.current = null
    }

    // MediaStreamの停止
    if (mediaStreamRef.current) {
      // 'both'モードの場合は複数ストリームを保持してる
      if (mediaStreamRef.current.micStream && mediaStreamRef.current.systemStream) {
        mediaStreamRef.current.micStream.getTracks().forEach((track) => track.stop())
        mediaStreamRef.current.systemStream.getTracks().forEach((track) => track.stop())
      } else {
        // 単一ストリームの場合
        mediaStreamRef.current.getTracks().forEach((track) => {
          track.stop()
        })
      }
      mediaStreamRef.current = null
    }

    setIsRecording(false)
    setAudioLevel(0)
    onAudioCallbackRef.current = null

    console.log('Recording stopped')
  }, [])

  /**
   * ゲインの設定
   */
  const setGain = useCallback((gain) => {
    if (workletNodeRef.current) {
      workletNodeRef.current.port.postMessage({ type: 'set', gain })
    }
  }, [])

  return {
    isRecording,
    audioLevel,
    startRecording,
    stopRecording,
    setGain,
    getAudioDevices,
    getSystemAudio,
    selectedSource,
    setSelectedSource,
  }
}
