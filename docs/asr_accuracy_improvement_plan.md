# whistx 精度改善統合プラン

## 目的

whistx の文字起こし精度を、単発のモデル変更ではなく、入力品質、チャンク制御、文脈付与、後処理、障害耐性、評価基盤の6層で継続改善できる状態にする。

本書は、既存実装の確認結果と [`docs/dynamic-snacking-dolphin.md`](/home/remon1129/ai/whistx/docs/dynamic-snacking-dolphin.md) の提案を統合し、優先順位と実装現実性を整理したもの。

## 主な評価指標

- 日本語 WER/CER
- 固有名詞再現率
- 無音誤認識率
- チャンク跨ぎの欠落率
- 重複字幕率
- ASR API 失敗時の欠落率
- 話者ラベル一致率

## 現状整理

現行実装には既に以下が入っている。

- サーバ側 overlap 付与: [`server/audio_pipeline.py`](/home/remon1129/ai/whistx/server/audio_pipeline.py)
- chunk 前処理と source 別 filter: [`server/audio_pipeline.py`](/home/remon1129/ai/whistx/server/audio_pipeline.py)
- 短期文脈プロンプトと重要語抽出: [`server/legacy_app.py`](/home/remon1129/ai/whistx/server/legacy_app.py)
- overlap 重複除去と反復ノイズ抑制: [`server/legacy_app.py`](/home/remon1129/ai/whistx/server/legacy_app.py)
- クライアント側 VAD ベース chunk 確定: [`web/src/app.js`](/home/remon1129/ai/whistx/web/src/app.js)
- 一部の silence hallucination 除去: [`server/openai_whisper.py`](/home/remon1129/ai/whistx/server/openai_whisper.py)
- セッション後の diarization: [`server/diarizer.py`](/home/remon1129/ai/whistx/server/diarizer.py)

一方で、精度面の主なボトルネックは以下。

- VAD が RMS 閾値ベース中心で、雑音や会議音声に弱い
- `CLIENT_VAD_DROP_ENABLED` が無効で、無音チャンクをまだ送っている: [`web/src/app.js`](/home/remon1129/ai/whistx/web/src/app.js)
- chunk 長と overlap が固定寄りで、発話密度や話者交代に追従しない
- context prompt が短期履歴中心で、用語辞書と優先順位制御がない
- ASR 呼び出しにリトライや失敗チャンク救済がない
- 評価用データセットと WER/CER 計測基盤がない
- 話者分離は後段パッチのみで、本文精度改善には未活用

## 方針

改善は次の順番で進める。

1. 計測できる状態を先に作る
2. 無音・境界・障害で失うチャンクを減らす
3. 固有名詞と文脈の再現率を上げる
4. 後処理と話者情報で仕上げる

理由:

- 評価基盤がないと改善と劣化を判定できない
- まず取りこぼしを減らす方が、後処理より効果が大きい
- 固有名詞改善はユーザー体感に直結する

## 優先度付き改善施策

### P0: 評価基盤を先に作る

精度改善は、まず計測できる状態にしないと最適化できない。

施策:

- 20から50件の実運用サンプルを用途別に収集する
- カテゴリを分ける
  - 1on1 会議
  - 複数人会議
  - 画面共有中心
  - マイクのみ
  - 専門用語多め
  - 雑音あり
- 正解文字起こしを人手で作成し、WER/CER を自動算出する
- `prompt`, `chunk 秒数`, `overlap`, `source mode`, `language`, `model`, `temperature`, `preprocess`, `diarization` を記録して比較できるようにする

実装候補:

- `tests/fixtures/asr_eval/` に音声と正解テキストを置く
- `scripts/eval_asr.py` を追加して JSON/CSV レポートを出す
- Langfuse に推論条件と品質指標を残す

期待効果:

- 改善施策の当たり外れを定量で判断できる
- 設定変更で劣化したケースをすぐ検出できる

### P1: 無音チャンクの早期除外

これは Claude 案の Quick Win で、現行実装に最も載せやすい。

施策:

- クライアント側 VAD ドロップを段階的に有効化する
- サーバ側でも PCM ベースの silence filter を追加して二重化する
- drop した理由をログと trace に残す

実装ポイント:

- `CLIENT_VAD_DROP_ENABLED = false` の扱いを設定可能にする: [`web/src/app.js`](/home/remon1129/ai/whistx/web/src/app.js)
- [`server/audio_pipeline.py`](/home/remon1129/ai/whistx/server/audio_pipeline.py) の `PreparedAudio` に `speech_ratio`, `rms`, `peak` を追加
- [`server/core/config/asr.py`](/home/remon1129/ai/whistx/server/core/config/asr.py) に以下を追加
  - `ASR_VAD_DROP_ENABLED`
  - `ASR_VAD_SPEECH_RATIO_MIN`

注意:

- クライアント側だけで drop すると取りすぎるリスクがあるため、最初はログのみ、その次に保守的閾値で有効化する

期待効果:

- 無音 hallucination の減少
- API コスト削減
- queue 詰まりの緩和

### P1: VAD と chunk 切断を固定閾値から適応型に改善

現状の VAD は軽量だが、早切りと切り遅れが残る。

施策:

- source ごとの固定閾値をやめ、録音開始後 1から2 秒のノイズ床を自動推定する
- `speech start`, `speech hold`, `speech end` のヒステリシスを入れる
- `display` と `both` は短い無音で切らず、`mic` より長めに保持する
- chunk 秒数を発話密度に応じて可変化する
- overlap も固定 3.5 秒ではなく発話量と境界状態で調整する

実装ポイント:

- [`web/src/app.js`](/home/remon1129/ai/whistx/web/src/app.js) の `sampleVad()`, `buildVadDecision()`, `shouldCutChunkOnSilence()` を拡張
- [`server/audio_pipeline.py`](/home/remon1129/ai/whistx/server/audio_pipeline.py) の `_resolve_overlap_ms()` を発話密度ベースへ拡張
- chunk メタに `speechRatio`, `activeMs` を送る

期待効果:

- 文中での早切り減少
- chunk 境界での語落ちと重複の減少
- 文境界の自然さ向上

### P1: ASR 呼び出しの障害耐性を強化

精度改善として見落とされやすいが、API 失敗でセグメント欠落すると結果として精度は大きく落ちる。

施策:

- ASR 呼び出しに指数バックオフ付きリトライを追加する
- 失敗チャンクを一定数バッファし、次チャンクと合わせて救済する
- リトライ不能な 4xx は即失敗とし、それ以外のみ再試行する

実装ポイント:

- [`server/openai_whisper.py`](/home/remon1129/ai/whistx/server/openai_whisper.py) に retry policy を追加
- [`server/core/config/asr.py`](/home/remon1129/ai/whistx/server/core/config/asr.py) に以下を追加
  - `ASR_RETRY_MAX_ATTEMPTS`
  - `ASR_RETRY_BASE_DELAY_MS`
- [`server/legacy_app.py`](/home/remon1129/ai/whistx/server/legacy_app.py) の `LiveSession` と `_session_worker()` に failed chunk buffer を追加

期待効果:

- 一時的な通信失敗時の欠落防止
- 部分的な API 障害時の継続性向上

### P1: 入力音声品質の制御を強化

[`web/src/app.js`](/home/remon1129/ai/whistx/web/src/app.js) の `getUserMedia` はまだデバイス差の影響を受けやすい。

施策:

- `channelCount: 1`, `sampleRate: 48000` を明示する
- `voiceIsolation` が使える環境では opt-in で有効化する
- マイク入力レベル過小時に UI 警告を出す
- クリッピング検知を追加する
- サーバ側 preprocess を音量統計つきで見直す

実装ポイント:

- [`web/src/app.js`](/home/remon1129/ai/whistx/web/src/app.js) の `requestMicStream()` を拡張
- [`server/audio_pipeline.py`](/home/remon1129/ai/whistx/server/audio_pipeline.py) に音量統計ログを追加
- `dynaudnorm` と `loudnorm` の A/B を取る

期待効果:

- 小音量、雑音混入、音割れ由来の誤認識減少

### P1: 用語辞書と prompt 構造化

現状の context term 抽出は直前テキストからの自動抽出中心で、固有名詞や略語の再現には弱い。

施策:

- セッションごとに用語辞書を持てるようにする
- UI の prompt を自由文と用語リストに分ける
- 履歴から頻出用語を学習し、次回候補として提示する
- 最近の用語ほど重みを上げる

実装ポイント:

- [`server/legacy_app.py`](/home/remon1129/ai/whistx/server/legacy_app.py) の `_build_prompt()` を `base_prompt`, `domain_terms`, `recent_context` の3層に分離
- [`server/legacy_app.py`](/home/remon1129/ai/whistx/server/legacy_app.py) の `_extract_context_terms()` を recency-weighted に改善
- [`server/core/config/asr.py`](/home/remon1129/ai/whistx/server/core/config/asr.py) の既定値を再調整する
  - `context_recent_lines`: 2 -> 4
  - `context_max_chars`: 1000 -> 2200
  - `context_term_limit`: 48 -> 80
- [`web/src/app.js`](/home/remon1129/ai/whistx/web/src/app.js) に「重要用語」入力欄を追加

期待効果:

- 人名、製品名、略語、英数字混在語の誤認識減少

### P1: silence hallucination 判定をスコアベースへ拡張

現状は一部の定型文を落としているが、汎化は弱い。

施策:

- `no_speech_prob`, セグメント長, 文字種分布, 繰り返し率でスコアリングする
- 破棄だけでなく `suspicious` として保存し、分析できるようにする
- 低信頼セグメントは再認識候補に回す

実装ポイント:

- [`server/openai_whisper.py`](/home/remon1129/ai/whistx/server/openai_whisper.py) で `avg_logprob`, `compression_ratio`, `no_speech_prob` を抽出する
- [`server/asr.py`](/home/remon1129/ai/whistx/server/asr.py) の `ASRChunkResult` に低信頼判定用の指標を追加する

期待効果:

- 無音時の誤字幕、締め文 hallucination、反復暴走の減少

### P2: 重複除去と境界処理の改善

Claude 案のファジーマッチ提案は有効だが、誤検出を防ぐ条件付けが必要。

施策:

- overlap prefix 除去にファジーマッチを導入する
- timestamp を併用して、直近セグメントとの近接時だけ閾値を下げる
- 日本語の表記揺れを少し吸収する

実装ポイント:

- [`server/legacy_app.py`](/home/remon1129/ai/whistx/server/legacy_app.py) の `_trim_overlap_prefix()` に SequenceMatcher を併用
- [`server/legacy_app.py`](/home/remon1129/ai/whistx/server/legacy_app.py) の `_is_near_duplicate()` に時間差条件を追加

期待効果:

- 境界重複のより正確な除去
- 誤った duplicate 判定の削減

### P2: 低信頼セグメントのマルチパス再認識

これは精度向上余地があるが、コスト増を伴うため P1 の後が妥当。

施策:

- `no_speech_prob > 0.4` かつテキストが短すぎる場合のみ再認識する
- 再認識時は temperature を少し上げる
- 元結果より改善が明確な場合のみ採用する

実装ポイント:

- [`server/openai_whisper.py`](/home/remon1129/ai/whistx/server/openai_whisper.py) に multi-pass ロジックを追加
- [`server/core/config/asr.py`](/home/remon1129/ai/whistx/server/core/config/asr.py) に `ASR_MULTI_PASS_ENABLED` を追加

期待効果:

- 小声、雑音環境、短い発話の取りこぼし回復

### P2: 後処理を language-aware に強化

現状の後処理は spacing と重複除去が中心。

施策:

- 数字、単位、日付、英字略語の正規化を追加する
- 日本語誤変換パターン辞書を導入する
- フィラー除去を option 化する
- LLM なしの軽量校正をリアルタイムパイプラインに入れる

実装ポイント:

- [`server/legacy_app.py`](/home/remon1129/ai/whistx/server/legacy_app.py) に `_light_proofread()` を追加
- 既存の `_sanitize_transcript_text()` の責務を分けて拡張する

注意:

- 過補正リスクがあるため、生 ASR テキストも保持する

期待効果:

- 読みやすさ向上
- 定型業務文脈での誤記削減

### P2: diarization を本文精度にも使う

今はセッション後に話者ラベルを貼るだけで、本文精度の改善には未活用。

施策:

- 話者交代点を chunk 切断候補に使う
- speaker turn をもとに短時間重複を再評価する
- 将来的に話者別 transcript lane を持ち、話者単位で用語補正を最適化する

期待効果:

- 被り発話時の混線緩和
- 発話者切替直後の欠落減少

### P3: モデル比較、WS 復旧、運用改善

これは安定運用と長期最適化向け。

施策:

- `ASR_MODEL` ごとの評価結果を蓄積する
- alternative backend との比較を可能にする
- WebSocket 再接続とセッション復旧を追加する

期待効果:

- 体感ではなく実測で設定を選べる
- 通信断時の transcript 欠落を抑制できる

## 実装優先順位

### Phase 1: 計測と Quick Win

- 評価スクリプト追加
- 評価用サンプル収集
- Langfuse 記録項目の整理
- クライアント VAD ドロップのログ化と段階的有効化
- サーバ側 silence filter
- ASR retry
- 音声品質メトリクス収集

### Phase 2: コア精度改善

- 動的 VAD 閾値
- source 別 silence cut 調整
- chunk/overlap 適応化
- 用語辞書 UI
- prompt 構造化
- context 既定値の見直し
- hallucination スコアリング

### Phase 3: 仕上げ

- ファジー重複除去
- 低信頼セグメント再認識
- 軽量校正
- diarization 連携
- WS 再接続

## まず着手すべき5件

費用対効果が高い順では次を推奨する。

1. 評価基盤の追加
2. サーバ側 silence filter と音声品質メトリクス
3. ASR retry
4. 動的 VAD と chunk 切断改善
5. 用語辞書と prompt 構造化

理由:

- 評価なしでは改善判断ができない
- silence 対策はハルシネーションとコストの両方に効く
- retry は欠落防止の即効性が高い
- VAD と chunk 境界は認識品質の主要因
- 固有名詞対策はユーザー体感改善が大きい

## 成果判定ライン

最初の1スプリントでは以下を目標にする。

- ベンチマークセット全体 CER を 10%以上改善
- 専門用語カテゴリの固有名詞再現率を 15%以上改善
- 無音 hallucination 件数を 50%以上削減
- ASR 一時障害時の欠落率を有意に削減
- chunk 跨ぎ欠落の手修正報告を有意に減らす

## 検証方法

1. silence filter:
   サイレント音声で録音し、`speech_ratio` と skip 理由がログに残ることを確認する
2. retry:
   ASR API を一時停止し、再試行後に復旧できることを確認する
3. context 改善:
   技術用語を含む会話で用語再現率を比較する
4. multi-pass:
   小声音声で `no_speech_prob` と再認識有無を確認する
5. 重複除去:
   chunk 境界をまたぐ発話で重複削減率を比較する
6. Langfuse:
   `speech_ratio`, `rms`, `avg_logprob`, `compression_ratio` が trace に残ることを確認する

## 補足

Claude 案の Quick Win は実装着手しやすく、そのまま取り込む価値がある。一方で、評価基盤を後回しにすると局所最適に寄りやすい。したがって、短期施策は取り込みつつも、必ず評価基盤を先頭に置くのが妥当。
