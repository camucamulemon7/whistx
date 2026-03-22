# whistx 精度改善 実装計画書

## 目的

[`docs/asr_accuracy_improvement_plan.md`](/home/remon1129/ai/whistx/docs/asr_accuracy_improvement_plan.md) を、実装可能なチケット単位まで分解した計画書。

対象は主に P0 と P1。P2 以降は先行タスク完了後に着手する。

## 進め方

- まず計測基盤を作る
- 次に取りこぼしと無音チャンクを減らす
- その後に文脈と用語辞書へ進む
- 各チケットは小さく閉じ、設定追加と検証を同じチケット内で完結させる

## マイルストーン

### M1: 計測基盤

- ✅ T01 評価データセット雛形
- ✅ T02 評価スクリプト
- T03 Langfuse/ログ計測項目の整理

### M2: Quick Win

- T04 クライアント VAD ドロップの段階的導入
- ✅ T05 サーバ側 silence filter と音声品質メトリクス
- ✅ T06 ASR retry
- ✅ T07 失敗チャンク救済

### M3: コア精度改善

- ✅ T08 動的 VAD 閾値
- ✅ T09 chunk/overlap 適応化
- ✅ T10 用語辞書 UI
- T11 prompt 構造化
- T12 hallucination スコアリング

### M4: 後続改善

- T13 ファジー重複除去
- T14 multi-pass 再認識
- T15 軽量校正
- T16 diarization 連携

## チケット一覧

### ✅ T01: 評価データセット雛形を追加

目的:

- 精度比較のための入力データと正解データの置き場所を定義する

作業内容:

- `tests/fixtures/asr_eval/README.md` を追加
- カテゴリ別ディレクトリを用意する
  - `one_on_one/`
  - `multi_speaker/`
  - `screen_share/`
  - `mic_only/`
  - `domain_terms/`
  - `noisy/`
- 音声ファイルと正解テキストの命名規約を定義する
- メタデータ JSON の形式を定義する

対象ファイル:

- 新規 [tests/fixtures/asr_eval/README.md](/home/remon1129/ai/whistx/tests/fixtures/asr_eval/README.md)
- 新規 `tests/fixtures/asr_eval/**`

完了条件:

- サンプル追加ルールが文書化されている
- 1件以上のダミーサンプルで構造確認できる

依存:

- なし

### ✅ T02: WER/CER 評価スクリプトを追加

目的:

- 改善施策の前後比較を定量化する

作業内容:

- `scripts/eval_asr.py` を追加
- 入力として音声、正解テキスト、設定メタデータを読めるようにする
- 出力として JSON と CSV を生成する
- 最低限以下を出す
  - WER
  - CER
  - 文字数
  - 語彙再現率
  - 実行設定

対象ファイル:

- 新規 [scripts/eval_asr.py](/home/remon1129/ai/whistx/scripts/eval_asr.py)
- 必要に応じて [README.md](/home/remon1129/ai/whistx/README.md)

完了条件:

- fixture 1件以上で実行できる
- JSON/CSV が出力される

依存:

- T01

### T03: Langfuse とログの計測項目を整理

目的:

- 評価基盤と実運用ログを繋げる

作業内容:

- 現行 trace に追加する項目を決める
  - `speech_ratio`
  - `rms`
  - `peak`
  - `avg_logprob`
  - `compression_ratio`
  - `retry_count`
  - `skip_reason`
- ASR セッションとチャンク単位のどちらに出すか整理する
- 開発用ログの形式を統一する

対象ファイル:

- [server/legacy_app.py](/home/remon1129/ai/whistx/server/legacy_app.py)
- [server/openai_whisper.py](/home/remon1129/ai/whistx/server/openai_whisper.py)

完了条件:

- 計測項目がコードコメントまたは doc に整理されている
- trace 出力点が実装されているか、少なくとも着地点が明確

依存:

- T02 推奨

### T04: クライアント VAD ドロップを段階的に導入

目的:

- 明らかな無音チャンクの送信を減らす

作業内容:

- `CLIENT_VAD_DROP_ENABLED` を直値ではなく設定化する
- 初回は drop せずログだけ残すモードを追加する
- 次段階で保守的閾値により drop 可能にする
- UI テレメトリに drop 判定を反映する

対象ファイル:

- [web/src/app.js](/home/remon1129/ai/whistx/web/src/app.js)

完了条件:

- ログのみモードと実 drop モードを切り替えられる
- 短い無音チャンクで誤送信削減を確認できる

依存:

- T03 推奨

### ✅ T05: サーバ側 silence filter と音声品質メトリクスを追加

目的:

- クライアント側判定に依存せず、サーバ側でも無音を弾く

作業内容:

- `PreparedAudio` に `speech_ratio`, `rms`, `peak`, `audio_metrics` を追加
- PCM から簡易音声比率を算出する
- silence skip 用 config を追加する
  - `ASR_VAD_DROP_ENABLED`
  - `ASR_VAD_SPEECH_RATIO_MIN`
- `_session_worker()` で skip と metrics をログ出力する

対象ファイル:

- [server/audio_pipeline.py](/home/remon1129/ai/whistx/server/audio_pipeline.py)
- [server/core/config/asr.py](/home/remon1129/ai/whistx/server/core/config/asr.py)
- [server/legacy_app.py](/home/remon1129/ai/whistx/server/legacy_app.py)

完了条件:

- 低音量/無音チャンクが ASR 呼び出し前に弾かれる
- メトリクスがログまたは trace に残る

依存:

- T03 推奨

### ✅ T06: ASR retry を追加

目的:

- 一時障害による欠落を減らす

作業内容:

- retry policy を追加する
- 対象例外を限定する
  - connection error
  - timeout
  - rate limit
- 4xx は原則 retry しない
- config を追加する
  - `ASR_RETRY_MAX_ATTEMPTS`
  - `ASR_RETRY_BASE_DELAY_MS`

対象ファイル:

- [server/openai_whisper.py](/home/remon1129/ai/whistx/server/openai_whisper.py)
- [server/core/config/asr.py](/home/remon1129/ai/whistx/server/core/config/asr.py)

完了条件:

- 一時障害時に再試行される
- retry 回数がログまたは trace に残る

依存:

- なし

### ✅ T07: 失敗チャンク救済バッファを追加

目的:

- retry 全失敗時でも音声欠落を最小化する

作業内容:

- `LiveSession` に failed chunk buffer を追加
- 失敗した chunk を最大2件保持する
- 次 chunk 処理前に merge して再送する
- サイズ上限と破棄条件を明示する

対象ファイル:

- [server/legacy_app.py](/home/remon1129/ai/whistx/server/legacy_app.py)

完了条件:

- 失敗後の次チャンクで回復できるケースが確認できる
- バッファが無限成長しない

依存:

- T06

### ✅ T08: 動的 VAD 閾値を導入

目的:

- 雑音環境での誤切断を減らす

作業内容:

- 録音開始直後にノイズ床を推定する
- source mode 別に保持時間を調整する
- `speech start`, `hold`, `end` のヒステリシスを入れる

対象ファイル:

- [web/src/app.js](/home/remon1129/ai/whistx/web/src/app.js)

完了条件:

- `mic`, `display`, `both` で挙動差を確認できる
- 固定閾値より誤切断が減る

依存:

- T04

### ✅ T09: chunk/overlap を適応化

目的:

- 境界での欠落と重複を減らす

作業内容:

- client から `speechRatio`, `activeMs` を送る
- server 側 overlap を発話密度で調整する
- 文途中らしい場合は切断を遅らせる条件を追加する

対象ファイル:

- [web/src/app.js](/home/remon1129/ai/whistx/web/src/app.js)
- [server/audio_pipeline.py](/home/remon1129/ai/whistx/server/audio_pipeline.py)
- [server/legacy_app.py](/home/remon1129/ai/whistx/server/legacy_app.py)

完了条件:

- overlap が固定値以外でも変動する
- chunk 境界の欠落/重複が評価セットで改善する

依存:

- T05
- T08

### ✅ T10: 用語辞書 UI を追加

目的:

- 固有名詞と略語の認識率を上げる

作業内容:

- UI に「重要用語」入力欄を追加
- localStorage に保存する
- WebSocket `start` payload に vocabulary を追加する

対象ファイル:

- [web/index.html](/home/remon1129/ai/whistx/web/index.html)
- [web/src/app.js](/home/remon1129/ai/whistx/web/src/app.js)

完了条件:

- 用語入力がセッション開始時に送られる
- 画面再読込後も保持される

依存:

- なし

### T11: prompt 構造化と context 拡張

目的:

- 用語辞書と直前文脈を整理して ASR に渡す

作業内容:

- `_build_prompt()` を `base_prompt`, `domain_terms`, `recent_context` に分離する
- `_extract_context_terms()` を recency-weighted に改善する
- context の既定値を見直す
  - `context_recent_lines`
  - `context_max_chars`
  - `context_term_limit`

対象ファイル:

- [server/legacy_app.py](/home/remon1129/ai/whistx/server/legacy_app.py)
- [server/core/config/asr.py](/home/remon1129/ai/whistx/server/core/config/asr.py)

完了条件:

- prompt の構造がコード上で明確に分離されている
- 用語辞書と recent context が両方反映される

依存:

- T10 推奨

### T12: hallucination スコアリングを追加

目的:

- 無音や低信頼チャンクの誤字幕を減らす

作業内容:

- `ASRChunkResult` に信頼度指標を追加する
- `avg_logprob`, `compression_ratio`, `no_speech_prob` を抽出する
- suspicious 判定を追加する
- suspicious な結果を drop または保留できるようにする

対象ファイル:

- [server/asr.py](/home/remon1129/ai/whistx/server/asr.py)
- [server/openai_whisper.py](/home/remon1129/ai/whistx/server/openai_whisper.py)
- [server/legacy_app.py](/home/remon1129/ai/whistx/server/legacy_app.py)

完了条件:

- 低信頼チャンクにフラグが立つ
- 定型 hallucination 以外も捕捉できる

依存:

- T03

### T13: ファジー重複除去を追加

目的:

- chunk 境界の重複をより自然に削る

作業内容:

- `_trim_overlap_prefix()` に fuzzy match を導入
- `_is_near_duplicate()` に時間差条件を加える
- 日本語表記揺れを少し吸収する

対象ファイル:

- [server/legacy_app.py](/home/remon1129/ai/whistx/server/legacy_app.py)

完了条件:

- 直近境界の重複削減率が改善する
- 過剰削除が増えない

依存:

- T09

### T14: multi-pass 再認識を追加

目的:

- 低信頼セグメントの回復率を上げる

作業内容:

- 条件付き再認識ロジックを追加
- config `ASR_MULTI_PASS_ENABLED` を追加
- 元結果より改善した場合のみ採用

対象ファイル:

- [server/openai_whisper.py](/home/remon1129/ai/whistx/server/openai_whisper.py)
- [server/core/config/asr.py](/home/remon1129/ai/whistx/server/core/config/asr.py)

完了条件:

- 特定条件下でのみ再認識が動く
- コスト増が制御されている

依存:

- T12

### T15: 軽量校正を追加

目的:

- リアルタイム表示テキストの読みやすさを上げる

作業内容:

- `_light_proofread()` を追加
- 数字、フィラー、引用符などの軽量正規化を行う
- 生 ASR テキストと表示テキストの扱いを整理する

対象ファイル:

- [server/legacy_app.py](/home/remon1129/ai/whistx/server/legacy_app.py)

完了条件:

- 表示品質が改善する
- 過補正時の切り戻しが容易

依存:

- T12 推奨

### T16: diarization 連携改善

目的:

- 話者情報を本文精度改善にも活かす

作業内容:

- 話者交代点を chunk 切断候補に利用する設計を追加
- speaker turn と重複検出の連携余地を実装する

対象ファイル:

- [server/diarizer.py](/home/remon1129/ai/whistx/server/diarizer.py)
- [server/legacy_app.py](/home/remon1129/ai/whistx/server/legacy_app.py)

完了条件:

- diarization 情報を後段ラベル付け以外でも参照できる

依存:

- T09
- T13

## 推奨実装順

1. T01
2. T02
3. T03
4. T05
5. T06
6. T07
7. T04
8. T08
9. T09
10. T10
11. T11
12. T12
13. T13
14. T14
15. T15
16. T16

## スプリント分割案

### Sprint 1

- T01
- T02
- T03
- T05
- T06

狙い:

- 計測できる状態を作り、無音と障害による欠落を減らす

### Sprint 2

- T07
- T04
- T08
- T09

狙い:

- チャンク品質と境界品質を改善する

### Sprint 3

- T10
- T11
- T12

狙い:

- 固有名詞と文脈の再現率を上げる

### Sprint 4

- T13
- T14
- T15
- T16

狙い:

- 境界処理と後処理を仕上げる

## 各スプリントの成果判定

### Sprint 1 完了条件

- 評価スクリプトが回る
- silence skip が動く
- retry が動く

現状:

- ✅ T01 完了
- ✅ T02 完了
- T03 は一部着手。`speech_ratio`, `rms`, `peak`, `retryCount` はログまたは trace 出力に反映済みだが、計測項目整理としては未完
- ✅ T05 完了
- ✅ T06 完了

### Sprint 2 完了条件

- chunk 境界の欠落率または重複率が改善する

### Sprint 3 完了条件

- 固有名詞再現率が改善する

### Sprint 4 完了条件

- 表示品質と境界品質がさらに改善する

## リスク

- VAD drop は誤検出すると取りこぼしが増える
- multi-pass は API コストを増やす
- 軽量校正は過補正のリスクがある
- diarization 連携は構造変更が広がりやすい

## 補足

最初に着手すべき実装は、文脈改善ではなく `計測`, `silence filter`, `retry`。ここが先に入っていないと、後続施策の効果測定ができず、欠落も残ったままになる。
