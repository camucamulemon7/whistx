# 文字起こし誤り修正機能 実装計画

## 1. 背景と目的
- 現状はリアルタイム文字起こし結果の表示と、別パネルでの AI 校正（生成）まではあるが、ユーザーが「この箇所だけ誤りを直す」直接編集フローはない。
- 会議記録用途では、固有名詞・数字・業界用語の最終確認で手修正ニーズが高い。
- 目的は、**文字起こし結果をユーザーが安全に手修正し、保存履歴・ダウンロード成果物へ反映できる**こと。

## 2. スコープ

### 2.1 対象（今回）
1. 文字起こし表示領域でのインライン編集（行単位）
2. 編集内容のローカル状態管理（未保存/保存済み）
3. サーバーへの編集済み transcript 保存 API
4. 履歴保存（`/api/history/save`）に編集済み本文を反映
5. ダウンロード (`.txt/.jsonl/.zip`) に編集反映
6. 最低限の監査情報（誰がいつ編集保存したか）

### 2.2 非対象（次フェーズ）
- Google Docs 的な同時共同編集
- 変更履歴の全文差分 UI
- 編集内容を ASR モデル学習へ自動フィードバック

## 3. 現状整理（実装上の前提）
- フロントエンドは `web/src/app.js` で transcript 表示・状態を集中管理している。
- バックエンドは `server/legacy_app.py` が WebSocket 受信から transcript artifact 出力までを担っている。
- 履歴保存は `server/services/history_service.py` で `proofreadText` 等を扱う既存経路があるため、ここへ「編集済み transcript」を追加するのが自然。

## 4. 仕様案

### 4.1 UX 仕様
1. 各 transcript 行に「編集」アクションを追加
2. 編集開始時:
   - 行を `contenteditable` もしくは `<textarea>` に切替
   - `保存` / `キャンセル` を表示
3. 保存時:
   - 行単位で state を更新
   - 画面上部に「未保存の編集があります」バッジを表示
4. セッション保存（既存保存ボタン）時:
   - 編集済み transcript を payload に含める
5. 編集競合:
   - 録音中に同じ行へ ASR 追記が入った場合は、その行の編集をロックしてトースト通知

### 4.2 データ仕様
- 新規概念:
  - `editedTranscriptText`: 行編集結果を結合した最終本文
  - `transcriptEdited`: true/false
  - `transcriptEditedAt`: ISO8601 timestamp
- JSONL 反映方針:
  - v1 は本文（TXT）優先で編集反映
  - JSONL は既存 segment の `text` を上書きする簡易実装
  - 将来は `edited_text` 追加へ拡張

### 4.3 API 仕様

#### A. 履歴保存 API 拡張（推奨・影響小）
- `POST /api/history/save` の payload に下記を追加:
  - `editedTranscriptText?: string`
  - `transcriptEditedAt?: string`
- サーバー側:
  - 受信時に空白正規化・最大文字数チェック
  - 既存 transcript より `editedTranscriptText` を優先保存

#### B. 編集専用 API（将来）
- `POST /api/transcript/{session_id}/edit`
- リアルタイムセッション中の都度保存が必要になったら導入

## 5. 実装ステップ

### Phase 1: フロントエンド基盤
1. `state` に編集用フィールド追加
2. transcript 行レンダリングに編集 UI を追加
3. 編集イベント（開始/保存/取消）を実装
4. 未保存警告バッジと離脱警告（beforeunload）を実装

### Phase 2: バックエンド保存経路
1. `server/schemas.py` の保存 request schema を拡張
2. `server/services/history_service.py` の `save_history` 系で編集本文を受理
3. `server/services/artifact_storage.py` の `transcript.txt`, `transcript.jsonl`, `zip` 出力へ反映
4. 必要なら `server/models.py` の `TranscriptHistory` に `edited_transcript_text` カラム追加（マイグレーション）

### Phase 3: 表示・再読込
1. 履歴取得 API のレスポンスに編集本文を含める
2. フロントエンドで履歴復元時に編集済み本文を優先表示

### Phase 4: 品質保証
1. ユニットテスト
   - schema バリデーション
   - 保存時の優先順位（edited > raw）
2. API テスト
   - `/api/history/save` に編集本文を送って取得結果を検証
3. E2E 手動確認
   - 編集→保存→再読込→ZIP ダウンロード整合

## 6. リスクと対策
1. **JSONL 整合性崩れ**
   - 対策: v1 は segment 再分割をせず、既存順序で text のみ更新
2. **録音中編集との競合**
   - 対策: 完了チャンクのみ編集可能に限定
3. **保存サイズ増加**
   - 対策: 文字数上限・差分保存は将来検討

## 7. 受け入れ条件（Definition of Done）
- ユーザーが transcript 行を編集し保存できる
- 保存後の履歴再表示で編集内容が保持される
- `transcript.txt` と ZIP 内 `transcript.txt` に編集が反映される
- 既存の要約・校正機能が退行しない

## 8. 推奨実装順（1スプリント想定）
1. FE 編集 UI + state（2日）
2. 履歴保存 payload 拡張（1日）
3. artifact 反映（1日）
4. テスト/不具合修正（1日）

## 9. 将来拡張
- 用語辞書との連携（編集内容を glossary 候補化）
- AI による「修正提案」モード（確定は人間）
- 編集差分の監査ログ（before/after, editor, timestamp）
