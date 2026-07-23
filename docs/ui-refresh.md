# UI/UX refresh

Issue #26 の設計判断と回帰確認をまとめる。録音設定を入口、文字起こしを主作業面、AI処理と保存を出口として、画面全体を3段階のジャーニーに整理した。

## 主要ユーザーフロー

### 録音前

1. ログインまたは許可されている環境でゲスト利用を選ぶ。
2. 「入力して録音」で言語、音声ソース、話者分離、入力補正を確認する。
3. 必要な場合だけ「話者設定」または「詳細設定」を開く。
4. 「録音開始」を押し、ブラウザの音声権限を許可する。

### 録音中

1. ジャーニー表示が「内容を確認」に移り、ヘッダー状態が「録音中」になる。
2. 入力レベルと処理テレメトリを確認する。
3. 文字起こし本文が主パネルへ逐次追加される。
4. 「停止」を押し、残りのチャンク確定を待つ。

### 停止後

1. ジャーニー表示が「活用して保存」に移る。
2. 本文を確認し、必要に応じて校正または要約を生成する。
3. ログイン利用ではタイトルを付けて保存する。
4. TXT、JSONL、ZIPのいずれかで書き出す。

### 履歴閲覧

1. デスクトップでは左レール、狭い画面ではドロワーから履歴を検索する。
2. 履歴を選ぶと本文・校正・要約領域へ内容を復元する。
3. 再編集、削除、書き出しは現在のパネル内で完結する。

## ワイヤーフレーム

### デスクトップ（1440px以上）

```text
┌ Brand ─────────────── Status ───────── Help / Settings / Account ┐
├ ① 入力して録音 ─────── ② 内容を確認 ─────── ③ 活用して保存 ┤
├ History ┬ Record + input settings ──────────────────────────────┤
│ search  ├──────────────────────────────┬─────────────────────────┤
│ items   │                              │ Proofread               │
│         │ Transcript (primary)         ├─────────────────────────┤
│         │                              │ Summary                 │
│         ├──────────────────────────────┴─────────────────────────┤
│         │ Save / Export                                           │
└─────────┴─────────────────────────────────────────────────────────┘
```

### モバイル（640px以下）

```text
┌ Brand / Status                    ┐
├ Help / Settings / Account         ┤
├ swipe: ① → ② → ③                ┤
├ Record                            ┤
├ Input settings                    ┤
├ Transcript                        ┤
├ tabs: Proofread / Summary         ┤
├ Save / Export                     ┤
└ History is an off-canvas drawer   ┘
```

## コンポーネントとトークン

| 区分 | トークン / コンポーネント | 用途 |
|---|---|---|
| Surface | `--surface-canvas`, `--surface-panel`, `--surface-muted` | 背景、カード、設定項目の階層 |
| Brand | `--brand-strong`, `--brand-bright` | 主要操作、現在ステップ |
| Accessibility | `--focus-ring` | 全操作要素の3pxフォーカス |
| Elevation | `--panel-border`, `--panel-shadow` | ライト・ダーク共通の境界 |
| Journey | `.journey-step` | 録音前・録音中・停止後の現在位置 |
| Primary action | `.record-btn` | 通常はティール、録音中は赤 |
| Work surface | `.transcript-panel` | デスクトップ幅の約65%を確保 |
| Secondary result | `.proofread-panel`, `.summary-panel` | デスクトップで縦積み、狭い画面でタブ切替 |
| Navigation | `.skip-link` | キーボードで主領域へ直接移動 |

色だけに依存せず、現在位置は `aria-current`、録音状態はテキストと `aria-pressed`、処理中は `aria-busy` でも伝える。本文ログはフォーカス可能にし、追加内容は `role="log"` と `aria-live` で通知する。

## 実装・回帰チェックリスト

- [x] 3段階ジャーニーと実行状態の同期
- [x] 文字起こしを主、校正・要約を副とするデスクトップ配置
- [x] 1100px、640pxでの段階的な単一カラム化
- [x] ライト・ダーク双方のsurface、border、focusトークン
- [x] skip link、landmark、見出し参照、フォーカス可能なログ
- [x] `prefers-reduced-motion` の既存対応を維持
- [x] 既存DOM IDとイベント接続を維持
- [x] Node構造テスト、JavaScript構文、Python回帰、SQLite/PostgreSQL/Docker CI

## 更新前後の実画面

| Viewport | 更新前 | 更新後 |
|---|---|---|
| Desktop 1440×1000 | [before-desktop.png](ui/before-desktop.png) | [after-desktop.png](ui/after-desktop.png) |
| Mobile 390×844 | [before-mobile.png](ui/before-mobile.png) | [after-mobile.png](ui/after-mobile.png) |

スクリーンショットは同一HTML、同一viewportをヘッドレスChromeでレンダリングし、認証オーバーレイだけを非表示にしてワークスペースを比較したもの。
