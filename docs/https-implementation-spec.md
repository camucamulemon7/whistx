# whistx HTTPS 対応実装仕様書

## 目的と前提
- [ ] 社内証明書（サーバー証明書 / 秘密鍵 / 任意で中間証明書）が PEM 形式でホスト側に保存されていることを確認する。
- [ ] Podman もしくは Docker で `--device nvidia.com/gpu=all` を指定してコンテナを起動できる現行手順（`run.sh` または `podman-run.sh`）が稼働していることを確認する。
- [ ] HTTPS 公開ポート（例: 8443 等）と既存 HTTP ポート (8005) の使い分けを関係者と合意しておく。

## 設計方針
- [✅] TLS の終端をアプリコンテナ内の uvicorn に持たせ、SSL 設定は `run.sh` の環境変数から渡す運用とする。
- [✅] 証明書ファイル群はホストに置き、`docker/podman run` の volume マウントで `/app/certs` 以下へ読み取り専用で提供する。
- [✅] HTTP/HTTPS の両対応ができるよう、TLS 変数が未設定なら従来通り HTTP で起動するフォールバックを維持する。

## run.sh の改修
- [✅] スクリプト冒頭に HTTPS 用の設定変数を追加する（例: `TLS_CERT_HOST_PATH`, `TLS_KEY_HOST_PATH`, `TLS_CHAIN_HOST_PATH`, `TLS_CONTAINER_CERT_PATH`, `TLS_CONTAINER_KEY_PATH`, `TLS_CONTAINER_CHAIN_PATH`, `TLS_PORT`）。
- [✅] 既定値は `${SCRIPT_DIR}/certs/server.crt` 等のダミーパスにして、利用時には実際の社内証明書パスに上書きできるようにする。
- [✅] `docker run` コマンドに証明書ファイルの volume マウント (`--volume "$TLS_CERT_HOST_PATH:$TLS_CONTAINER_CERT_PATH:ro"` など) を条件付きで追加する。
- [✅] TLS 変数が空の場合はマウントも環境変数伝播もスキップする条件分岐を実装する。
- [✅] `docker run` のポート公開を `--publish ${TLS_PORT:-8443}:${TLS_PORT:-8443}` に変更し、デフォルトで HTTPS ポートを公開する（必要に応じて 8005 との二重公開をオプション化する）。
- [✅] TLS 関連の環境変数 (`TLS_CERT_FILE`, `TLS_KEY_FILE`, `TLS_CA_BUNDLE`, `TLS_PORT`) をコンテナへ渡す `--env` 追記を行う。

## start.sh の改修
- [✅] `TLS_CERT_FILE` と `TLS_KEY_FILE` が設定されているか検証し、存在しない場合はエラーメッセージを出して終了する処理を追加する（TLS を強制したい環境向け）。
- [✅] `TLS_CERT_FILE` / `TLS_KEY_FILE` が空の場合は現行通り HTTP (ポート 8005) で `uvicorn` を起動するフォールバックを残す。
- [✅] TLS が有効な場合は `UVICORN_TLS_ARGS=(--ssl-certfile "$TLS_CERT_FILE" --ssl-keyfile "$TLS_KEY_FILE")` を構成し、`TLS_CA_BUNDLE` が指定されたときは `--ssl-ca-certs` を追加する。
- [✅] 監視ポートを `TLS_PORT` （未設定時は 8443）に切り替え、HTTP フォールバック時のみ 8005 を使用するロジックを実装する。
- [✅] ログに HTTPS 起動を明示する (`echo "Starting uvicorn with HTTPS on port $TLS_PORT"`) ことで運用が確認しやすくなるようにする。

## 付随変更
- [✅] `Dockerfile` / `docker-compose.yaml` に変更は不要か確認し、不要であれば記録に残す。必要な場合（例: ポート番号更新）は合わせて修正する。
- [✅] `README.md` に HTTPS 起動手順（証明書配置・環境変数設定・接続 URL 変更）を追記する。
- [✅] 社内証明書の更新手順（例: 有効期限前に新しい PEM を差し替える方法）をドキュメントにまとめる。

## 動作確認
- [ ] `run.sh` で証明書パスを設定し直し、`./run.sh` 実行後に `https://localhost:8443/` へアクセスしてブラウザ警告が出ないことを確認する。
- [ ] WebSocket が `wss://` で接続され、リアルタイム転送が問題なく行えることをブラウザのデベロッパーツールで確認する。
- [ ] NeMo 推論が想定通り GPU で動作することをログで確認する。
- [ ] TLS 変数を未設定で起動し HTTP モードに戻れることを確認する（フォールバック検証）。
