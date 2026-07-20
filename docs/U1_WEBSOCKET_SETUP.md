# Python ↔ Unity WebSocket JSON通信

## ファイル

- `src/kokoro_shogi/server/server.py`: 再利用可能なPythonサーバーモジュール
- `scripts/server_gui.py`: Tkinterサーバー管理GUI
- `unity/KokoroShogi/Assets/_Project/Scripts/Net/`: Unity通信コード
- Python依存ライブラリはルートの `pyproject.toml` で管理
- JSONLは `unity/KokoroShogi/Assets/StreamingAssets/sample_data/` に配置

## 1. Python側の準備

```bash
cd kokoro-shogi
uv sync
uv run python scripts/server_gui.py
```

GUIで待受IPを `0.0.0.0`、ポートを `8765` にして「サーバー起動」を押します。

ファイアウォールの確認画面が出た場合は、利用するプライベートネットワークでPythonの通信を許可してください。

## 2. Unity側の準備

1. `unity/KokoroShogi/Assets/_Project/Scripts/Net/UnityWebSocketClient.cs` を使用します。
2. 最初のシーンに空のGameObjectを作り、名前を `WebSocket` にします。
3. `UnityWebSocketClient` をアタッチします。
4. `Server Ip` にPythonサーバーPCのIPv4アドレス、`Server Port` に `8765` を設定します。
5. PythonのGUIサーバーを起動してからUnityをPlayします。

同じPCで試す場合は `127.0.0.1` を使用できます。別PC・別端末から接続する場合は、Python側PCのLAN内IPv4アドレスを指定します。`0.0.0.0` はサーバーの待受指定であり、Unityの接続先には使用しません。

## 3. Unityから送信

```csharp
UnityWebSocketClient.Instance.SendText("success");
```

実際の送信データ:

```json
{"text":"success"}
```

JSON文字列をそのまま送る場合:

```csharp
UnityWebSocketClient.Instance.SendRawJson(
    "{\"type\":\"result\",\"success\":true}"
);
```

UI入力欄から接続先を設定する場合は、別のUIスクリプトから次を呼びます。

```csharp
UnityWebSocketClient.Instance.ConfigureAndConnect(
    ipInput.text,
    portInput.text
);
```

## 4. Pythonモジュールを直接使う例

```python
import time
from kokoro_shogi.server import server as ws_server

ws_server.start_server()  # 0.0.0.0:8765

try:
    while True:
        clients = ws_server.get_connected_clients()
        if clients:
            client_ip = clients[0]

            # テキストは {"text":"success"} に変換される
            ws_server.send_client(client_ip, "success")

            # JSONファイルの中身を送信
            ws_server.send_client(client_ip, "sample.json")

            received = ws_server.receive_client(client_ip)
            if received is not None:
                print("Unityから受信:", received)

        time.sleep(0.1)
finally:
    ws_server.end_server()
```

## 仕様上の注意

- GUIの接続一覧はIP単位です。同じIPから複数のUnityクライアントが接続している場合、`send_client()` はそのIPの全接続へ送信します。
- Python・Unity間のアプリケーションメッセージはUTF-8のJSONテキストです。
- JSONファイルは一度Pythonで解析し、妥当なJSONであることを確認してから送信します。
- 既定の最大受信サイズは64 MiBです。
- このUnity実装は `ClientWebSocket` を利用できるEditor/Standalone向けです。WebGLではJavaScript側WebSocketを使う別実装が必要です。
