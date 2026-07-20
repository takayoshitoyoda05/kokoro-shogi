# Python ↔ Unity WebSocket通信システム

PythonをWebSocketサーバー、UnityをWebSocketクライアントとして動作させ、JSONデータやテキストを双方向に送受信するためのプログラムです。

この構成には、次の4ファイルがあります。

|ファイル|役割|
|-|-|
|`ws\_server.py`|WebSocketサーバー本体。接続管理、送信、受信を担当します。|
|`server\_gui.py`|`ws\_server.py`を操作するTkinter GUIアプリです。|
|`UnityWebSocketClient.cs`|Pythonサーバーへ接続するUnityクライアントです。|
|`keyboard\_action.cs`|Unity上でキーボードから接続・切断・送信・接続先変更を試すためのスクリプトです。|

\---

## 1\. 通信の構成

```text
Python PC
  ws\_server.py
       ↑
  server\_gui.py
       │
       │ WebSocket / UTF-8 JSON
       │
Unity
  UnityWebSocketClient.cs
       ↑
  keyboard\_action.cs
```

* Pythonがサーバーです。
* Unityがクライアントです。
* Pythonサーバーは通常、`0.0.0.0:8765`で待ち受けます。
* Unityには、Pythonを実行しているPCの実際のローカルIPアドレスとポート番号を設定します。
* 同じPC上でPythonとUnityを動かす場合は、Unityの接続先IPに`127.0.0.1`を指定できます。

> `0.0.0.0`はサーバーがすべてのネットワークインターフェースで待ち受けるための値です。Unity側の接続先IPには使用しません。

\---

# Python側

## 2\. 動作環境

推奨環境は次のとおりです。

* Python 3.10以上
* `websockets`ライブラリ
* Tkinter

`json`、`asyncio`、`threading`、`queue`、`socket`などはPythonの標準ライブラリです。

### Windows

通常、Python公式インストーラーでPythonを導入するとTkinterも含まれます。

### Ubuntu・WSL

Tkinterが入っていない場合は、次を実行します。

```bash
sudo apt update
sudo apt install python3-tk
```

WSL上でGUIを表示するには、WSLgまたは別途GUI表示環境が必要です。Windows上で直接Pythonを実行する方が簡単です。

\---

## 3\. Pythonライブラリのインストール

仮想環境を使用する場合は、次のように準備します。

### Windows PowerShell

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "websockets>=13"
```

### Linux・WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "websockets>=13"
```

\---

## 4\. Pythonファイルの配置

`ws\_server.py`と`server\_gui.py`を同じフォルダに配置してください。

```text
PythonWebSocketServer/
├── ws\_server.py
└── server\_gui.py
```

`server\_gui.py`は次のように`ws\_server.py`を読み込みます。

```python
import ws\_server
```

そのため、ファイル名は原則として`ws\_server.py`にしてください。

\---

## 5\. `ws\_server.py`

WebSocketサーバー本体です。サーバー処理は別スレッド上のasyncioイベントループで動作します。

主な公開関数は次のとおりです。

### サーバー起動

```python
ws\_server.start\_server(host="0.0.0.0", port=8765)
```

* `host="0.0.0.0"`：すべてのネットワークインターフェースで待ち受けます。
* `port=8765`：使用するポート番号です。
* すでに起動している場合は`False`を返します。

### サーバー終了

```python
ws\_server.end\_server()
```

接続中のクライアントを閉じ、WebSocketサーバーを終了します。

### 指定IPのクライアントへ送信

```python
ws\_server.send\_client("192.168.1.50", "hello")
```

通常テキストは、自動的に次のJSONへ変換されます。

```json
{"text":"hello"}
```

JSONファイルを送信する場合は、ファイルパスを渡します。

```python
from pathlib import Path

ws\_server.send\_client(
    "192.168.1.50",
    Path("sample.json")
)
```

JSONファイルは内容を読み取り、妥当なJSONであることを確認してから送信されます。

### 指定IPから受信

```python
data = ws\_server.receive\_client("192.168.1.50")

if data is not None:
    print(data)
```

未受信の場合は`None`を返します。

### 接続中クライアントの取得

```python
client\_ips = ws\_server.get\_connected\_clients()
print(client\_ips)
```

### 送信元を問わず受信

```python
record = ws\_server.receive\_any()

if record is not None:
    print(record\["client\_ip"])
    print(record\["data"])
```

\---

## 6\. `server\_gui.py`

`ws\_server.py`をGUIから操作するTkinterアプリです。

### 起動方法

```bash
python server\_gui.py
```

### GUIの主な機能

* WebSocketサーバーの起動と終了
* 待受IPとポート番号の設定
* Unityが接続するときに使用するローカルIPとURLの表示
* 接続中クライアントIPの一覧表示
* 選択したクライアントへのテキスト送信
* 選択したクライアントへのJSONファイル送信
* Unityから受信した`text`内容の表示
* 接続、切断、送信、受信ログの表示

### 推奨設定

```text
待受IP: 0.0.0.0
ポート: 8765
```

サーバーを起動すると、GUI上部に次のような接続先が表示されます。

```text
接続先: ws://192.168.1.20:8765
同一PCから: ws://127.0.0.1:8765
```

別PC上のUnityから接続する場合は、最初のローカルIPをUnityへ設定します。

### テキスト送信

1. 接続中クライアントIPを選択します。
2. 「テキスト」を選択します。
3. 送信内容を入力します。
4. 「送信」を押します。

PythonからUnityへは、次のJSON形式で送信されます。

```json
{"text":"入力した内容"}
```

### JSONファイル送信

1. 接続中クライアントIPを選択します。
2. 「JSONファイル」を選択します。
3. 「参照」から`.json`ファイルを選択します。
4. 「送信」を押します。

JSONファイルの中身が、そのままJSONメッセージとしてUnityへ送信されます。

### Unityからの受信表示

Unityから次のデータが届いた場合、

```json
{"text":"テストメッセージ1回目"}
```

「クライアント受信テキスト」欄には、次のように本文が表示されます。

```text
テストメッセージ1回目
```

JSON全体は「受信・接続ログ」にも表示されます。

\---

# Unity側

## 7\. Unityの動作環境

Unity側では、次の機能を使用します。

* `ClientWebSocket`
* 新しいInput System
* `DontDestroyOnLoad`

### Input Systemの導入

`keyboard\_action.cs`は新しいInput Systemを使用します。

1. Unityのメニューから`Window > Package Manager`を開きます。
2. `Input System`を検索してインストールします。
3. `Edit > Project Settings > Player`を開きます。
4. `Active Input Handling`を次のどちらかに設定します。

   * `Input System Package (New)`
   * `Both`
5. Unityから再起動を求められた場合は再起動します。

`keyboard\_action.cs`は`UnityEngine.InputSystem.Keyboard`を使用しているため、旧Input APIの`Input.GetKeyDown()`は使用しません。

\---

## 8\. Unityファイルの配置

2つのC#ファイルをUnityプロジェクトへ配置します。

```text
Assets/
└── Scripts/
    ├── UnityWebSocketClient.cs
    └── keyboard\_action.cs
```

ファイル名とクラス名は一致させてください。

```text
UnityWebSocketClient.cs → UnityWebSocketClient
keyboard\_action.cs      → keyboard\_action
```

\---

## 9\. `UnityWebSocketClient.cs`のアタッチ方法

### 1\. 空のGameObjectを作成

最初のシーンでHierarchyを右クリックし、次を選択します。

```text
Create Empty
```

GameObject名の例：

```text
WebSocketManager
```

### 2\. スクリプトをアタッチ

`UnityWebSocketClient.cs`を`WebSocketManager`へドラッグしてアタッチします。

### 3\. ルートObjectとして配置

`UnityWebSocketClient`は`DontDestroyOnLoad`を使用します。`WebSocketManager`は別のGameObjectの子にせず、Hierarchyのルートに配置してください。

### 4\. Inspector設定

|項目|内容|
|-|-|
|`Server Ip`|PythonサーバーPCのローカルIP。例：`192.168.1.20`|
|`Server Port`|Pythonサーバーのポート。標準は`8765`|
|`AutoConnect`|ONならObject登場時に自動接続します。|
|`EchoSignal`|ONならメッセージ受信後に`success\_echo`を返信します。|
|`Echo Preview Length`|echoへ含める受信内容の先頭文字数です。|
|`On Message Received`|メッセージ受信時に任意の処理を登録できます。|
|`On Connection Status Changed`|接続状態変更時に任意の処理を登録できます。|

同じPC上のPythonへ接続する場合：

```text
Server Ip: 127.0.0.1
Server Port: 8765
```

別PC上のPythonへ接続する場合：

```text
Server Ip: Python GUIに表示されたローカルIP
Server Port: 8765
```

### AutoConnectの挙動

`AutoConnect`がONの場合、`Start()`で自動的に`Connect()`が実行されます。

OFFの場合は自動接続せず、次の公開関数が呼ばれたときに接続します。

```csharp
UnityWebSocketClient.Instance.Connect();
```

Unity UIのButtonから使用する場合は、Buttonの`On Click()`へ`WebSocketManager`を登録し、次を選択します。

```text
UnityWebSocketClient > Connect()
```

### 接続解除

次の公開関数で接続を解除できます。

```csharp
UnityWebSocketClient.Instance.Disconnect();
```

Buttonから使用する場合：

```text
UnityWebSocketClient > Disconnect()
```

### メッセージ送信

送信関数は`Send(string data)`です。

```csharp
UnityWebSocketClient.Instance.Send("success");
```

通常テキストは、次のJSONへ変換されます。

```json
{"text":"success"}
```

JSONオブジェクトまたはJSON配列形式の文字列を渡した場合は、そのまま送信します。

```csharp
UnityWebSocketClient.Instance.Send(
    "{\\"grade\\":\\"秀\\",\\"success\\":true}"
);
```

### 受信時の挙動

PythonからJSONを受信すると、Unity Consoleへ次の形式で表示されます。

```text
\[WebSocket RECEIVE] {"grade":"秀"}
```

また、Inspectorの`On Message Received`イベントにも受信文字列が渡されます。

### EchoSignalの挙動

`EchoSignal`がONの場合、受信後に次の形式のテキストをPythonへ返信します。

```text
success\_echo\[受信内容の先頭部分 ・・・]
```

返信には通常送信と同じ`Send()`関数が使用されるため、実際の送信JSONは次の形式です。

```json
{"text":"success\_echo\[受信内容の先頭部分 ・・・]"}
```

Unity Consoleには、echo要求と送信結果が表示されます。

```text
\[WebSocket ECHO REQUEST] success\_echo\[...]
\[WebSocket SEND REQUEST] {"text":"success\_echo\[...]"}
\[WebSocket SEND SUCCESS] {"text":"success\_echo\[...]"}
```

### シーン遷移時の挙動

`UnityWebSocketClient`をアタッチしたGameObjectは`DontDestroyOnLoad`により、シーンを移動しても破棄されません。

別シーンに同じ`UnityWebSocketClient`を持つObjectが存在した場合は、Singleton処理によって後から作られた重複Objectが削除されます。

\---

## 10\. `keyboard\_action.cs`のアタッチ方法

### 1\. 空のGameObjectを作成

Hierarchyへ新しい空Objectを作成します。

GameObject名の例：

```text
KeyboardAction
```

### 2\. スクリプトをアタッチ

`keyboard\_action.cs`を`KeyboardAction`へアタッチします。

### 3\. WebSocket参照を設定

Inspectorの`Web Socket Client`欄へ、`UnityWebSocketClient`をアタッチした`WebSocketManager`をドラッグします。

未設定の場合は、実行時に次のSingletonから自動取得します。

```csharp
UnityWebSocketClient.Instance
```

ただし、確実に参照させるため、Inspectorで設定する方法を推奨します。

### 4\. Show Input Overlay

`Show Input Overlay`がONの場合、IPまたはPortの入力中にGameビュー左上へ入力内容を表示します。

\---

## 11\. キーボード操作

|キー|動作|
|-|-|
|`C`|現在設定されているIPとPortへ接続します。|
|`D`|WebSocket接続を解除します。|
|`S`|テストメッセージを送信します。|
|`I`|IPアドレス変更モードを開始します。|
|`P`|Port番号変更モードを開始します。|
|`Enter`|IPまたはPortの入力を確定します。|
|`Backspace`|入力中の最後の1文字を削除します。|
|`Escape`|IPまたはPortの入力をキャンセルします。|

### Sキーによるテスト送信

Sキーを押すたびに回数が増えます。

```text
テストメッセージ1回目
テストメッセージ2回目
テストメッセージ3回目
```

Pythonへは次のJSONとして届きます。

```json
{"text":"テストメッセージ1回目"}
```

### IキーによるIP変更

1. `I`キーを押します。
2. IPアドレスを入力します。
3. `Enter`を押します。

例：

```text
192.168.1.20
```

IPv4の4区画がそれぞれ`0～255`の範囲である場合のみ変更されます。

成功ログ：

```text
\[success]IP\_change 127.0.0.1 > 192.168.1.20
```

不正な例：

```text
192.168.1
192.168.1.300
192.168.a.10
```

### PキーによるPort変更

1. `P`キーを押します。
2. Port番号を入力します。
3. `Enter`を押します。

Portは`1～65535`のみ受理されます。

成功ログ：

```text
\[success]Port\_change 8765 > 9000
```

### 接続中にIPまたはPortを変更した場合

設定値は変更されますが、現在確立されている接続先は自動では切り替わりません。

次の順番で再接続してください。

```text
Dキーで切断
↓
Cキーで新しい接続先へ接続
```

\---

# 動作確認

## 12\. 最短の確認手順

### Python

1. `ws\_server.py`と`server\_gui.py`を同じフォルダへ置きます。
2. `websockets`をインストールします。
3. 次を実行します。

```bash
python server\_gui.py
```

4. 待受IPを`0.0.0.0`、Portを`8765`にします。
5. 「サーバー起動」を押します。
6. GUIに表示されたローカルIPを確認します。

### Unity

1. `UnityWebSocketClient.cs`と`keyboard\_action.cs`を`Assets/Scripts`へ置きます。
2. `WebSocketManager`へ`UnityWebSocketClient`をアタッチします。
3. `KeyboardAction`へ`keyboard\_action`をアタッチします。
4. `keyboard\_action`の参照欄へ`WebSocketManager`を設定します。
5. UnityWebSocketClientの`Server Ip`へPython GUIのローカルIPを入力します。
6. `Server Port`を`8765`にします。
7. UnityをPlayします。
8. `AutoConnect`がOFFなら`C`キーを押します。
9. Python GUIの接続中クライアント一覧にUnityのIPが表示されることを確認します。
10. Unityで`S`キーを押します。
11. Python GUIに`テストメッセージ1回目`が表示されることを確認します。
12. Python GUIからUnityへテキストまたはJSONを送信します。
13. Unity Consoleに`\[WebSocket RECEIVE]`が表示されることを確認します。
14. `EchoSignal`がONなら、Python GUIに`success\_echo\[...]`が返ることを確認します。

\---

# 通信データ形式

## 13\. テキスト

通常テキストは、Python側・Unity側ともに次の形式へ変換して送信します。

```json
{
  "text": "メッセージ内容"
}
```

## 14\. JSON

JSONオブジェクト例：

```json
{
  "grade": "秀",
  "damages": \[
    {
      "name": "傷",
      "size": 12.5
    }
  ]
}
```

JSON配列例：

```json
\[
  {"id": 1, "result": "success"},
  {"id": 2, "result": "failed"}
]
```

通信にはUTF-8を使用します。

\---

# トラブルシューティング

## 15\. Unityから接続できない

次を確認してください。

* Python GUIでサーバーを起動しているか
* Unityに`0.0.0.0`を設定していないか
* UnityのIPがPython GUIに表示されたローカルIPと一致しているか
* PythonとUnityのPortが一致しているか
* PythonとUnityのPCが同じLANに接続されているか
* Windows Defender FirewallでPythonの通信が許可されているか
* VPNや仮想ネットワークが別のIPを優先していないか

## 16\. Windows Firewall

初回起動時にPythonのネットワークアクセス確認が表示された場合は、使用するネットワークに対してアクセスを許可してください。

必要に応じて、使用中のPort、標準ではTCP `8765`を受信許可します。

## 17\. Input Systemのエラー

次のエラーが出る場合、旧Input APIを使用している別スクリプトが残っている可能性があります。

```text
InvalidOperationException: You are trying to read Input using the UnityEngine.Input class...
```

今回の`keyboard\_action.cs`は新Input Systemを使用しています。Unityプロジェクト内で次の記述を検索してください。

```csharp
Input.GetKeyDown
Input.inputString
```

別の古い`keyboard\_action.cs`が残っていないかも確認してください。

## 18\. echoが返らない

次を確認してください。

* `UnityWebSocketClient`の`EchoSignal`がONか
* Unity Consoleに`\[WebSocket RECEIVE]`が表示されているか
* Unity Consoleに`\[WebSocket ECHO REQUEST]`が表示されているか
* Unity Consoleに`\[WebSocket SEND SUCCESS]`が表示されているか
* 受信直後に接続が切れていないか

`SEND FAILED`が表示された場合は、そのログに原因が表示されます。

## 19\. Python GUIにUnityのテキストが表示されない

Unity Consoleで次の送信成功ログを確認してください。

```text
\[WebSocket SEND SUCCESS]
```

Python GUIでは、`{"text":"..."}`形式のデータは「クライアント受信テキスト」欄へ表示されます。それ以外のJSONは「受信・接続ログ」で確認してください。

## 20\. ファイル名に`(1)`が付いている

ダウンロード時に同名ファイルが存在すると、OSやブラウザによって次のような名前になる場合があります。

```text
ws\_server(1).py
server\_gui(1).py
UnityWebSocketClient(1).cs
keyboard\_action(1).cs
```

実際に使用するときは、次の名前へ変更してください。

```text
ws\_server.py
server\_gui.py
UnityWebSocketClient.cs
keyboard\_action.cs
```

特にUnityでは、MonoBehaviourのクラス名とファイル名を一致させる必要があります。

\---

# 補足

このプログラムはローカルネットワーク内での開発・検証を想定しています。通信は暗号化されていない`ws://`です。インターネット経由で利用する場合は、認証、入力検証、アクセス制限、TLSを使用した`wss://`などを追加してください。

