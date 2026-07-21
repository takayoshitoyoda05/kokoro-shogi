# U1 Unity-Python連携

このフォルダのC#ファイルは `docs/INTERFACE.md` Schema 1.0 に対応する。

## Unityへの配置

- `Net/`: `UnityWebSocketClient`, メッセージ型、ルーター
- `Core/`: U2向けイベントとSFEN盤面状態
- `Replay/`: `StreamingAssets/sample_data/*.jsonl` の再生

最初のシーンのルートGameObjectへ `UnityWebSocketClient` を追加する。
`ReplayPlayer` を使う場合は別のGameObjectへ追加し、`Relative Path` に
`sample_data/game01.jsonl` のようなStreamingAssetsからの相対パスを設定する。

受信した `state_update`, `legal_moves`, `career`, `game_control` は
`GameEvents` に型付きで通知される。U2側はJSONを直接解析せず、このイベントを購読する。

`EchoSignal` は旧疎通確認用であり、本番ではOFFにする。既定値もOFFである。

Unityからの送信には、汎用 `Send` ではなく次を使う。

```csharp
UnityWebSocketClient.Instance.SendGameControl("start");
UnityWebSocketClient.Instance.SendMoveRequest(new KokoroShogi.Net.LegalMove {
    from = "77", to = "76", promote = false
});
```
