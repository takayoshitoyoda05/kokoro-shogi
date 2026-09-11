# U1 Unity-Python連携

## 駒のカメラ演出

`Shogi/PieceCameraAnimator.cs` は移動先が確定したらそのマスへ接近し、最接近後に
駒の移動・獲得・成りを実行する。処理後は注視時間を置いて元の位置へ戻る。
`GameSceneDirector` が起動時に自動追加するため、既存シーンでも動作する。
設定を保存して調整する場合は、編集モードで `GameSceneDirector` と同じオブジェクトに
`PieceCameraAnimator` を追加する（別オブジェクトの場合は Director の参照欄へ割り当てる）。

- `Target Camera`: 未設定なら MainCamera を使用。
- `Close Up Offset` / `Look At Offset`: 接近位置と注視点の調整。
- `Close Up Yaw`: 横への回り込み角度。既定値25度で斜めから駒を捉える。0度で正面、負の値で反対側。注視点までの距離と高さを保って回り込む。
- `Close Up Pitch`: X軸の角度補正。既定値0度。正の値で上から、負の値で低い位置から捉える。距離と注視点を保ち、最終的な見下ろし角度を5～85度に制限する。
- `Approach Duration` / `Hold Duration` / `Return Duration`: 接近・注視・帰還の秒数。
- `Close Up Orthographic Size`: 平行投影カメラのズーム量。

成りの確認は移動前に行い、確認中は盤面・駒の種類を変更しない。
「成らない」でも獲得があれば演出する。獲得と成りが同時なら1回の接近でまとめて実行する。
演出中は盤上クリックを止め、カメラが戻ってから次のターンに進む。
`FocusAt(destination, true)` で接近を開始し、`HasReachedFocus` が真になったら
駒の処理を実行して `CompleteAction()` を呼ぶと、注視・帰還を開始する。
単純な駒の注視には `FocusOn(piece.transform)` を使用できる。`Cancel()` や無効化で元へ戻す。
カメラ未設定・無効化時は演出を省略して着手を進める。

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
