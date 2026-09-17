# U1 Unity-Python連携

## Pythonの局面でUnityの駒を動かす

`MainGame` の `GameSceneDirector` が起動時に `ServerBoardSynchronizer` を自動追加する。
シーンへの追加設定は不要。既存の `UnityWebSocketClient` をPythonのサーバー
（既定 `127.0.0.1:8765`）に接続し、モード選択で対局を開始すると、受信した
`state_update` に合わせて全駒が移動する。

処理の流れは `UnityWebSocketClient.Update → MessageRouter → GameEvents.OnStateUpdated
→ ServerBoardSynchronizer → GameSceneDirector.ApplyServerState → UnitController`。

| ファイル | 責務・保持する状態 |
|---|---|
| `Net/UnityWebSocketClient.cs`, `MessageRouter.cs` | 接続・送受信・JSON解析（既存） |
| `Core/ReceivedBoardSnapshot.cs` | piecesとSFENの整合性、ID・座標・持ち駒の検証、手番の取得 |
| `Shogi/ServerBoardSynchronizer.cs` | 受信局面の待ち行列、現在の合法手、演出完了待ち |
| `Shogi/GameSceneDirector.Network.cs` | Directorの受信表示部分。IDとUnityの駒の対応、盤上・持ち駒の配置 |
| `Shogi/UnitController.cs` | 個々の駒の位置・種類・向きと `PieceId` |

- `piece_id` は恒久IDとして扱い、文字列に含まれる初期位置から現在位置を推測しない。
- `pieces` は盤上と持ち駒の全件が必要。`square: "hand"` が持ち駒で、着手の
  `from: "00"` は駒打ち。`owner: 0` は先手、`1` は後手。
- 筋段 `"76"` は既存盤配列の `(2, 3)`。SFENの数字は空マスの数であり、駒の個数ではない。
- SFENの `b/w` で手番、`species` で成り状態を確定する。獲得した駒は向きと成りを戻し、
  持ち駒欄にまとめる。駒打ちでは同じIDの駒を盤へ戻す。
- `last_move` の獲得・成りに既存カメラ演出を使う。演出中の次局面は受信順に待つ。
  初回の途中局面、同一局面の再送、リプレイの巻き戻しにも全駒の状態から対応する。
- 受信後の対局操作は `legal_moves` から候補を表示する。クリックでは盤を変更せず、
  `move_request` を送信してPythonが確定した局面を待つ。観戦用JSONLは合法手を送らないので操作不可。
- 不正な局面は盤を変更せず、Consoleの `[ServerBoard]` に理由を出す。

### modelサンプルでの確認（Pythonサーバー不要）

1. Unity Editorではリポジトリの `sample_data/model` を直接読み込めるため、コピー不要。
   ビルド版で再生する場合は `Assets/StreamingAssets/sample_data/model` へコピーする。
   `StreamingAssets` は `Assets` 直下（`Assets/_Project` の中ではない）。
2. `MainGame` の任意のGameObjectに `ReplayPlayer` を追加し、Relative Pathを
   `sample_data/model/game01.jsonl` に設定する。
   以前の初期値 `sample_data/game01.jsonl` も、Editorではmodel内の同名ファイルを探す。
3. Playモードでモード選択画面を閉じ、ReplayPlayerのコンポーネントメニューから
   `Replay/Load`、`Replay/Play` の順に実行する。サーバーとの同時再生はしない。
4. 最初に初期配置、次に先手の銀が `39 → 48`、後手の歩が `83 → 84` と動く。
   38手目に獲得、40手目に歩の駒打ちを確認できる。
   `Replay/Pause`、`Replay/Step Forward`、`Replay/Step Back` で1局面ずつ確認できる。

検証コマンド（リポジトリ直下、PowerShell 7）:
`pwsh -NoProfile -File scripts/test_unity_received_board.ps1`
modelの10ファイル全局面を実際のC#検証器へ通し、不正な局面を拒否することも確認する。

## 駒のカメラ演出

`Shogi/PieceCameraAnimator.cs` は移動先が確定したらそのマスへ接近し、最接近後に
駒の移動・獲得・成りを実行する。処理後は注視時間を置いて元の位置へ戻る。
`GameSceneDirector` が起動時に自動追加するため、既存シーンでも動作する。
設定を保存して調整する場合は、編集モードで `GameSceneDirector` と同じオブジェクトに
`PieceCameraAnimator` を追加する（別オブジェクトの場合は Director の参照欄へ割り当てる）。

- `Enable Piece Camera`: カメラ演出のオン・オフ。既定はオン。オフでも駒の移動・獲得・成りは通常どおり進む。演出中にオフにすると元のカメラ位置へ戻る。
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
