# 終局通知のPython実装手順

作成日：2026-09-19

これはPython担当者が実装するための手順書です。以下のPythonコードは**未適用**です。
背景・依頼内容は [引き継ぎ文書](AI_RESULT_HANDOFF.md) を先に参照してください。

## 1. 変更範囲を確認する

| ファイル | 対応 |
|---|---|
| `src/kokoro_shogi/logging/jsonl.py` | 終局結果の型と、任意の `career.result` を追加 |
| `src/kokoro_shogi/server/game_session.py` | 終局・投了時に結果を付けて返す |
| `tests/test_game_session.py` | 勝者、終了理由、送信順、継続できる局面を検証 |
| `docs/INTERFACE.md` | 採用する形式・値域・送信順・バージョンを正式に記録 |

通信の仕組みや学習済みモデルを変更する必要はありません。
`scripts/play_server.py` はセッションが返すメッセージをJSONにして送るため、
今回の形式を採用するだけならサーバー起動スクリプトへの追加処理は不要です。

以下は現在のコードに対する最小の追加例です。schemaの番号はコード例では変更していません。
正式採用時のバージョン変更は `docs/INTERFACE.md` §7 に合わせ、Unity担当と同時に行ってください。

## 2. 結果のデータ型を追加する

`src/kokoro_shogi/logging/jsonl.py` の `CareerMessage` の直前に追加します。
`Literal`、`Any`、`model_serializer`、`SerializerFunctionWrapHandler` は既にインポートされています。

```python
class GameResult(_Strict):
    winner: Literal["black", "white", "draw"]
    human: Literal[0, 1]
    reason: Literal[
        "checkmate", "no_legal_moves", "repetition",
        "max_plies", "resign", "engine_no_move",
    ]
```

`CareerMessage` を次の形にします。既存のフィールドを残し、`result` を任意項目にします。

```python
class CareerMessage(_Message):
    type: Literal["career"] = "career"
    pieces: list[CareerPiece] = Field(default_factory=list)
    last_game_mvp: CareerMvp | None = None
    result: GameResult | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_result(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if self.result is None:
            data.pop("result", None)
        return data
```

`result` がないときはキーそのものを省略します。これにより、既存の成績JSONの出力を保ち、
旧形式をそのまま読み書きできるか確認している `tests/test_jsonl.py` のテストも維持できます。
`GameResult` は `career` の内部データなので、新たなメッセージ種別として
`MESSAGE_MODELS` に登録する必要はありません。

## 3. 終局時に勝者と理由を付ける

`src/kokoro_shogi/server/game_session.py` の
`from kokoro_shogi.logging.jsonl import (...)` に `GameResult` を追加します。

続いて `GameSession._finish()` を次に置き換えます。

```python
def _finish(self, *, reason: str | None = None) -> list[_Message]:
    assert self.tracker is not None
    winner = "draw"

    if reason == "resign":
        winner = "black" if self.ai == BLACK else "white"
    elif self.board.is_game_over():
        winner = "white" if self.board.turn == BLACK else "black"
        reason = "checkmate" if self.board.is_check() else "no_legal_moves"
    elif self.board.is_draw() == cshogi.REPETITION_DRAW:
        reason = "repetition"
    elif self.ply >= self.max_plies:
        reason = "max_plies"
    else:
        # 合法手があるのにエンジンがNoneを返した異常終了。
        # 詰みや人間・AIの勝ちと誤表示しない。
        reason = "engine_no_move"

    self.ledger.record(self.tracker.states, self._promoted)
    career = self.ledger.message()
    career.result = GameResult(winner=winner, human=self.human, reason=reason)
    self._to_idle()
    return [career]
```

判断のポイントは次のとおりです。

- `board.turn` は**現在これから指す側**です。最後の着手後に詰んでいれば、その反対側が勝者です。
- `black` は常に先手であり、人間とは限りません。人間側は必ず `self.human` で通知します。
- 詰みと手数上限が同じ手で成立したときは、上記の順序では詰みを優先します。
- エンジンの戻り値が `None` というだけで「詰み」と決めつけません。
- 異常終了の `engine_no_move` はこの案では `winner="draw"` にし、Unityは通常の引き分け表示より先に
  `reason` を見て「AIの着手を取得できず対局終了」と表示します。正式な成績評価としての引き分けを意味しません。
- ここでは既存の `_finished()` が検出する終局範囲を維持します。反復規則全般の拡張は別対応です。

`_on_move_request()` と `_ai_reply()` は既に最終 `state_update` の後へ `_finish()` の戻り値を
追加しています。この順序を保ち、終局後に通常の `legal_moves` を追加しないでください。

## 4. 投了も同じ通知経路にまとめる

`GameSession._on_control()` の `resign` 分岐だけを次に置き換えます。
`start` と `reset` は現在の動作を維持します。

```python
if message.command == "resign":
    if self.phase is not Phase.IDLE and self.tracker is not None:
        return self._finish(reason="resign")
    self._to_idle()
    return []
```

これにより、投了でも成績集計と勝敗通知が1回だけ実行されます。
待機中の投了には何も返しません。`reset` は対局破棄なので、勝敗を新たに送らないでください。

## 5. Pythonのテストを追加する

`tests/test_game_session.py` にある既存の詰み・投了・手数上限テストを利用します。
既存のアサーションを残したまま、以下を追記してください。

### 人間が詰ませた場合

`test_checkmate_by_human_ends_game_with_career()` の最後へ追加します。

```python
assert out[1].result.winner == "black"
assert out[1].result.human == BLACK
assert out[1].result.reason == "checkmate"
assert validate_line(to_json_line(out[1])).result == out[1].result
```

### 投了と手数上限

`test_resign_reports_career_and_returns_to_idle()` の最後へ追加します。

```python
assert out[0].result.winner == "white"
assert out[0].result.reason == "resign"
```

`test_max_plies_ends_game_as_draw()` の最後へ追加します。

```python
assert out[-1].result.winner == "draw"
assert out[-1].result.reason == "max_plies"
```

### AIが詰ませた場合を先手・後手の両方で確認

同じテストファイルに以下を追加します。`RandomEngine`、`start`、`request` と必要な
インポートは同ファイルに既にあります。学習済みモデルを使わず、詰ませる手を固定します。

```python
@pytest.mark.parametrize("human", [BLACK, WHITE])
def test_ai_checkmate_reports_winner_after_final_state(human):
    class MatingEngine(RandomEngine):
        def step(self, board, tracker, ply, record, *, ai_moved):
            state, move = super().step(
                board, tracker, ply, record, ai_moved=ai_moved,
            )
            if ply == 0:
                move = board.move_from_usi("6g5h" if human == BLACK else "6c5b")
                assert board.is_legal(move)
            return state, move

    sfen = (
        "4r3k/9/9/9/9/9/3g5/9/4K4 w - 1" if human == BLACK
        else "4k4/9/3G5/9/9/9/9/9/4R3K b - 1"
    )
    session = GameSession(MatingEngine(), human=human, initial_sfen=sfen)
    out = start(session)
    assert [type(m) for m in out] == [StateUpdate, StateUpdate, CareerMessage]
    assert session.board.is_game_over()
    assert session.phase is Phase.IDLE
    result = validate_line(to_json_line(out[-1])).result
    assert result.winner == ("white" if human == BLACK else "black")
    assert result.human == human
    assert result.reason == "checkmate"
    assert request(session, {"from": "59", "to": "49"}) == []
```

### 王手だけでは終了しないこと、再開できることを確認

```python
def test_check_with_escape_does_not_report_result():
    session = GameSession(
        RandomEngine(), human=WHITE,
        initial_sfen="4k4/9/9/9/9/9/9/9/4R3K w - 1",
    )
    out = start(session)
    assert session.board.is_check() and not session.board.is_game_over()
    assert [type(m) for m in out] == [StateUpdate, LegalMovesMessage]
    assert out[-1].moves and session.phase is Phase.HUMAN_TURN


def test_new_game_after_result_has_no_stale_result():
    session = GameSession(RandomEngine())
    start(session)
    session.handle({"schema": "1.0", "type": "game_control", "command": "resign"})
    out = start(session)
    assert [type(m) for m in out] == [StateUpdate, LegalMovesMessage]
    assert session.phase is Phase.HUMAN_TURN
    assert session.ledger.message().result is None
```

既存の `tests/test_jsonl.py` も一緒に実行し、`result` のない旧形式の成績データを保てているか確認します。

```powershell
uv run --group train python -m pytest tests/test_game_session.py tests/test_jsonl.py -q
```

既に依存が入っているWindows環境では、同期せず次でも実行できます。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_game_session.py tests/test_jsonl.py -q
```

この実装案に相当する変更は、一度ローカルで適用し、上記2ファイルのテストが
**55成功・2スキップ**となることを確認しました。その後、担当分離のためPythonの変更は取り消しました。
これは過去の検証結果です。担当者が適用した変更は改めてテストしてください。
千日手・異常終了など、上記の追加例が直接カバーしていない分岐は必要に応じてテストを補ってください。

## 6. Unityと接続して確認する

1. 正式な通信仕様・バージョンをUnity担当と揃えます。
2. 起動中のPythonサーバーを `Ctrl+C` で停止します。
3. リポジトリ直下で次を実行します。

   ```powershell
   uv run --group train python scripts/play_server.py --host 0.0.0.0
   ```

4. UnityでPlayを停止してから再開し、対局します。
5. 終局時の受信JSONに `career.result` があり、勝者と人間の手番が正しいことを確認します。
6. 最後の着手・カメラ演出後に、`MainGameCanvas/TextResultInfo` に結果が表示されることを確認します。
7. 再戦・タイトルボタンが表示され、盤面操作が止まっていることを確認します。

Python単体のテストとUnity画面の確認は別です。上記のSFENはPythonテスト用であり、
本番の `play_server.py` に局面指定の引数を追加する手順ではありません。
詰みをすぐ再現できない場合でも、投了の通知で結果表示の接続確認ができます。

## 7. 表示されない場合の切り分け

| 確認内容 | 次に見る場所 |
|---|---|
| `career` は来るが `result` がない | サーバーを再起動したか、`GameSession._finish()` の変更が適用されているか |
| `result` 自体が送られない | サーバーの例外、現在の盤面が本当に終局か、`_finished()` の戻り値 |
| 勝者が逆になる | `board.turn` と勝者を取り違えていないか、`human` が固定値になっていないか |
| 結果が早すぎる／後で消える | Unityが局面・演出を反映し終えてから `ShowServerResult()` を呼んでいるか |
| JSONは正しいが表示されない | Unityの受信ログ、`ReceiveCareer()`、`TextResultInfo` の参照と親Canvas |
| schema不一致の警告が出る | PythonとUnity双方のバージョン定数と実際の送信JSON |

Python担当者はまず「終局直前の `state_update` と、その直後の `career`」を共有すると、
Unity担当者が表示処理との接続を確認できます。
