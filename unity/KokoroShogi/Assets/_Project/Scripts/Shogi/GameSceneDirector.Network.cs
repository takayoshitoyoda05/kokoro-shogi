using System.Collections;
using System.Collections.Generic;
using KokoroShogi.Core;
using KokoroShogi.Net;
using UnityEngine;

// 受信局面の表示処理。通信の接続・JSON解析はNet、受信待ちはServerBoardSynchronizerが担当。
public partial class GameSceneDirector
{
    readonly Dictionary<string, UnitController> serverUnits = new Dictionary<string, UnitController>();
    public bool CanApplyServerState => units != null && unitTiles != null && nowMode != Mode.Animating &&
        nextMode != Mode.Animating && (!pieceCameraAnimator || !pieceCameraAnimator.IsPlaying);

    public static Vector2Int FromShogiSquare(string square)
    {
        if (!ReceivedBoardSnapshot.IsBoardSquare(square))
            throw new System.ArgumentException("盤上の座標は11～99の筋段です。", nameof(square));
        return new Vector2Int(9 - (square[0] - '0'), 9 - (square[1] - '0'));
    }

    public void CancelServerStateApplication()
    {
        if (pieceCameraAnimator) pieceCameraAnimator.Cancel();
        nowMode = Mode.Select;
        nextMode = Mode.None;
    }

    public void ShowServerResult(GameResult result)
    {
        setSelectCursors();
        movableTiles.Clear();
        pendingMoveUnit = null;
        pendingPlayerMove = null;
        buttonEvolutionApply.gameObject.SetActive(false);
        buttonEvolutionCancel.gameObject.SetActive(false);
        if (result.reason == "engine_no_move")
            textResultInfo.text = "AIの着手を取得できず対局終了";
        else if (result.winner == "draw")
            textResultInfo.text = "引き分け";
        else
        {
            int winner = result.winner == "black" ? 0 : 1;
            textResultInfo.text = winner == result.human ? "人間の勝ち" : "AIの勝ち";
        }
        textResultInfo.gameObject.SetActive(true);
        textResultInfo.enabled = true;
        textTurnInfo.text = "";
        buttonRematch.gameObject.SetActive(true);
        buttonTitle.gameObject.SetActive(true);
        nowMode = Mode.Result;
        nextMode = Mode.None;
    }

    public IEnumerator ApplyServerState(ReceivedBoardSnapshot snapshot)
    {
        StateUpdate message = snapshot.Message;
        buttonRematch.gameObject.SetActive(false);
        buttonTitle.gameObject.SetActive(false);
        setSelectCursors();
        movableTiles.Clear();
        pendingMoveUnit = null;
        pendingPlayerMove = null;
        buttonEvolutionApply.gameObject.SetActive(false);
        buttonEvolutionCancel.gameObject.SetActive(false);
        buttonTitle.gameObject.SetActive(false);
        buttonRematch.gameObject.SetActive(false);
        nowMode = Mode.Animating;
        nextMode = Mode.None;

        LastMove move = message.last_move;
        // 既に表示済みの駒が動くときだけ待つ。初回の局面表示や同じ局面の再送は待たない。
        bool isNewMove = move != null && ReceivedBoardSnapshot.IsBoardSquare(move.to) &&
            serverUnits.TryGetValue(move.piece_id ?? "", out UnitController previousUnit) &&
            (previousUnit.FieldStatus == FieldStatus.Captured || previousUnit.Pos != FromShogiSquare(move.to));
        if (isNewMove && snapshot.PlayerToMove != aiPlayer && aiMoveDelaySeconds > 0f)
            yield return new WaitForSeconds(aiMoveDelaySeconds);

        bool animate = move != null && (move.capture || move.promote) &&
            ReceivedBoardSnapshot.IsBoardSquare(move.to) &&
            serverUnits.TryGetValue(move.piece_id ?? "", out UnitController movingUnit) &&
            (movingUnit.FieldStatus == FieldStatus.Captured || movingUnit.Pos != FromShogiSquare(move.to)) &&
            pieceCameraAnimator && pieceCameraAnimator.IsAvailable;
        if (animate)
        {
            Vector3 focus = tiles[FromShogiSquare(move.to)].transform.position;
            focus.y = UnitController.UnSelectUnitY;
            pieceCameraAnimator.FocusAt(focus, true);
            while (pieceCameraAnimator && pieceCameraAnimator.IsPlaying && !pieceCameraAnimator.HasReachedFocus)
                yield return null;
        }

        ApplySnapshotPieces(message.pieces);
        nowPlayer = snapshot.PlayerToMove;
        SetMoveCount(message.ply);
        textTurnInfo.text = (nowPlayer + 1) + "Pの番です";
        textResultInfo.text = "";
        if (animate)
        {
            if (pieceCameraAnimator) pieceCameraAnimator.CompleteAction();
            while (pieceCameraAnimator && pieceCameraAnimator.IsPlaying) yield return null;
        }
        nowMode = Mode.Select;
    }

    void ApplySnapshotPieces(List<PieceInfo> pieces)
    {
        // 最初の受信だけ既存の生駒を対応付ける。IDの文字列を位置として解釈しない。
        var available = new List<UnitController>(captureUnits);
        foreach (UnitController unit in units) if (unit) available.Add(unit);
        var retained = new HashSet<UnitController>();
        var nextUnits = new Dictionary<string, UnitController>();
        foreach (PieceInfo piece in pieces)
        {
            UnitType baseType = (UnitType)ReceivedBoardSnapshot.BaseTypeNumber(piece.species);
            serverUnits.TryGetValue(piece.piece_id, out UnitController unit);
            if (unit && unit.OldUnitType != baseType) unit = null;
            if (!unit)
            {
                // 既知IDの駒を別のIDに流用しない。初回や別対局の新規IDのみ割り当てる。
                unit = available.Find(candidate => !retained.Contains(candidate) &&
                    !serverUnits.ContainsValue(candidate) && candidate.OldUnitType == baseType &&
                    candidate.Player == piece.owner && piece.square != "hand" &&
                    candidate.Pos == FromShogiSquare(piece.square));
                if (!unit) unit = available.Find(candidate => !retained.Contains(candidate) &&
                    !serverUnits.ContainsValue(candidate) && candidate.OldUnitType == baseType);
            }
            if (!unit)
            {
                GameObject obj = Instantiate(prefabUnits[(int)baseType - 1]);
                if (!obj.GetComponent<Rigidbody>()) obj.AddComponent<Rigidbody>();
                unit = obj.GetComponent<UnitController>();
                if (!unit) unit = obj.AddComponent<UnitController>();
            }
            retained.Add(unit);
            nextUnits.Add(piece.piece_id, unit);
        }
        foreach (UnitController unit in available)
            if (!retained.Contains(unit)) { unit.gameObject.SetActive(false); Destroy(unit.gameObject); }

        units = new UnitController[boardWidth, boardHeight];
        captureUnits.Clear();
        serverUnits.Clear();
        foreach (PieceInfo piece in pieces)
        {
            UnitController unit = nextUnits[piece.piece_id];
            serverUnits.Add(piece.piece_id, unit);
            unit.PieceId = piece.piece_id;
            bool inHand = piece.square == "hand";
            Vector2Int position = inHand ? Vector2Int.zero : FromShogiSquare(piece.square);
            UnitType baseType = (UnitType)ReceivedBoardSnapshot.BaseTypeNumber(piece.species);
            bool promoted = ReceivedBoardSnapshot.IsPromoted(piece.species);
            bool wasPromoted = unit.UnitType != unit.OldUnitType;
            FieldStatus fieldStatus = inHand ? FieldStatus.Captured : FieldStatus.OnBoard;
            bool changed = unit.OldUnitType != baseType || unit.Player != piece.owner ||
                unit.FieldStatus != fieldStatus || wasPromoted != promoted ||
                (!inHand && unit.Pos != position);

            // Initは高さも戻すので、変化のない駒には触れず、物理で落ち着いた位置を保つ。
            if (changed)
            {
                unit.gameObject.SetActive(true);
                unit.Init(piece.owner, (int)baseType, tiles[position], position);
                // 生駒から成りを適用することで、成り解除やリプレイの巻き戻しにも対応する。
                if (inHand) unit.Capture(piece.owner);
                else
                {
                    unit.Evolution(promoted);
                    unit.GetComponent<Rigidbody>().isKinematic = false;
                }
            }
            if (inHand) captureUnits.Add(unit);
            else units[position.x, position.y] = unit;
        }
        alignCaptureUnits(0);
        alignCaptureUnits(1);
    }
}
