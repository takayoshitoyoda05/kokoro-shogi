using System;
using System.Collections;
using System.Collections.Generic;
using KokoroShogi.Core;
using KokoroShogi.Net;
using UnityEngine;

/// <summary>受信順を保ち、検証済み局面をGameSceneDirectorへ渡す。</summary>
[DisallowMultipleComponent]
public sealed class ServerBoardSynchronizer : MonoBehaviour
{
    readonly Queue<ReceivedBoardSnapshot> pendingStates = new Queue<ReceivedBoardSnapshot>();
    GameSceneDirector director;
    LegalMovesMessage legalMoves;
    bool applying;
    public bool HasServerState { get; private set; }
    public bool CanSelectMove => !applying && pendingStates.Count == 0 && legalMoves != null &&
        UnityWebSocketClient.Instance && UnityWebSocketClient.Instance.IsConnected;

    void Awake() { director = GetComponent<GameSceneDirector>(); }
    void OnEnable()
    {
        GameEvents.OnStateUpdated += ReceiveState;
        GameEvents.OnLegalMoves += ReceiveLegalMoves;
        GameEvents.OnGameControl += ReceiveControl;
    }
    void OnDisable()
    {
        GameEvents.OnStateUpdated -= ReceiveState;
        GameEvents.OnLegalMoves -= ReceiveLegalMoves;
        GameEvents.OnGameControl -= ReceiveControl;
        bool wasApplying = applying;
        StopAllCoroutines();
        if (wasApplying && director) director.CancelServerStateApplication();
        applying = false;
        pendingStates.Clear();
        legalMoves = null;
    }
    void ReceiveState(StateUpdate message)
    {
        try
        {
            var snapshot = new ReceivedBoardSnapshot(message);
            pendingStates.Enqueue(snapshot);
            HasServerState = true;
            legalMoves = null; // この局面用のlegal_movesが届くまで入力しない。
        }
        catch (Exception exception)
        {
            Debug.LogWarning($"[ServerBoard] 局面を反映できません: {exception.Message}", this);
        }
    }
    void ReceiveLegalMoves(LegalMovesMessage message) { legalMoves = message; }
    void ReceiveControl(GameControl message)
    {
        if (message.command != "reset" && message.command != "resign" && message.command != "start") return;
        legalMoves = null;
        pendingStates.Clear();
    }
    void Update()
    {
        if (!director || !director.CanApplyServerState || applying || pendingStates.Count == 0) return;
        HasServerState = true;
        applying = true;
        StartCoroutine(ApplyNext());
    }
    IEnumerator ApplyNext()
    {
        try { yield return director.ApplyServerState(pendingStates.Dequeue()); }
        finally { applying = false; }
    }

    public List<Vector2Int> GetDestinations(UnitController unit)
    {
        var destinations = new List<Vector2Int>();
        if (!CanSelectMove || legalMoves.moves == null) return destinations;
        foreach (LegalMove move in legalMoves.moves)
        {
            if (!MatchesUnit(move, unit) || !ReceivedBoardSnapshot.IsBoardSquare(move.to)) continue;
            Vector2Int destination = GameSceneDirector.FromShogiSquare(move.to);
            if (!destinations.Contains(destination)) destinations.Add(destination);
        }
        return destinations;
    }
    static bool MatchesUnit(LegalMove move, UnitController unit)
    {
        if (move == null) return false;
        if (unit.FieldStatus == FieldStatus.Captured)
            return move.from == "00" && move.drop_species == GameSceneDirector.GetDropSpecies(unit.OldUnitType);
        return move.from == GameSceneDirector.ToShogiSquare(unit.Pos) && string.IsNullOrEmpty(move.drop_species);
    }
    public bool IsLegal(LegalMove candidate, bool promote)
    {
        return CanSelectMove && candidate != null && legalMoves.moves != null && legalMoves.moves.Exists(move =>
            move != null && move.from == candidate.from && move.to == candidate.to && move.promote == promote &&
            (move.drop_species ?? "") == (candidate.drop_species ?? ""));
    }
    public void RequestMove(LegalMove move)
    {
        if (!IsLegal(move, move.promote)) return;
        legalMoves = null;
        // 送信時には盤を変更せず、Pythonが確定したstate_updateだけを反映する。
        UnityWebSocketClient.Instance.SendMoveRequest(move);
    }
}
