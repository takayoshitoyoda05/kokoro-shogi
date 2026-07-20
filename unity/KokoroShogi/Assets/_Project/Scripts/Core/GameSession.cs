using System;
using KokoroShogi.Net;

namespace KokoroShogi.Core
{
    public enum GameSessionState { Waiting, HumanTurn, AiThinking }

    public sealed class GameSession : IDisposable
    {
        public GameSessionState State { get; private set; } = GameSessionState.Waiting;
        public LegalMovesMessage LegalMoves { get; private set; }
        public event Action<GameSessionState> OnStateChanged;

        public GameSession()
        {
            GameEvents.OnLegalMoves += HandleLegalMoves;
            GameEvents.OnStateUpdated += HandleStateUpdate;
            GameEvents.OnGameControl += HandleControl;
        }

        public bool TryBeginMove(LegalMove move, out string error)
        {
            error = null;
            if (State != GameSessionState.HumanTurn) { error = "現在は人間手番ではありません。"; return false; }
            if (!Contains(move)) { error = "AIから受信した合法手一覧に含まれません。"; return false; }
            SetState(GameSessionState.AiThinking);
            return true;
        }

        private bool Contains(LegalMove candidate)
        {
            if (candidate == null || LegalMoves?.moves == null) return false;
            return LegalMoves.moves.Exists(move => move.from == candidate.from && move.to == candidate.to && move.promote == candidate.promote && move.drop_species == candidate.drop_species);
        }

        private void HandleLegalMoves(LegalMovesMessage value) { LegalMoves = value; SetState(GameSessionState.HumanTurn); }
        private void HandleStateUpdate(StateUpdate _) { if (State == GameSessionState.AiThinking) SetState(GameSessionState.HumanTurn); }
        private void HandleControl(GameControl value) { if (value.command == "resign" || value.command == "reset") { LegalMoves = null; SetState(GameSessionState.Waiting); } }
        private void SetState(GameSessionState value) { if (State == value) return; State = value; OnStateChanged?.Invoke(value); }

        public void Dispose()
        {
            GameEvents.OnLegalMoves -= HandleLegalMoves;
            GameEvents.OnStateUpdated -= HandleStateUpdate;
            GameEvents.OnGameControl -= HandleControl;
        }
    }
}
