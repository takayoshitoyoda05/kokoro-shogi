using System;
using KokoroShogi.Net;

namespace KokoroShogi.Core
{
    public static class GameEvents
    {
        public static event Action<StateUpdate> OnStateUpdated;
        public static event Action<LastMove> OnMovePlayed;
        public static event Action<string> OnPieceCaptured;
        public static event Action<string> OnPiecePromoted;
        public static event Action<string> OnPieceDropped;
        public static event Action<string> OnNarration;
        public static event Action<LegalMovesMessage> OnLegalMoves;
        public static event Action<CareerMessage> OnCareer;
        public static event Action<GameControl> OnGameControl;

        public static void Raise(StateUpdate value)
        {
            OnStateUpdated?.Invoke(value);
            if (value.last_move != null)
            {
                OnMovePlayed?.Invoke(value.last_move);
                if (value.last_move.capture) OnPieceCaptured?.Invoke(value.last_move.piece_id);
                if (value.last_move.promote) OnPiecePromoted?.Invoke(value.last_move.piece_id);
                if (value.last_move.drop) OnPieceDropped?.Invoke(value.last_move.piece_id);
            }
            if (!string.IsNullOrEmpty(value.narration)) OnNarration?.Invoke(value.narration);
        }

        public static void Raise(LegalMovesMessage value) => OnLegalMoves?.Invoke(value);
        public static void Raise(CareerMessage value) => OnCareer?.Invoke(value);
        public static void Raise(GameControl value) => OnGameControl?.Invoke(value);
    }
}
