using System;
using System.Collections.Generic;

namespace KokoroShogi.Net
{
    public static class Protocol
    {
        public const string SchemaVersion = "1.1";
    }

    [Serializable] public class MessageHeader { public string schema; public string type; }
    [Serializable] public class LastMove { public string from; public string to; public string piece_id; public bool capture; public bool promote; public bool drop; }
    [Serializable] public class Mood { public float fear; public float aggression; public float valence; }
    [Serializable] public class Desire { public float survive; public float attack; public float promote; public float defend; public float advance; public float redeploy; }
    [Serializable] public class Relation { public string to; public float r; }
    [Serializable] public class PieceInfo { public string piece_id; public string species; public int owner; public string square; public Mood mood; public Desire desire; public float alpha; public List<Relation> relations = new List<Relation>(); }
    [Serializable] public class Proposal { public string piece_id; public string move; public float bid; }
    [Serializable] public class CouncilRound { public int round; public List<Proposal> proposals = new List<Proposal>(); }

    [Serializable]
    public class StateUpdate : MessageHeader
    {
        public int ply;
        public string sfen;
        public LastMove last_move;
        public float eval;
        public List<PieceInfo> pieces = new List<PieceInfo>();
        public List<CouncilRound> council = new List<CouncilRound>();
        public string narration;
    }

    [Serializable] public class LegalMove { public string from; public string to; public bool promote; public string drop_species; }
    [Serializable] public class LegalMovesMessage : MessageHeader { public List<LegalMove> moves = new List<LegalMove>(); }
    [Serializable] public class MoveRequest : MessageHeader { public LegalMove move; }
    [Serializable] public class GameControl : MessageHeader { public string command; }
    [Serializable] public class CareerPiece { public string piece_id; public string species; public int games; public float survival_rate; public int promotions; public int mvp_count; }
    [Serializable] public class CareerMvp { public string piece_id; public float contribution; }
    [Serializable] public class GameResult { public string winner; public int human; public string reason; }
    [Serializable] public class CareerMessage : MessageHeader { public List<CareerPiece> pieces = new List<CareerPiece>(); public CareerMvp last_game_mvp; public GameResult result; }
}
