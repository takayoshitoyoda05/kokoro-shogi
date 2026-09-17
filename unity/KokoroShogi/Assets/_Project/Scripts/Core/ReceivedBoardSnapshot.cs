using System;
using System.Collections.Generic;
using KokoroShogi.Net;

namespace KokoroShogi.Core
{
    // Unityのオブジェクトを変更する前に、局面全体を検証する。
    public sealed class ReceivedBoardSnapshot
    {
        public StateUpdate Message { get; }
        public int PlayerToMove { get; }

        public ReceivedBoardSnapshot(StateUpdate message)
        {
            if (message == null || message.pieces == null || message.pieces.Count == 0 || message.pieces.Count > 40)
                throw new FormatException("state_updateには盤上と持ち駒の全piecesが必要です。");
            string[] fields = (message.sfen ?? "").Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (fields.Length != 4 || (fields[1] != "b" && fields[1] != "w") ||
                !int.TryParse(fields[3], out int moveNumber) || moveNumber < 1 || message.ply < 0)
                throw new FormatException("SFENの手番・手数が不正です。");

            var board = new BoardState();
            board.LoadSfen(message.sfen);
            var ids = new HashSet<string>();
            var occupied = new HashSet<string>();
            var hands = new Dictionary<string, int>();
            foreach (PieceInfo piece in message.pieces)
            {
                if (piece == null || string.IsNullOrEmpty(piece.piece_id) || !ids.Add(piece.piece_id) ||
                    (piece.owner != 0 && piece.owner != 1))
                    throw new FormatException("駒のID重複・欠落、またはownerが不正です。");
                BaseTypeNumber(piece.species); // 未対応の駒種を事前に拒否する。
                if (piece.square == "hand")
                {
                    if (IsPromoted(piece.species) || piece.species == "OU")
                        throw new FormatException("持ち駒の種類が不正です。");
                    string key = piece.owner + piece.species;
                    hands.TryGetValue(key, out int count);
                    hands[key] = count + 1;
                }
                else if (!IsBoardSquare(piece.square) || !occupied.Add(piece.square) ||
                    !board.Squares.TryGetValue(piece.square, out BoardState.SfenPiece expected) ||
                    expected.Owner != piece.owner || expected.Species != piece.species)
                    throw new FormatException($"piecesとSFENの盤面が一致しません: {piece.piece_id}");
            }
            if (occupied.Count != board.Squares.Count)
                throw new FormatException("piecesに盤上の駒が不足しています。");
            ValidateHands(fields[2], hands);
            Message = message;
            PlayerToMove = fields[1] == "b" ? 0 : 1;
        }

        public static bool IsBoardSquare(string square) => square != null && square.Length == 2 &&
            square[0] >= '1' && square[0] <= '9' && square[1] >= '1' && square[1] <= '9';

        // prefabUnits / UnitType の生駒の並びに対応する。
        public static int BaseTypeNumber(string species)
        {
            switch (species)
            {
                case "FU": case "TO": return 1;
                case "KA": case "UM": return 2;
                case "HI": case "RY": return 3;
                case "KY": case "NY": return 4;
                case "KE": case "NK": return 5;
                case "GI": case "NG": return 6;
                case "KI": return 7;
                case "OU": return 8;
                default: throw new FormatException($"未知の駒種: {species}");
            }
        }

        public static bool IsPromoted(string species) => species == "TO" || species == "UM" ||
            species == "RY" || species == "NY" || species == "NK" || species == "NG";

        static void ValidateHands(string field, Dictionary<string, int> expected)
        {
            var actual = new Dictionary<string, int>();
            int count = 0;
            if (field != "-") foreach (char token in field)
            {
                if (token >= '0' && token <= '9')
                {
                    count = checked(count * 10 + token - '0');
                    if (count == 0 || count > 40) throw new FormatException("SFEN持ち駒数が不正です。");
                    continue;
                }
                string species;
                switch (char.ToUpperInvariant(token))
                {
                    case 'P': species = "FU"; break; case 'L': species = "KY"; break;
                    case 'N': species = "KE"; break; case 'S': species = "GI"; break;
                    case 'G': species = "KI"; break; case 'B': species = "KA"; break;
                    case 'R': species = "HI"; break;
                    default: throw new FormatException("SFEN持ち駒文字が不正です。");
                }
                string key = (char.IsUpper(token) ? 0 : 1) + species;
                if (actual.ContainsKey(key)) throw new FormatException("SFEN持ち駒の種類が重複しています。");
                actual.Add(key, count == 0 ? 1 : count);
                count = 0;
            }
            if (count != 0 || actual.Count != expected.Count) throw new FormatException("SFENとpiecesの持ち駒が一致しません。");
            foreach (var pair in expected)
                if (!actual.TryGetValue(pair.Key, out int value) || value != pair.Value)
                    throw new FormatException("SFENとpiecesの持ち駒数が一致しません。");
        }
    }
}
