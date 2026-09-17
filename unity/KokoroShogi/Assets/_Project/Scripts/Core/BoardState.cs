using System;
using System.Collections.Generic;

namespace KokoroShogi.Core
{
    public sealed class BoardState
    {
        public readonly struct SfenPiece
        {
            public readonly string Species;
            public readonly int Owner;
            public SfenPiece(string species, int owner) { Species = species; Owner = owner; }
        }

        private readonly Dictionary<string, SfenPiece> squares = new Dictionary<string, SfenPiece>();
        public IReadOnlyDictionary<string, SfenPiece> Squares => squares;

        public void LoadSfen(string sfen)
        {
            if (string.IsNullOrWhiteSpace(sfen)) throw new ArgumentException("SFENが空です。", nameof(sfen));
            string[] fields = sfen.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            string[] ranks = fields[0].Split('/');
            if (ranks.Length != 9) throw new FormatException("SFEN盤面は9段必要です。");
            squares.Clear();
            for (int rank = 0; rank < 9; rank++)
            {
                int fileIndex = 0;
                bool promoted = false;
                foreach (char token in ranks[rank])
                {
                    if (token >= '1' && token <= '9')
                    {
                        if (promoted) throw new FormatException("成り記号の直後に空マスがあります。");
                        fileIndex += token - '0';
                        continue;
                    }
                    if (token == '+')
                    {
                        if (promoted) throw new FormatException("成り記号が重複しています。");
                        promoted = true;
                        continue;
                    }
                    if (fileIndex >= 9) throw new FormatException("SFENの筋数が9を超えています。");
                    int owner = char.IsUpper(token) ? 0 : 1;
                    string species = Species(char.ToUpperInvariant(token), promoted);
                    string square = $"{9 - fileIndex}{rank + 1}";
                    squares.Add(square, new SfenPiece(species, owner));
                    fileIndex++;
                    promoted = false;
                }
                if (fileIndex != 9 || promoted) throw new FormatException($"SFEN {rank + 1}段目が不正です。");
            }
        }

        private static string Species(char value, bool promoted)
        {
            string baseName;
            switch (value)
            {
                case 'P': baseName = "FU"; break; case 'L': baseName = "KY"; break;
                case 'N': baseName = "KE"; break; case 'S': baseName = "GI"; break;
                case 'G': baseName = "KI"; break; case 'B': baseName = "KA"; break;
                case 'R': baseName = "HI"; break; case 'K': baseName = "OU"; break;
                default: throw new FormatException($"未知のSFEN駒文字です: {value}");
            }
            if (!promoted) return baseName;
            switch (baseName)
            {
                case "FU": return "TO"; case "KY": return "NY"; case "KE": return "NK";
                case "GI": return "NG"; case "KA": return "UM"; case "HI": return "RY";
                default: throw new FormatException($"成れない駒に+があります: {baseName}");
            }
        }
    }
}
