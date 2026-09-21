using System;
using System.Collections.Generic;
using KokoroShogi.Net;

namespace KokoroShogi.Core
{
    // 通信データだけから比率を計算し、Unityの表示処理から独立して検証できるようにする。
    public static class TeamValenceBalance
    {
        public static float CalculateFirstPlayerRatio(IEnumerable<PieceInfo> pieces)
        {
            double firstTotal = 0;
            double secondTotal = 0;
            if (pieces != null)
            {
                foreach (PieceInfo piece in pieces)
                {
                    if (piece == null || piece.mood == null) continue;
                    float valence = piece.mood.valence;
                    if (float.IsNaN(valence) || float.IsInfinity(valence)) continue;
                    // 持ち駒も含め、現在のownerで合計する。エフェクトの補正は使用しない。
                    if (piece.owner == 0) firstTotal += valence;
                    else if (piece.owner == 1) secondTotal += valence;
                }
            }
            // 負の長さは表示できないため、各駒ではなく陣営の合計を0以上にする。
            firstTotal = Math.Max(0, firstTotal);
            secondTotal = Math.Max(0, secondTotal);
            double total = firstTotal + secondTotal;
            return total > 0 ? (float)(firstTotal / total) : 0.5f;
        }
    }
}
