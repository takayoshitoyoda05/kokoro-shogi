using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

// 対局履歴の保持と、CanvasResult内の既存Elementへの表示を担当する。
public partial class GameSceneDirector
{
    sealed class FinishedGame
    {
        public DateTime finishedAt;
        public int moveCount;
        public bool humanWon;
    }

    // 再戦時のシーン再読み込みでも保持する。アプリ終了後の保存は行わない。
    static readonly List<FinishedGame> resultHistory = new List<FinishedGame>();
    readonly List<Transform> resultRows = new List<Transform>();
    Transform resultRowTemplate;
    UnityEngine.UI.ScrollRect resultScroll;
    bool resultRecordedForCurrentGame;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ResetResultHistory()
    {
        resultHistory.Clear();
    }

    void InitializeResultHistory()
    {
        if (!canvasResult) return;
        resultScroll = canvasResult.GetComponentInChildren<UnityEngine.UI.ScrollRect>(true);
        if (!resultScroll || !resultScroll.content) return;

        foreach (Transform child in resultScroll.content)
        {
            if (child.name == "Element" || child.name.StartsWith("Element (", StringComparison.Ordinal))
            {
                // 編集用サンプルは表示せず、先頭の行だけ複製元として参照する。
                child.gameObject.SetActive(false);
                if (!resultRowTemplate) resultRowTemplate = child;
            }
        }
        if (!resultRowTemplate)
        {
            Debug.LogWarning("CanvasResultのContentに履歴表示用のElementがありません。", this);
            return;
        }

        // 履歴件数に合わせてContentを伸ばし、先頭の行を上端に並べる。
        var layout = resultScroll.content.GetComponent<UnityEngine.UI.VerticalLayoutGroup>();
        if (layout)
        {
            layout.childAlignment = TextAnchor.UpperLeft;
            layout.reverseArrangement = false;
        }
        var fitter = resultScroll.content.GetComponent<UnityEngine.UI.ContentSizeFitter>();
        if (fitter) fitter.verticalFit = UnityEngine.UI.ContentSizeFitter.FitMode.PreferredSize;

        RefreshResultHistory();
    }

    void RecordCheckmateResult(bool humanWon)
    {
        // 同じ終局通知が再送されても、1対局につき1行だけ追加する。
        if (resultRecordedForCurrentGame) return;
        resultRecordedForCurrentGame = true;
        resultHistory.Insert(0, new FinishedGame
        {
            finishedAt = DateTime.Now,
            moveCount = turnCount,
            humanWon = humanWon
        });
        RefreshResultHistory();
    }

    void RefreshResultHistory()
    {
        if (!resultScroll || !resultScroll.content || !resultRowTemplate) return;

        // 実際の結果がある分だけ、非表示のサンプルから表示行を生成する。
        while (resultRows.Count < resultHistory.Count)
        {
            Transform row = Instantiate(resultRowTemplate, resultScroll.content, false);
            row.name = "ResultElement (" + resultRows.Count + ")";
            resultRows.Add(row);
        }

        for (int i = 0; i < resultRows.Count; i++)
        {
            Transform row = resultRows[i];
            bool hasResult = i < resultHistory.Count;
            if (!hasResult)
            {
                row.gameObject.SetActive(false);
                continue;
            }
            row.SetSiblingIndex(i);
            FinishedGame result = resultHistory[i];
            foreach (TMP_Text label in row.GetComponentsInChildren<TMP_Text>(true))
            {
                if (label.name.EndsWith("_finTime", StringComparison.Ordinal))
                    label.text = result.finishedAt.ToString("M/d HH:mm");
                else if (label.name.EndsWith("_AIResultArea", StringComparison.Ordinal))
                {
                    label.text = result.humanWon ? "LOSE" : "WIN";
                    label.color = result.humanWon ? Color.blue : Color.red;
                }
                else if (label.name.EndsWith("_humanResultArea", StringComparison.Ordinal))
                {
                    label.text = result.humanWon ? "WIN" : "LOSE";
                    label.color = result.humanWon ? Color.red : Color.blue;
                }
                else if (label.name.EndsWith("_finNumber", StringComparison.Ordinal))
                    label.text = result.moveCount + "手（詰み）";
            }
            // サンプルの文字が一瞬表示されないよう、値を設定してから表示する。
            row.gameObject.SetActive(true);
        }
    }

    void ScrollResultHistoryToTop()
    {
        if (!resultScroll || !resultScroll.content || !resultScroll.gameObject.activeInHierarchy) return;
        Canvas.ForceUpdateCanvases();
        UnityEngine.UI.LayoutRebuilder.ForceRebuildLayoutImmediate(resultScroll.content);
        resultScroll.StopMovement();
        resultScroll.verticalNormalizedPosition = 1f;
    }
}
