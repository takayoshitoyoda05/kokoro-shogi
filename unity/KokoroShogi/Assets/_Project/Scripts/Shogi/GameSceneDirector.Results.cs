using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using TMPro;
using UnityEngine;

// 対局履歴の保持と、CanvasResult内の既存Elementへの表示を担当する。
public partial class GameSceneDirector
{
    [Serializable]
    sealed class FinishedGame
    {
        public long finishedAtTicks;
        public int moveCount;
        public bool humanWon;
        public string reason;
    }

    [Serializable]
    sealed class SavedResultHistory
    {
        public List<FinishedGame> games = new List<FinishedGame>();
    }

    const string ResultHistoryFileName = "result_history.json";
    static readonly List<FinishedGame> resultHistory = new List<FinishedGame>();
    static bool resultHistoryLoaded;
    readonly List<Transform> resultRows = new List<Transform>();
    Transform resultRowTemplate;
    UnityEngine.UI.ScrollRect resultScroll;
    [SerializeField] TMP_Text textAiWinNum;
    [SerializeField] TMP_Text textHumanWinNum;
    bool resultRecordedForCurrentGame;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ResetResultHistory()
    {
        resultHistory.Clear();
        resultHistoryLoaded = false;
    }

    void InitializeResultHistory()
    {
        LoadResultHistory();
        if (!canvasResult) return;
        if (!textAiWinNum || !textHumanWinNum)
        {
            foreach (TMP_Text label in canvasResult.GetComponentsInChildren<TMP_Text>(true))
            {
                if (!textAiWinNum && string.Equals(label.name, "Text_AIWinNum", StringComparison.OrdinalIgnoreCase))
                    textAiWinNum = label;
                else if (!textHumanWinNum && string.Equals(label.name, "Text_HumanWinNum", StringComparison.OrdinalIgnoreCase))
                    textHumanWinNum = label;
            }
        }
        RefreshWinCounts();
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

    void RecordFinishedGameResult(bool humanWon, string reason)
    {
        // 同じ終局通知が再送されても、1対局につき1行だけ追加する。
        if (resultRecordedForCurrentGame) return;
        resultRecordedForCurrentGame = true;
        resultHistory.Insert(0, new FinishedGame
        {
            finishedAtTicks = DateTime.Now.Ticks,
            moveCount = turnCount,
            humanWon = humanWon,
            reason = reason
        });
        SaveResultHistory();
        RefreshResultHistory();
    }

    static string ResultHistoryPath => Path.Combine(Application.persistentDataPath, ResultHistoryFileName);

    void LoadResultHistory()
    {
        if (resultHistoryLoaded) return;
        resultHistoryLoaded = true;
        try
        {
            if (!File.Exists(ResultHistoryPath)) return;
            SavedResultHistory saved = JsonUtility.FromJson<SavedResultHistory>(
                File.ReadAllText(ResultHistoryPath, Encoding.UTF8));
            if (saved?.games == null) return;
            foreach (FinishedGame game in saved.games)
            {
                if (game == null || game.finishedAtTicks < DateTime.MinValue.Ticks ||
                    game.finishedAtTicks > DateTime.MaxValue.Ticks) continue;
                resultHistory.Add(game);
            }
        }
        catch (Exception exception)
        {
            Debug.LogWarning($"戦績を読み込めませんでした: {exception.Message}");
        }
    }

    void SaveResultHistory()
    {
        try
        {
            Directory.CreateDirectory(Application.persistentDataPath);
            var saved = new SavedResultHistory { games = new List<FinishedGame>(resultHistory) };
            File.WriteAllText(ResultHistoryPath, JsonUtility.ToJson(saved), Encoding.UTF8);
        }
        catch (Exception exception)
        {
            Debug.LogWarning($"戦績を保存できませんでした: {exception.Message}", this);
        }
    }

    void RefreshResultHistory()
    {
        RefreshWinCounts();
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
                    label.text = new DateTime(result.finishedAtTicks).ToString("M/d HH:mm");
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
                    label.text = result.moveCount + "手（" + result.reason + "）";
            }
            // サンプルの文字が一瞬表示されないよう、値を設定してから表示する。
            row.gameObject.SetActive(true);
        }
    }

    void RefreshWinCounts()
    {
        int humanWins = 0;
        foreach (FinishedGame result in resultHistory)
            if (result.humanWon) humanWins++;

        if (textHumanWinNum) textHumanWinNum.text = $"{humanWins}勝";
        if (textAiWinNum) textAiWinNum.text = $"{resultHistory.Count - humanWins}勝";
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
