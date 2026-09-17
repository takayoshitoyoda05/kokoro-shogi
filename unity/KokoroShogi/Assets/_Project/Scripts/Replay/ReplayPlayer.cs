using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using KokoroShogi.Net;
using UnityEngine;

namespace KokoroShogi.Replay
{
    public sealed class ReplayPlayer : MonoBehaviour
    {
        [Tooltip("StreamingAssetsからの相対パス。Editorではリポジトリのsample_data/modelも読み込めます。")]
        [SerializeField] private string relativePath = "sample_data/model/game01.jsonl";
        [SerializeField, Min(0.05f)] private float secondsPerMove = 1f;
        [SerializeField, Min(0.25f)] private float speed = 1f;
        private readonly List<string> records = new List<string>();
        private Coroutine playback;
        public int Position { get; private set; } = -1;
        public int Count => records.Count;

        [ContextMenu("Replay/Load")]
        public void Load()
        {
            Stop(); records.Clear(); Position = -1;
            string path = ResolveReplayPath();
            if (path == null) return;
            foreach (string line in File.ReadLines(path)) if (!string.IsNullOrWhiteSpace(line)) records.Add(line);
            Debug.Log($"[Replay] {records.Count}レコードを読み込みました: {path}");
        }

        private string ResolveReplayPath()
        {
            if (string.IsNullOrWhiteSpace(relativePath))
            {
                Debug.LogError("[Replay] Relative Pathにsample_data/model/game01.jsonlなどを設定してください。");
                return null;
            }
            string configuredPath = relativePath.Trim().Replace('\\', '/');
            string streamingPath = Path.Combine(Application.streamingAssetsPath, configuredPath);
            if (File.Exists(streamingPath)) return streamingPath;

#if UNITY_EDITOR
            // 開発時はサンプルのコピー不要。以前の初期値もmodel内の同名ファイルへ対応付ける。
            string samplePath = configuredPath;
            if (samplePath.StartsWith("sample_data/", StringComparison.Ordinal) &&
                samplePath.Substring("sample_data/".Length).IndexOf('/') < 0)
                samplePath = "sample_data/model/" + Path.GetFileName(samplePath);

            if (samplePath.StartsWith("sample_data/model/", StringComparison.Ordinal))
            {
                string repositoryPath = Path.GetFullPath(Path.Combine(Application.dataPath, "../../..", samplePath));
                if (File.Exists(repositoryPath)) return repositoryPath;
                Debug.LogError($"[Replay] JSONLが見つかりません。確認した場所:\n{streamingPath}\n{repositoryPath}");
                return null;
            }
#endif
            Debug.LogError($"[Replay] JSONLが見つかりません: {streamingPath}\nRelative Pathとファイルの配置を確認してください。StreamingAssetsはAssets直下です。");
            return null;
        }

        [ContextMenu("Replay/Play")]
        public void Play() { if (playback == null && Position + 1 < records.Count) playback = StartCoroutine(PlayLoop()); }
        [ContextMenu("Replay/Pause")]
        public void Pause() { if (playback != null) { StopCoroutine(playback); playback = null; } }
        public void Stop() { Pause(); Position = -1; }
        [ContextMenu("Replay/Step Forward")]
        public void StepForward() { Pause(); Apply(Position + 1); }
        [ContextMenu("Replay/Step Back")]
        public void StepBack() { Pause(); Apply(Math.Max(0, Position - 1)); }
        public void SetSpeed(float value) => speed = Mathf.Clamp(value, 0.25f, 4f);
        public void Seek(int index) { Pause(); Apply(index); }

        private IEnumerator PlayLoop()
        {
            while (Position + 1 < records.Count)
            {
                Apply(Position + 1);
                yield return new WaitForSeconds(secondsPerMove / speed);
            }
            playback = null;
        }

        private void Apply(int index)
        {
            if (index < 0 || index >= records.Count) return;
            if (!MessageRouter.TryRoute(records[index], out string error)) { Debug.LogError($"[Replay] {index + 1}行目: {error}"); return; }
            Position = index;
        }
    }
}
