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
        [SerializeField] private string relativePath = "sample_data/game01.jsonl";
        [SerializeField, Min(0.05f)] private float secondsPerMove = 1f;
        [SerializeField, Min(0.25f)] private float speed = 1f;
        private readonly List<string> records = new List<string>();
        private Coroutine playback;
        public int Position { get; private set; } = -1;
        public int Count => records.Count;

        public void Load()
        {
            Stop(); records.Clear(); Position = -1;
            string path = Path.Combine(Application.streamingAssetsPath, relativePath);
            if (!File.Exists(path)) { Debug.LogError($"JSONLが見つかりません: {path}"); return; }
            foreach (string line in File.ReadLines(path)) if (!string.IsNullOrWhiteSpace(line)) records.Add(line);
            Debug.Log($"[Replay] {records.Count}レコードを読み込みました: {path}");
        }

        public void Play() { if (playback == null && Position + 1 < records.Count) playback = StartCoroutine(PlayLoop()); }
        public void Pause() { if (playback != null) { StopCoroutine(playback); playback = null; } }
        public void Stop() { Pause(); Position = -1; }
        public void StepForward() { Pause(); Apply(Position + 1); }
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
