using System;
using KokoroShogi.Core;
using UnityEngine;

namespace KokoroShogi.Net
{
    public static class MessageRouter
    {
        public static bool TryRoute(string json, out string error)
        {
            error = null;
            if (string.IsNullOrWhiteSpace(json)) { error = "空のメッセージです。"; return false; }

            MessageHeader header;
            try { header = JsonUtility.FromJson<MessageHeader>(json); }
            catch (Exception exception) { error = $"JSONを解析できません: {exception.Message}"; return false; }

            if (header == null || string.IsNullOrEmpty(header.type))
            {
                error = "必須フィールド type がありません。";
                return false;
            }
            if (header.schema != Protocol.SchemaVersion)
                Debug.LogWarning($"[Kokoro Protocol] schema不一致: expected={Protocol.SchemaVersion}, actual={header.schema ?? "null"}");

            try
            {
                switch (header.type)
                {
                    case "state_update":
                        StateUpdate state = JsonUtility.FromJson<StateUpdate>(json);
                        if (string.IsNullOrEmpty(state.sfen)) { error = "state_update.sfen がありません。"; return false; }
                        GameEvents.Raise(state); return true;
                    case "legal_moves": GameEvents.Raise(JsonUtility.FromJson<LegalMovesMessage>(json)); return true;
                    case "career": GameEvents.Raise(JsonUtility.FromJson<CareerMessage>(json)); return true;
                    case "game_control": GameEvents.Raise(JsonUtility.FromJson<GameControl>(json)); return true;
                    default: error = $"未知のメッセージ種別です: {header.type}"; return false;
                }
            }
            catch (Exception exception) { error = $"{header.type} の解析に失敗しました: {exception.Message}"; return false; }
        }

        public static string SerializeMoveRequest(LegalMove move) => JsonUtility.ToJson(new MoveRequest { schema = Protocol.SchemaVersion, type = "move_request", move = move });
        public static string SerializeGameControl(string command)
        {
            if (command != "start" && command != "resign" && command != "reset") throw new ArgumentException("commandは start/resign/reset のいずれかです。", nameof(command));
            return JsonUtility.ToJson(new GameControl { schema = Protocol.SchemaVersion, type = "game_control", command = command });
        }
    }
}
