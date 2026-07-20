using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using KokoroShogi.Net;
using UnityEngine;
using UnityEngine.Events;

/// <summary>
/// Python WebSocketサーバーへ接続するUnityクライアントです。
///
/// ・Python側がサーバー、Unity側がクライアント
/// ・シーンをまたいで生存
/// ・AutoConnectがONなら登場時に自動接続
/// ・EchoSignalがONなら受信後にsuccess_echoを返信
/// ・外部ボタンからConnect()、Disconnect()、Send(string)を呼び出し可能
///
/// </summary>
public sealed class UnityWebSocketClient : MonoBehaviour
{
    [Serializable]
    public sealed class StringEvent : UnityEvent<string>
    {
    }

    public static UnityWebSocketClient Instance { get; private set; }

    [Header("Server")]
    [Tooltip("PythonサーバーのIPアドレス。同じPCなら127.0.0.1です。")]
    [SerializeField] private string serverIp = "127.0.0.1";

    [Tooltip("Pythonサーバーのポート番号です。")]
    [SerializeField] private int serverPort = 8765;

    [Header("Options")]
    [Tooltip("ONなら、このObjectが最初に登場したときに接続を試みます。")]
    [SerializeField, InspectorName("AutoConnect")]
    private bool autoConnect = true;

    [Tooltip("ONなら、受信直後にsuccess_echoをPythonへ送信します。")]
    [SerializeField, InspectorName("EchoSignal")]
    private bool echoSignal = false;

    [Tooltip("success_echoに含める、受信内容の先頭文字数です。")]
    [SerializeField, Min(1)] private int echoPreviewLength = 20;

    [Header("Events (Optional)")]
    [SerializeField] private StringEvent onMessageReceived = new StringEvent();
    [SerializeField] private StringEvent onConnectionStatusChanged = new StringEvent();

    public bool IsConnected =>
        webSocket != null && webSocket.State == WebSocketState.Open;

    public string ServerIp => serverIp;
    public int ServerPort => serverPort;

    private ClientWebSocket webSocket;
    private CancellationTokenSource connectionCancellation;
    private Task receiveTask;

    // ClientWebSocketで複数のSendAsyncを同時実行しないために使用します。
    private readonly SemaphoreSlim sendLock = new SemaphoreSlim(1, 1);

    // ConnectとDisconnectが同時実行されないようにします。
    private readonly SemaphoreSlim connectionLock = new SemaphoreSlim(1, 1);

    // 受信処理などの別スレッドからUnityのログとイベントを安全に実行します。
    private readonly ConcurrentQueue<Action> mainThreadActions =
        new ConcurrentQueue<Action>();

    private bool applicationIsQuitting;

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        Instance = this;
        DontDestroyOnLoad(gameObject);
    }

    private void Start()
    {
        if (autoConnect)
        {
            Connect();
        }
    }

    private void Update()
    {
        while (mainThreadActions.TryDequeue(out Action action))
        {
            try
            {
                action?.Invoke();
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
            }
        }
    }

    /// <summary>
    /// Inspectorに設定されたIPとポートで接続します。
    /// Unity ButtonのOnClickから直接呼び出せます。
    /// </summary>
    public void Connect()
    {
        _ = ConnectAsync();
    }

    /// <summary>
    /// InputFieldなどからIPとポートを設定して接続するときに使用します。
    /// </summary>
    public void ConfigureAndConnect(string ip, string portText)
    {
        if (!int.TryParse(portText, out int port))
        {
            EnqueueError($"[WebSocket CONNECT FAILED] ポート番号が不正です: {portText}");
            return;
        }

        if (!SetConnectionSettings(ip, port))
        {
            return;
        }

        Connect();
    }

    /// <summary>
    /// 接続先を設定します。接続中の場合は次回接続から反映されます。
    /// </summary>
    public bool SetConnectionSettings(string ip, int port)
    {
        if (string.IsNullOrWhiteSpace(ip))
        {
            EnqueueError("[WebSocket] IPアドレスが空です。");
            return false;
        }

        if (port < 1 || port > 65535)
        {
            EnqueueError($"[WebSocket] ポート番号は1～65535で指定してください: {port}");
            return false;
        }

        serverIp = ip.Trim();
        serverPort = port;
        return true;
    }

    /// <summary>
    /// このクライアントで使用する唯一の公開送信関数です。
    /// EchoSignalによるsuccess_echoも、このSend()を利用します。
    ///
    /// JSON文字列を渡した場合はそのまま送信します。
    /// 通常テキストを渡した場合は {"text":"内容"} に変換します。
    /// </summary>
    public void Send(string data)
    {
        string payload = CreatePayload(data);

        EnqueueLog($"[WebSocket SEND REQUEST] {payload}");
        _ = SendPayloadAsync(payload);
    }

    /// <summary>Schema 1.0準拠の人間の指し手を送信します。</summary>
    public void SendMoveRequest(LegalMove move)
    {
        if (move == null) throw new ArgumentNullException(nameof(move));
        Send(MessageRouter.SerializeMoveRequest(move));
    }

    /// <summary>Schema 1.0準拠の対局制御を送信します。</summary>
    public void SendGameControl(string command)
    {
        Send(MessageRouter.SerializeGameControl(command));
    }

    /// <summary>
    /// 接続を解除します。
    /// Unity ButtonのOnClickから直接呼び出せます。
    /// </summary>
    public void Disconnect()
    {
        _ = DisconnectAsync();
    }

    private async Task ConnectAsync()
    {
        await connectionLock.WaitAsync();

        try
        {
            if (applicationIsQuitting)
            {
                return;
            }

            if (IsConnected)
            {
                EnqueueStatus("既に接続されています。");
                return;
            }

            await DisposeConnectionAsync();

            string uriText = BuildWebSocketUri(serverIp, serverPort);
            Uri uri;

            try
            {
                uri = new Uri(uriText);
            }
            catch (Exception exception)
            {
                EnqueueError(
                    $"[WebSocket CONNECT FAILED] 接続先が不正です: {uriText}\n" +
                    exception.Message
                );
                return;
            }

            ClientWebSocket newSocket = new ClientWebSocket();
            CancellationTokenSource newCancellation =
                new CancellationTokenSource();

            webSocket = newSocket;
            connectionCancellation = newCancellation;

            EnqueueStatus($"接続中: {uri}");

            try
            {
                await newSocket.ConnectAsync(uri, newCancellation.Token);
            }
            catch (Exception exception)
            {
                EnqueueError(
                    $"[WebSocket CONNECT FAILED] {uri}\n{exception.Message}"
                );
                await DisposeConnectionAsync();
                return;
            }

            if (
                newSocket != webSocket ||
                newSocket.State != WebSocketState.Open
            )
            {
                EnqueueError(
                    "[WebSocket CONNECT FAILED] 接続完了前に接続状態が変更されました。"
                );
                return;
            }

            EnqueueStatus($"接続成功: {uri}");
            receiveTask = ReceiveLoopAsync(
                newSocket,
                newCancellation.Token
            );
        }
        finally
        {
            connectionLock.Release();
        }
    }

    private async Task SendPayloadAsync(string payload)
    {
        ClientWebSocket socket = webSocket;
        CancellationTokenSource cancellation = connectionCancellation;

        if (socket == null || socket.State != WebSocketState.Open)
        {
            EnqueueError(
                $"[WebSocket SEND FAILED] 未接続のため送信できません: {payload}"
            );
            return;
        }

        bool lockTaken = false;

        try
        {
            CancellationToken token = cancellation != null
                ? cancellation.Token
                : CancellationToken.None;

            await sendLock.WaitAsync(token);
            lockTaken = true;

            // ロック待機中に切断または再接続された場合は送信しません。
            if (socket != webSocket || socket.State != WebSocketState.Open)
            {
                EnqueueError(
                    $"[WebSocket SEND FAILED] 接続状態が変更されました: {payload}"
                );
                return;
            }

            byte[] bytes = Encoding.UTF8.GetBytes(payload);

            await socket.SendAsync(
                new ArraySegment<byte>(bytes),
                WebSocketMessageType.Text,
                true,
                token
            );

            EnqueueLog($"[WebSocket SEND SUCCESS] {payload}");
        }
        catch (OperationCanceledException)
        {
            EnqueueError(
                $"[WebSocket SEND FAILED] 送信がキャンセルされました: {payload}"
            );
        }
        catch (Exception exception)
        {
            EnqueueError(
                $"[WebSocket SEND FAILED] {exception.Message}: {payload}"
            );
        }
        finally
        {
            if (lockTaken)
            {
                sendLock.Release();
            }
        }
    }

    private async Task ReceiveLoopAsync(
        ClientWebSocket socket,
        CancellationToken cancellationToken
    )
    {
        byte[] buffer = new byte[8192];

        try
        {
            while (
                !cancellationToken.IsCancellationRequested &&
                socket == webSocket &&
                socket.State == WebSocketState.Open
            )
            {
                using (MemoryStream messageBuffer = new MemoryStream())
                {
                    WebSocketReceiveResult result;

                    do
                    {
                        result = await socket.ReceiveAsync(
                            new ArraySegment<byte>(buffer),
                            cancellationToken
                        );

                        if (result.MessageType == WebSocketMessageType.Close)
                        {
                            EnqueueStatus(
                                "サーバーから切断要求を受信しました。"
                            );
                            await CloseSocketAsync(socket);
                            return;
                        }

                        if (result.Count > 0)
                        {
                            messageBuffer.Write(buffer, 0, result.Count);
                        }
                    }
                    while (!result.EndOfMessage);

                    if (result.MessageType != WebSocketMessageType.Text)
                    {
                        EnqueueLog(
                            "[WebSocket RECEIVE] テキスト以外のデータを受信しました。"
                        );
                        continue;
                    }

                    string receivedText = Encoding.UTF8.GetString(
                        messageBuffer.ToArray()
                    );

                    EnqueueReceivedMessage(receivedText);

                    if (echoSignal)
                    {
                        string preview = GetLeadingCharacters(
                            receivedText,
                            echoPreviewLength
                        );

                        string echoMessage =
                            $"success_echo[{preview} ・・・]";

                        EnqueueLog(
                            $"[WebSocket ECHO REQUEST] {echoMessage}"
                        );

                        // success_echoも通常送信と同じ公開Send()を使用します。
                        Send(echoMessage);
                    }
                    else
                    {
                        EnqueueLog(
                            "[WebSocket ECHO] EchoSignalがOFFのため返信しません。"
                        );
                    }
                }
            }
        }
        catch (OperationCanceledException)
        {
            if (!applicationIsQuitting)
            {
                EnqueueStatus("受信処理を停止しました。");
            }
        }
        catch (Exception exception)
        {
            EnqueueError(
                $"[WebSocket RECEIVE FAILED] {exception.Message}"
            );
        }
        finally
        {
            if (socket == webSocket && !applicationIsQuitting)
            {
                EnqueueStatus("WebSocket接続が終了しました。");
            }
        }
    }

    private async Task DisconnectAsync()
    {
        await connectionLock.WaitAsync();

        try
        {
            if (webSocket == null)
            {
                EnqueueStatus("既に切断されています。");
                return;
            }

            EnqueueStatus("切断処理を開始します。");
            await DisposeConnectionAsync();
            EnqueueStatus("切断しました。");
        }
        finally
        {
            connectionLock.Release();
        }
    }

    private async Task DisposeConnectionAsync()
    {
        ClientWebSocket socket = webSocket;
        CancellationTokenSource cancellation = connectionCancellation;
        Task currentReceiveTask = receiveTask;

        webSocket = null;
        connectionCancellation = null;
        receiveTask = null;

        if (socket == null && cancellation == null)
        {
            return;
        }

        try
        {
            if (socket != null && socket.State == WebSocketState.Open)
            {
                using (CancellationTokenSource timeout =
                       new CancellationTokenSource(TimeSpan.FromSeconds(2)))
                {
                    try
                    {
                        await socket.CloseAsync(
                            WebSocketCloseStatus.NormalClosure,
                            "Unity client disconnect",
                            timeout.Token
                        );
                    }
                    catch
                    {
                        // Closeハンドシェイクに失敗しても破棄処理を続けます。
                    }
                }
            }
        }
        finally
        {
            try
            {
                cancellation?.Cancel();
            }
            catch
            {
                // 破棄処理を続けます。
            }

            if (currentReceiveTask != null)
            {
                try
                {
                    await currentReceiveTask;
                }
                catch
                {
                    // ReceiveLoop側でログを表示します。
                }
            }

            socket?.Dispose();
            cancellation?.Dispose();
        }
    }

    private static async Task CloseSocketAsync(ClientWebSocket socket)
    {
        if (
            socket.State != WebSocketState.Open &&
            socket.State != WebSocketState.CloseReceived
        )
        {
            return;
        }

        using (CancellationTokenSource timeout =
               new CancellationTokenSource(TimeSpan.FromSeconds(2)))
        {
            try
            {
                await socket.CloseAsync(
                    WebSocketCloseStatus.NormalClosure,
                    "Close acknowledged by Unity",
                    timeout.Token
                );
            }
            catch
            {
                // 切断処理中のため無視します。
            }
        }
    }

    private static string CreatePayload(string data)
    {
        string value = data ?? string.Empty;
        string trimmed = value.Trim();

        if (LooksLikeJson(trimmed))
        {
            return trimmed;
        }

        return "{\"text\":\"" + EscapeJsonString(value) + "\"}";
    }

    private static bool LooksLikeJson(string value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return false;
        }

        return
            (value[0] == '{' && value[value.Length - 1] == '}') ||
            (value[0] == '[' && value[value.Length - 1] == ']');
    }

    private static string EscapeJsonString(string value)
    {
        StringBuilder builder = new StringBuilder(value.Length + 16);

        foreach (char character in value)
        {
            switch (character)
            {
                case '\"':
                    builder.Append("\\\"");
                    break;
                case '\\':
                    builder.Append("\\\\");
                    break;
                case '\b':
                    builder.Append("\\b");
                    break;
                case '\f':
                    builder.Append("\\f");
                    break;
                case '\n':
                    builder.Append("\\n");
                    break;
                case '\r':
                    builder.Append("\\r");
                    break;
                case '\t':
                    builder.Append("\\t");
                    break;
                default:
                    if (character < 0x20)
                    {
                        builder.Append("\\u");
                        builder.Append(((int)character).ToString("x4"));
                    }
                    else
                    {
                        builder.Append(character);
                    }
                    break;
            }
        }

        return builder.ToString();
    }

    private static string GetLeadingCharacters(string value, int length)
    {
        if (string.IsNullOrEmpty(value))
        {
            return string.Empty;
        }

        int safeLength = Math.Max(1, length);

        return value.Length <= safeLength
            ? value
            : value.Substring(0, safeLength);
    }

    private static string BuildWebSocketUri(string ip, int port)
    {
        string host = (ip ?? string.Empty).Trim();

        if (host.StartsWith("ws://", StringComparison.OrdinalIgnoreCase))
        {
            host = host.Substring("ws://".Length);
        }
        else if (host.StartsWith("wss://", StringComparison.OrdinalIgnoreCase))
        {
            host = host.Substring("wss://".Length);
        }

        host = host.TrimEnd('/');
        return $"ws://{host}:{port}/";
    }

    private void EnqueueReceivedMessage(string message)
    {
        mainThreadActions.Enqueue(() =>
        {
            Debug.Log($"[WebSocket RECEIVE] {message}");
            onMessageReceived?.Invoke(message);

            if (!MessageRouter.TryRoute(message, out string error))
            {
                Debug.LogWarning($"[Kokoro Protocol] {error}");
            }
        });
    }

    private void EnqueueStatus(string status)
    {
        mainThreadActions.Enqueue(() =>
        {
            Debug.Log($"[WebSocket] {status}");
            onConnectionStatusChanged?.Invoke(status);
        });
    }

    private void EnqueueLog(string message)
    {
        mainThreadActions.Enqueue(() => Debug.Log(message));
    }

    private void EnqueueError(string message)
    {
        mainThreadActions.Enqueue(() => Debug.LogError(message));
    }

    private void OnApplicationQuit()
    {
        applicationIsQuitting = true;

        try
        {
            connectionCancellation?.Cancel();
        }
        catch
        {
            // アプリ終了時のため無視します。
        }
    }

    private void OnDestroy()
    {
        if (Instance != this)
        {
            return;
        }

        Instance = null;
        applicationIsQuitting = true;

        try
        {
            connectionCancellation?.Cancel();
        }
        catch
        {
            // Object破棄時のため無視します。
        }

        webSocket?.Dispose();
        connectionCancellation?.Dispose();
        sendLock.Dispose();
        connectionLock.Dispose();
    }
}
