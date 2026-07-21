using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// UnityWebSocketClientの機能をキーボードから確認するスクリプトです。
///
/// C：サーバーへ接続
/// D：接続解除
/// S：テストメッセージ送信
/// I：IPアドレス変更
/// P：ポート番号変更
///
/// 入力中：
/// Enter：確定
/// Backspace：1文字削除
/// Escape：キャンセル
/// </summary>
public sealed class keyboard_action : MonoBehaviour
{
    private enum InputMode
    {
        None,
        IpAddress,
        Port
    }

    [Header("WebSocket Reference")]
    [Tooltip("UnityWebSocketClientがアタッチされたObjectを指定します。")]
    [SerializeField]
    private UnityWebSocketClient webSocketClient;

    [Header("Display")]
    [Tooltip("IPやPortの入力内容をGameビュー左上に表示します。")]
    [SerializeField]
    private bool showInputOverlay = true;

    private InputMode inputMode = InputMode.None;
    private string inputBuffer = string.Empty;
    private int sendCount = 0;

    private void Start()
    {
        ResolveWebSocketClient();
        ShowHelp();
    }

    private void Update()
    {
        ResolveWebSocketClient();

        Keyboard keyboard = Keyboard.current;

        if (keyboard == null)
        {
            return;
        }

        if (inputMode == InputMode.None)
        {
            HandleCommandKeys(keyboard);
        }
        else
        {
            HandleValueInput(keyboard);
        }
    }

    /// <summary>
    /// UnityWebSocketClientの参照を取得します。
    /// Inspectorで未指定の場合はInstanceを使用します。
    /// </summary>
    private void ResolveWebSocketClient()
    {
        if (webSocketClient == null)
        {
            webSocketClient = UnityWebSocketClient.Instance;
        }
    }

    /// <summary>
    /// UnityWebSocketClientが利用可能か確認します。
    /// </summary>
    private bool EnsureWebSocketClient()
    {
        ResolveWebSocketClient();

        if (webSocketClient != null)
        {
            return true;
        }

        Debug.LogError(
            "[keyboard_action] UnityWebSocketClientが見つかりません。\n" +
            "InspectorのWeb Socket Clientへ、" +
            "UnityWebSocketClientがアタッチされたObjectを設定してください。"
        );

        return false;
    }

    /// <summary>
    /// 通常状態のキー入力を処理します。
    /// </summary>
    private void HandleCommandKeys(Keyboard keyboard)
    {
        if (keyboard[Key.C].wasPressedThisFrame)
        {
            Connect();
        }

        if (keyboard[Key.D].wasPressedThisFrame)
        {
            Disconnect();
        }

        if (keyboard[Key.S].wasPressedThisFrame)
        {
            SendTestMessage();
        }

        if (keyboard[Key.I].wasPressedThisFrame)
        {
            BeginIpInput();
        }

        if (keyboard[Key.P].wasPressedThisFrame)
        {
            BeginPortInput();
        }
    }

    /// <summary>
    /// IPまたはPortの入力中のキー処理です。
    /// </summary>
    private void HandleValueInput(Keyboard keyboard)
    {
        if (keyboard[Key.Escape].wasPressedThisFrame)
        {
            CancelInput();
            return;
        }

        if (keyboard[Key.Enter].wasPressedThisFrame ||
            keyboard[Key.NumpadEnter].wasPressedThisFrame)
        {
            CompleteInput();
            return;
        }

        if (keyboard[Key.Backspace].wasPressedThisFrame)
        {
            RemoveLastCharacter();
            return;
        }

        AppendNumberKeys(keyboard);

        if (inputMode == InputMode.IpAddress)
        {
            if (keyboard[Key.Period].wasPressedThisFrame ||
                keyboard[Key.NumpadPeriod].wasPressedThisFrame)
            {
                inputBuffer += ".";
            }
        }
    }

    /// <summary>
    /// 入力文字列の最後の1文字を削除します。
    /// </summary>
    private void RemoveLastCharacter()
    {
        if (inputBuffer.Length == 0)
        {
            return;
        }

        inputBuffer = inputBuffer.Substring(
            0,
            inputBuffer.Length - 1
        );
    }

    /// <summary>
    /// 数字キーの入力を確認します。
    /// 上部数字キーとテンキーの両方に対応しています。
    /// </summary>
    private void AppendNumberKeys(Keyboard keyboard)
    {
        AppendIfPressed(keyboard, Key.Digit0, '0');
        AppendIfPressed(keyboard, Key.Digit1, '1');
        AppendIfPressed(keyboard, Key.Digit2, '2');
        AppendIfPressed(keyboard, Key.Digit3, '3');
        AppendIfPressed(keyboard, Key.Digit4, '4');
        AppendIfPressed(keyboard, Key.Digit5, '5');
        AppendIfPressed(keyboard, Key.Digit6, '6');
        AppendIfPressed(keyboard, Key.Digit7, '7');
        AppendIfPressed(keyboard, Key.Digit8, '8');
        AppendIfPressed(keyboard, Key.Digit9, '9');

        AppendIfPressed(keyboard, Key.Numpad0, '0');
        AppendIfPressed(keyboard, Key.Numpad1, '1');
        AppendIfPressed(keyboard, Key.Numpad2, '2');
        AppendIfPressed(keyboard, Key.Numpad3, '3');
        AppendIfPressed(keyboard, Key.Numpad4, '4');
        AppendIfPressed(keyboard, Key.Numpad5, '5');
        AppendIfPressed(keyboard, Key.Numpad6, '6');
        AppendIfPressed(keyboard, Key.Numpad7, '7');
        AppendIfPressed(keyboard, Key.Numpad8, '8');
        AppendIfPressed(keyboard, Key.Numpad9, '9');
    }

    private void AppendIfPressed(
        Keyboard keyboard,
        Key key,
        char character
    )
    {
        if (keyboard[key].wasPressedThisFrame)
        {
            inputBuffer += character;
        }
    }

    /// <summary>
    /// 設定されているIPとPortへ接続します。
    /// CキーまたはUnityのButtonから呼び出せます。
    /// </summary>
    public void Connect()
    {
        if (!EnsureWebSocketClient())
        {
            return;
        }

        Debug.Log(
            "[keyboard_action] 接続を開始します。\n" +
            $"接続先: ws://{webSocketClient.ServerIp}:" +
            $"{webSocketClient.ServerPort}"
        );

        webSocketClient.Connect();
    }

    /// <summary>
    /// WebSocket接続を解除します。
    /// DキーまたはUnityのButtonから呼び出せます。
    /// </summary>
    public void Disconnect()
    {
        if (!EnsureWebSocketClient())
        {
            return;
        }

        Debug.Log("[keyboard_action] 接続解除を開始します。");

        webSocketClient.Disconnect();
    }

    /// <summary>
    /// テストメッセージを送信します。
    /// </summary>
    public void SendTestMessage()
    {
        if (!EnsureWebSocketClient())
        {
            return;
        }

        sendCount++;

        string message =
            $"テストメッセージ{sendCount}回目";

        Debug.Log(
            $"[keyboard_action] テスト送信: {message}"
        );

        webSocketClient.Send(message);
    }

    /// <summary>
    /// IPアドレスの入力を開始します。
    /// </summary>
    public void BeginIpInput()
    {
        if (!EnsureWebSocketClient())
        {
            return;
        }

        inputMode = InputMode.IpAddress;
        inputBuffer = string.Empty;

        Debug.Log(
            "[keyboard_action] IP入力開始\n" +
            $"現在のIP: {webSocketClient.ServerIp}\n" +
            "数字とピリオドを入力してください。\n" +
            "Enter: 確定 / Escape: キャンセル"
        );
    }

    /// <summary>
    /// ポート番号の入力を開始します。
    /// </summary>
    public void BeginPortInput()
    {
        if (!EnsureWebSocketClient())
        {
            return;
        }

        inputMode = InputMode.Port;
        inputBuffer = string.Empty;

        Debug.Log(
            "[keyboard_action] Port入力開始\n" +
            $"現在のPort: {webSocketClient.ServerPort}\n" +
            "1～65535の数字を入力してください。\n" +
            "Enter: 確定 / Escape: キャンセル"
        );
    }

    /// <summary>
    /// 入力されたIPまたはPortを確定します。
    /// </summary>
    private void CompleteInput()
    {
        if (!EnsureWebSocketClient())
        {
            ResetInput();
            return;
        }

        if (inputMode == InputMode.IpAddress)
        {
            ApplyIpAddress(inputBuffer);
        }
        else if (inputMode == InputMode.Port)
        {
            ApplyPort(inputBuffer);
        }

        ResetInput();
    }

    /// <summary>
    /// IPアドレスを検証して変更します。
    /// </summary>
    private void ApplyIpAddress(string newIp)
    {
        if (!IsValidIpv4Address(newIp))
        {
            Debug.LogWarning(
                $"[failed]IP_change 入力形式が不正です: {newIp}"
            );

            return;
        }

        string oldIp = webSocketClient.ServerIp;

        bool changed =
            webSocketClient.SetConnectionSettings(
                newIp,
                webSocketClient.ServerPort
            );

        if (!changed)
        {
            Debug.LogWarning(
                $"[failed]IP_change {oldIp} > {newIp}"
            );

            return;
        }

        Debug.Log(
            $"[success]IP_change {oldIp} > {newIp}"
        );

        LogReconnectNoticeWhenConnected();
    }

    /// <summary>
    /// ポート番号を検証して変更します。
    /// </summary>
    private void ApplyPort(string portText)
    {
        if (!IsOnlyAsciiDigits(portText))
        {
            Debug.LogWarning(
                $"[failed]Port_change 数字以外が入力されています: " +
                $"{portText}"
            );

            return;
        }

        if (!int.TryParse(portText, out int newPort))
        {
            Debug.LogWarning(
                $"[failed]Port_change 数値へ変換できません: " +
                $"{portText}"
            );

            return;
        }

        if (newPort < 1 || newPort > 65535)
        {
            Debug.LogWarning(
                $"[failed]Port_change 範囲外です: {newPort}\n" +
                "Portは1～65535で指定してください。"
            );

            return;
        }

        int oldPort = webSocketClient.ServerPort;

        bool changed =
            webSocketClient.SetConnectionSettings(
                webSocketClient.ServerIp,
                newPort
            );

        if (!changed)
        {
            Debug.LogWarning(
                $"[failed]Port_change {oldPort} > {newPort}"
            );

            return;
        }

        Debug.Log(
            $"[success]Port_change {oldPort} > {newPort}"
        );

        LogReconnectNoticeWhenConnected();
    }

    /// <summary>
    /// 接続中に設定を変更した場合の案内を表示します。
    /// </summary>
    private void LogReconnectNoticeWhenConnected()
    {
        if (!webSocketClient.IsConnected)
        {
            return;
        }

        Debug.Log(
            "[keyboard_action] IPまたはPortを変更しましたが、" +
            "現在接続中のサーバーは変更されません。\n" +
            "Dキーで切断してからCキーで再接続してください。"
        );
    }

    /// <summary>
    /// IPまたはPortの入力をキャンセルします。
    /// </summary>
    private void CancelInput()
    {
        string target =
            inputMode == InputMode.IpAddress
                ? "IP"
                : "Port";

        Debug.Log(
            $"[keyboard_action] {target}入力をキャンセルしました。"
        );

        ResetInput();
    }

    private void ResetInput()
    {
        inputMode = InputMode.None;
        inputBuffer = string.Empty;
    }

    /// <summary>
    /// IPv4形式か確認します。
    /// 各数値は0～255の範囲です。
    /// </summary>
    private static bool IsValidIpv4Address(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return false;
        }

        string[] parts = value.Split('.');

        if (parts.Length != 4)
        {
            return false;
        }

        foreach (string part in parts)
        {
            if (string.IsNullOrEmpty(part))
            {
                return false;
            }

            if (part.Length > 3)
            {
                return false;
            }

            if (!IsOnlyAsciiDigits(part))
            {
                return false;
            }

            if (!int.TryParse(part, out int number))
            {
                return false;
            }

            if (number < 0 || number > 255)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// 文字列が半角数字だけか確認します。
    /// </summary>
    private static bool IsOnlyAsciiDigits(string value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return false;
        }

        foreach (char character in value)
        {
            if (character < '0' || character > '9')
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// 操作方法をConsoleへ表示します。
    /// </summary>
    private void ShowHelp()
    {
        Debug.Log(
            "[keyboard_action] 操作方法\n" +
            "C: 接続\n" +
            "D: 接続解除\n" +
            "S: テストメッセージ送信\n" +
            "I: IPアドレス変更\n" +
            "P: Port番号変更\n" +
            "入力中 Enter: 確定\n" +
            "入力中 Escape: キャンセル\n" +
            "入力中 Backspace: 1文字削除"
        );
    }

    /// <summary>
    /// IPまたはPort入力中の内容をGameビューへ表示します。
    /// </summary>
    private void OnGUI()
    {
        if (!showInputOverlay)
        {
            return;
        }

        if (inputMode == InputMode.None)
        {
            return;
        }

        string target =
            inputMode == InputMode.IpAddress
                ? "IPアドレス"
                : "Port番号";

        GUI.Box(
            new Rect(15f, 15f, 620f, 80f),
            $"{target}を入力してください。\n" +
            $"Enterで確定 / Escapeでキャンセル\n" +
            $"> {inputBuffer}_"
        );
    }
}