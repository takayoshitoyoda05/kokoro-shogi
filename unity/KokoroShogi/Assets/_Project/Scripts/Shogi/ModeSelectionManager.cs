using UnityEngine;
using UnityEngine.UI;

[DisallowMultipleComponent]
[RequireComponent(typeof(Button))]
public class ModeSelectionManager : MonoBehaviour
{
    const string StartCommand = "start";

    [SerializeField] Canvas targetCanvas;

    static Canvas activeSelectionCanvas;
    static int canvasHiddenFrame = -1;

    Button button;

    /// <summary>
    /// モード選択画面が閉じており、盤や駒をクリックできるかどうか。
    /// Canvasを閉じたクリックが背後の3Dオブジェクトにも届かないよう、
    /// 閉じた次のフレームから入力を許可する。
    /// </summary>
    public static bool IsWorldInteractionAllowed
    {
        get
        {
            bool isCanvasHidden = !activeSelectionCanvas
                || !activeSelectionCanvas.isActiveAndEnabled;

            return isCanvasHidden && Time.frameCount > canvasHiddenFrame;
        }
    }

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ResetStaticState()
    {
        activeSelectionCanvas = null;
        canvasHiddenFrame = -1;
    }

    void Awake()
    {
        button = GetComponent<Button>();

        if (!targetCanvas)
        {
            targetCanvas = GetComponentInParent<Canvas>();
        }

        if (targetCanvas)
        {
            activeSelectionCanvas = targetCanvas;
        }
    }

    void OnEnable()
    {
        if (!button)
        {
            button = GetComponent<Button>();
        }

        button.onClick.AddListener(SelectMode);

        if (targetCanvas)
        {
            activeSelectionCanvas = targetCanvas;
        }
    }

    void OnDisable()
    {
        if (button)
        {
            button.onClick.RemoveListener(SelectMode);
        }
    }

    void SelectMode()
    {
        GameSceneDirector director = FindFirstObjectByType<GameSceneDirector>();
        if (director) director.BeginSelectedGame();
        UnityWebSocketClient webSocketClient = UnityWebSocketClient.Instance;

        if (webSocketClient)
        {
            webSocketClient.SendGameControl(StartCommand);
        }
        else
        {
            Debug.LogError(
                "ModeSelectionManager: UnityWebSocketClientが見つからないため、対局開始を送信できません。",
                this
            );
        }

        HideSelectionCanvas();
    }

    public void HideSelectionCanvas()
    {
        if (!targetCanvas)
        {
            Debug.LogError("ModeSelectionManager: 非表示にするCanvasが見つかりません。", this);
            return;
        }

        canvasHiddenFrame = Time.frameCount;
        targetCanvas.gameObject.SetActive(false);
    }
}
