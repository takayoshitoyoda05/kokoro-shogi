using System.Collections.Generic;
using DG.Tweening;
using KokoroShogi.Net;
using UnityEngine;

public partial class GameSceneDirector
{
    [SerializeField] UnityEngine.UI.Slider sliderFill1PValue;
    [SerializeField] UnityEngine.UI.Slider sliderFill2PValue;
    GameObject iconUp1P;
    GameObject iconDown1P;
    GameObject iconUp2P;
    GameObject iconDown2P;
    [SerializeField, Min(0f), Tooltip("形勢バーが新しい比率へ移動する秒数。0なら即時反映。")]
    float valenceBarAnimationSeconds = 0.4f;
    [SerializeField, Range(0f, 0.5f), Tooltip("1P・2Pそれぞれに常に表示する最低割合。0.1なら両者ともバーの10%以上を表示します。")]
    float minimumValenceBarFill = 0.05f;
    Tween valenceBarTween;
    float displayedFirstPlayerRatio = 0.5f;

    void InitializeValenceBar()
    {
        // 既存シーンの非表示Canvasも含めて参照する。別シーンのSliderは取得しない。
        if (!sliderFill1PValue || !sliderFill2PValue)
        {
            foreach (GameObject root in gameObject.scene.GetRootGameObjects())
            foreach (UnityEngine.UI.Slider slider in root.GetComponentsInChildren<UnityEngine.UI.Slider>(true))
            {
                if (!sliderFill1PValue && slider.name == "Slider_Fill_1PValue") sliderFill1PValue = slider;
                if (!sliderFill2PValue && slider.name == "Slider_Fill_2PValue") sliderFill2PValue = slider;
            }
        }
        ConfigureValenceSlider(sliderFill1PValue);
        ConfigureValenceSlider(sliderFill2PValue);
        InitializeValenceIcons();
        if (!sliderFill1PValue || !sliderFill2PValue)
            Debug.LogWarning("形勢バーのSlider_Fill_1PValue / Slider_Fill_2PValueが見つかりません。", this);
        SetValenceBarRatio(0.5f);
    }

    static void ConfigureValenceSlider(UnityEngine.UI.Slider slider)
    {
        if (!slider) return;
        slider.minValue = 0f;
        slider.maxValue = 1f;
        slider.wholeNumbers = false;
        // 受信値の表示専用なので、クリックやキー操作で値が変わらないようにする。
        slider.interactable = false;
    }

    void UpdateValenceBar(List<PieceInfo> pieces)
    {
        SetValenceBarRatio(KokoroShogi.Core.TeamValenceBalance.CalculateFirstPlayerRatio(pieces), true);
    }

    void SetValenceBarRatio(float firstPlayerRatio, bool animate = false)
    {
        float minimum = Mathf.Clamp(minimumValenceBarFill, 0f, 0.5f);
        firstPlayerRatio = Mathf.Clamp(firstPlayerRatio, minimum, 1f - minimum);
        // 更新途中に次の局面が来たら、その時点の表示位置から新しい目標へ向かう。
        StopValenceBarTween();
        if (!animate || valenceBarAnimationSeconds <= 0f || !isActiveAndEnabled ||
            Mathf.Approximately(displayedFirstPlayerRatio, firstPlayerRatio))
        {
            ApplyDisplayedValenceBarRatio(firstPlayerRatio);
            return;
        }
        // 1本のTweenから両方を更新し、アニメーション中も合計を1に保つ。
        valenceBarTween = DOTween.To(() => displayedFirstPlayerRatio,
                ApplyDisplayedValenceBarRatio, firstPlayerRatio, valenceBarAnimationSeconds)
            .SetEase(Ease.OutCubic)
            .SetUpdate(true)
            .SetLink(gameObject, LinkBehaviour.KillOnDestroy)
            .OnKill(() => valenceBarTween = null);
    }

    void ApplyDisplayedValenceBarRatio(float firstPlayerRatio)
    {
        float minimum = Mathf.Clamp(minimumValenceBarFill, 0f, 0.5f);
        firstPlayerRatio = Mathf.Clamp(firstPlayerRatio, minimum, 1f - minimum);
        displayedFirstPlayerRatio = firstPlayerRatio;
        if (sliderFill1PValue) sliderFill1PValue.SetValueWithoutNotify(firstPlayerRatio);
        if (sliderFill2PValue) sliderFill2PValue.SetValueWithoutNotify(1f - firstPlayerRatio);
        UpdateValenceIcons(firstPlayerRatio);
    }

    void InitializeValenceIcons()
    {
        foreach (GameObject root in gameObject.scene.GetRootGameObjects())
        foreach (Transform child in root.GetComponentsInChildren<Transform>(true))
        {
            if (child.name == "IconGauge_1P")
            {
                iconUp1P = FindIcon(child, "Icon_Up");
                iconDown1P = FindIcon(child, "Icon_Down");
            }
            else if (child.name == "IconGauge_2P")
            {
                iconUp2P = FindIcon(child, "Icon_Up");
                iconDown2P = FindIcon(child, "Icon_Down");
            }
        }
    }

    static GameObject FindIcon(Transform gauge, string name)
    {
        Transform icon = gauge.Find(name);
        return icon ? icon.gameObject : null;
    }

    void UpdateValenceIcons(float firstPlayerRatio)
    {
        float secondPlayerRatio = 1f - firstPlayerRatio;
        bool firstPlayerAhead = firstPlayerRatio > secondPlayerRatio &&
            !Mathf.Approximately(firstPlayerRatio, secondPlayerRatio);
        bool secondPlayerAhead = secondPlayerRatio > firstPlayerRatio &&
            !Mathf.Approximately(firstPlayerRatio, secondPlayerRatio);
        if (iconUp1P) iconUp1P.SetActive(firstPlayerAhead);
        if (iconDown1P) iconDown1P.SetActive(secondPlayerAhead);
        if (iconUp2P) iconUp2P.SetActive(secondPlayerAhead);
        if (iconDown2P) iconDown2P.SetActive(firstPlayerAhead);
    }

    void StopValenceBarTween()
    {
        if (valenceBarTween != null) valenceBarTween.Kill();
        valenceBarTween = null;
    }
}
