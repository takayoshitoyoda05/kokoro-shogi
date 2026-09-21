using System.Collections.Generic;
using KokoroShogi.Net;
using UnityEngine;
using UnityEngine.VFX;

public partial class UnitController
{
    const float MinimumFireScale = 13f;
    const float MaximumFireScale = 77f;
    const float DebuffScaleMultiplier = 1.5f;

    readonly List<Transform> fearEffectRoots = new List<Transform>();
    readonly Dictionary<Transform, Vector3> fearEffectBaseScales = new Dictionary<Transform, Vector3>();
    readonly List<Transform> aggressionEffectRoots = new List<Transform>();
    readonly List<ParticleSystem> fearParticles = new List<ParticleSystem>();
    readonly List<Renderer> fearRenderers = new List<Renderer>();
    readonly List<Renderer> aggressionRenderers = new List<Renderer>();
    readonly List<VisualEffect> aggressionEffects = new List<VisualEffect>();
    ParticleSystem.Particle[] liveFearParticles = new ParticleSystem.Particle[0];
    bool effectsCached;

    // state_updateの各駒のmoodを、同じIDの駒へ反映する。
    public void ApplyMood(Mood mood)
    {
        CacheMoodEffects();
        float fear = ClampMoodValue(mood == null ? 0f : mood.fear);
        float aggression = ClampMoodValue(mood == null ? 0f : mood.aggression);
        bool showFear = fear > aggression;
        bool showAggression = aggression > fear;

        // 同値・未受信ではどちらも表示しない。小さい方の粒子は残さない。
        if (!showFear)
        {
            foreach (ParticleSystem particles in fearParticles)
                particles.Stop(false, ParticleSystemStopBehavior.StopEmittingAndClear);
        }
        else
        {
            // 小さいfearでも見えるよう、0～1を維持する平方根で透明度を補正する。
            float fearAlpha = Mathf.Sqrt(fear);
            foreach (ParticleSystem particles in fearParticles)
                ApplyFearAlpha(particles, fearAlpha);
        }
        foreach (Transform root in fearEffectRoots)
        {
            // 初期サイズを基準にすることで、受信のたびに拡大し続けるのを防ぐ。
            if (showFear) root.localScale = fearEffectBaseScales[root] * DebuffScaleMultiplier;
            root.gameObject.SetActive(showFear);
        }
        foreach (Renderer effectRenderer in fearRenderers)
            effectRenderer.enabled = showFear;

        if (showFear)
        {
            foreach (ParticleSystem particles in fearParticles)
            {
                // Debuff配下で非アクティブな中間オブジェクトも再生可能にする。
                for (Transform parent = particles.transform; parent != transform; parent = parent.parent)
                    parent.gameObject.SetActive(true);
                if (!particles.isPlaying) particles.Play(false);
            }
        }

        // 低いaggressionでも炎が駒の周囲に見えるよう、サイズにもルート補正を使う。
        float aggressionStrength = Mathf.Sqrt(aggression);
        float fireScale = Mathf.Lerp(MinimumFireScale, MaximumFireScale, aggressionStrength);
        // GameObjectが有効でもRendererが無効なPrefabは描画されない。
        foreach (Renderer effectRenderer in aggressionRenderers)
            effectRenderer.enabled = showAggression;
        bool startFire = false;
        foreach (Transform root in aggressionEffectRoots)
        {
            if (showAggression)
            {
                startFire |= !root.gameObject.activeInHierarchy;
                root.localScale = Vector3.one * fireScale;
                for (Transform parent = root.parent; parent != transform; parent = parent.parent)
                    parent.gameObject.SetActive(true);
            }
            root.gameObject.SetActive(showAggression);
        }
        foreach (VisualEffect effect in aggressionEffects)
        {
            if (showAggression)
            {
                bool wasDisabled = !effect.enabled || !effect.gameObject.activeInHierarchy;
                for (Transform parent = effect.transform; parent != transform; parent = parent.parent)
                    parent.gameObject.SetActive(true);
                effect.enabled = true;
                // 感情の数値更新だけでは再スタートしない。
                if (startFire || wasDisabled) effect.Play();
            }
            else effect.Stop();
        }
    }

    void CacheMoodEffects()
    {
        if (effectsCached) return;
        // Prefabの非表示オブジェクトも取得。Inspectorでの追加設定は不要。
        foreach (Transform child in GetComponentsInChildren<Transform>(true))
        {
            if (child.name == "Debuff")
            {
                fearEffectRoots.Add(child);
                fearEffectBaseScales.Add(child, child.localScale);
                fearRenderers.AddRange(child.GetComponentsInChildren<Renderer>(true));
                foreach (ParticleSystem particles in child.GetComponentsInChildren<ParticleSystem>(true))
                    if (!fearParticles.Contains(particles)) fearParticles.Add(particles);
            }
            else if (child.name == "vfx_Fire")
            {
                aggressionEffectRoots.Add(child);
                aggressionRenderers.AddRange(child.GetComponentsInChildren<Renderer>(true));
                aggressionEffects.AddRange(child.GetComponentsInChildren<VisualEffect>(true));
            }
        }
        effectsCached = true;
    }

    void ApplyFearAlpha(ParticleSystem particles, float alpha)
    {
        var main = particles.main;
        var startColor = main.startColor;
        // RGBと色のランダム化方式は維持し、全ての開始色のαだけを置き換える。
        switch (startColor.mode)
        {
            case ParticleSystemGradientMode.Color:
                startColor.color = WithAlpha(startColor.color, alpha);
                break;
            case ParticleSystemGradientMode.TwoColors:
                startColor.colorMin = WithAlpha(startColor.colorMin, alpha);
                startColor.colorMax = WithAlpha(startColor.colorMax, alpha);
                break;
            case ParticleSystemGradientMode.TwoGradients:
                startColor.gradientMin = WithAlpha(startColor.gradientMin, alpha);
                startColor.gradientMax = WithAlpha(startColor.gradientMax, alpha);
                break;
            default:
                startColor.gradient = WithAlpha(startColor.gradient, alpha);
                break;
        }
        main.startColor = startColor;

        // StartColorだけでは既に出ている粒子に反映されないため、そちらも更新する。
        int required = particles.particleCount;
        if (liveFearParticles.Length < required)
            liveFearParticles = new ParticleSystem.Particle[required];
        int count = particles.GetParticles(liveFearParticles);
        for (int i = 0; i < count; i++)
            liveFearParticles[i].startColor = WithAlpha(liveFearParticles[i].startColor, alpha);
        particles.SetParticles(liveFearParticles, count);
    }

    static Color WithAlpha(Color color, float alpha)
    {
        // UnityのColorは0～1。画面上の0～255表記では255 * sqrt(fear)に相当する。
        color.a = alpha;
        return color;
    }

    static Gradient WithAlpha(Gradient gradient, float alpha)
    {
        var result = new Gradient();
        result.SetKeys(gradient.colorKeys, new[]
        {
            new GradientAlphaKey(alpha, 0f), new GradientAlphaKey(alpha, 1f)
        });
        result.mode = gradient.mode;
        return result;
    }

    static float ClampMoodValue(float value)
    {
        return float.IsNaN(value) ? 0f : Mathf.Clamp01(value);
    }
}
