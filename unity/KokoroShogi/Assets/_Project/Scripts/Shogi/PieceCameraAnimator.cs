using UnityEngine;

/// <summary>成り・駒の獲得時に対象を追い、元のカメラ位置へ戻す。</summary>
[DisallowMultipleComponent]
public sealed class PieceCameraAnimator : MonoBehaviour
{
    [Tooltip("オフにすると駒のカメラ演出を省略します。演出中にオフにした場合は元のカメラ位置へ戻します。")]
    [SerializeField] bool enablePieceCamera = true;
    [SerializeField] Camera targetCamera;
    [Tooltip("注視点からカメラまでのワールド座標のオフセット")]
    [SerializeField] Vector3 closeUpOffset = new Vector3(0f, 7.5f, -7.5f);
    [Tooltip("注視点を中心に横へ回り込む角度。0で正面、負の値で反対側から捉える")]
    [Range(-90f, 90f)] [SerializeField] float closeUpYaw = 25f;
    [Tooltip("X軸の角度補正。0で現在の見下ろし角度、正で上から、負で低い位置から捉える。最終角度は5～85度")]
    [Range(-80f, 80f)] [SerializeField] float closeUpPitch = 0f;
    [SerializeField] Vector3 lookAtOffset = new Vector3(0f, 0.15f, 0f);
    [Min(0.01f)] [SerializeField] float approachDuration = 0.45f;
    [Min(0f)] [SerializeField] float holdDuration = 0.7f;
    [Min(0.01f)] [SerializeField] float returnDuration = 0.6f;
    [Min(0.1f)] [SerializeField] float closeUpOrthographicSize = 2f;

    Camera activeCamera;
    Transform subject;
    Vector3 lastFocusPoint;
    Vector3 homePosition, startPosition;
    Quaternion homeRotation, startRotation;
    float homeSize, startSize, elapsed;
    bool waitForAction;
    enum Phase { Idle, Approach, AwaitAction, Hold, Return }
    Phase phase;

    public bool IsAvailable => enablePieceCamera && isActiveAndEnabled;
    public bool IsPlaying => phase != Phase.Idle;
    public bool HasReachedFocus => phase == Phase.AwaitAction || phase == Phase.Hold || phase == Phase.Return;

    /// <summary>連続呼び出しでも最初のカメラ位置を帰還先として保持する。</summary>
    public void FocusOn(Transform piece)
    {
        if (!IsAvailable || !piece) return;
        FocusAt(piece.position);
        subject = piece;
    }

    /// <summary>移動先へ先に接近する。waitForAction指定時はCompleteActionまで帰還しない。</summary>
    public void FocusAt(Vector3 position, bool waitForAction = false)
    {
        if (!IsAvailable) return;
        if (!IsPlaying)
        {
            activeCamera = targetCamera ? targetCamera : Camera.main;
            if (!activeCamera) return;
            homePosition = activeCamera.transform.position;
            homeRotation = activeCamera.transform.rotation;
            homeSize = activeCamera.orthographicSize;
        }
        if (!activeCamera) { Cancel(); return; }
        subject = null;
        this.waitForAction = waitForAction;
        lastFocusPoint = position + lookAtOffset;
        BeginPhase(Phase.Approach);
    }

    public void CompleteAction()
    {
        if (phase == Phase.AwaitAction && activeCamera) BeginPhase(Phase.Hold);
    }

    void BeginPhase(Phase next)
    {
        startPosition = activeCamera.transform.position;
        startRotation = activeCamera.transform.rotation;
        startSize = activeCamera.orthographicSize;
        elapsed = 0f;
        phase = next;
    }

    void LateUpdate()
    {
        if (!enablePieceCamera) { Cancel(); return; }
        if (!IsPlaying) return;
        if (!activeCamera) { Cancel(); return; }
        elapsed += Time.unscaledDeltaTime;
        if (subject) lastFocusPoint = subject.position + lookAtOffset;

        if (phase == Phase.Return)
        {
            float t = Mathf.Clamp01(elapsed / Mathf.Max(0.01f, returnDuration));
            ApplyPose(homePosition, homeRotation, homeSize, Mathf.SmoothStep(0f, 1f, t));
            if (t >= 1f) Cancel();
            return;
        }

        Vector3 offset = closeUpOffset.sqrMagnitude > 0.01f
            ? closeUpOffset : new Vector3(0f, 3.5f, -3.5f);
        // 注視点を中心に位置も回し、駒を画面中央に捉えたまま斜めの構図にする。
        offset = Quaternion.AngleAxis(closeUpYaw, Vector3.up) * offset;
        float distance = offset.magnitude;
        float elevation = Mathf.Asin(Mathf.Clamp(offset.y / distance, -1f, 1f)) * Mathf.Rad2Deg;
        float pitch = Mathf.Clamp(elevation + closeUpPitch, 5f, 85f) * Mathf.Deg2Rad;
        Vector3 horizontal = Vector3.ProjectOnPlane(offset, Vector3.up);
        if (horizontal.sqrMagnitude < 0.0001f)
            horizontal = Quaternion.AngleAxis(closeUpYaw, Vector3.up) * Vector3.back;
        offset = distance * (horizontal.normalized * Mathf.Cos(pitch) + Vector3.up * Mathf.Sin(pitch));
        Quaternion rotation = Quaternion.LookRotation(-offset,
            Mathf.Abs(Vector3.Dot(offset.normalized, Vector3.up)) > 0.99f
                ? Vector3.forward : Vector3.up);
        float progress = phase == Phase.Hold || phase == Phase.AwaitAction ? 1f
            : Mathf.Clamp01(elapsed / Mathf.Max(0.01f, approachDuration));
        ApplyPose(lastFocusPoint + offset, rotation,
            Mathf.Max(0.1f, closeUpOrthographicSize), Mathf.SmoothStep(0f, 1f, progress));
        if (phase == Phase.Approach && progress >= 1f)
            BeginPhase(waitForAction ? Phase.AwaitAction : Phase.Hold);
        else if (phase == Phase.Hold && elapsed >= holdDuration) BeginPhase(Phase.Return);
    }

    void ApplyPose(Vector3 position, Quaternion rotation, float size, float t)
    {
        activeCamera.transform.SetPositionAndRotation(
            Vector3.Lerp(startPosition, position, t), Quaternion.Slerp(startRotation, rotation, t));
        activeCamera.orthographicSize = Mathf.Lerp(startSize, size, t);
    }

    public void Cancel()
    {
        if (IsPlaying && activeCamera)
        {
            activeCamera.transform.SetPositionAndRotation(homePosition, homeRotation);
            activeCamera.orthographicSize = homeSize;
        }
        phase = Phase.Idle;
        subject = null;
        activeCamera = null;
    }

    void OnDisable() => Cancel();
}
