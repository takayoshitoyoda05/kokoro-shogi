using TMPro;
using UnityEngine;

public partial class GameSceneDirector
{
    [Header("手番表示の文字マテリアル")]
    [SerializeField, Tooltip("「あなた」の文字だけに適用するTextMeshProのマテリアルプリセット。")]
    Material playerTurnWordMaterial;
    [SerializeField, Tooltip("「AI」の文字だけに適用するTextMeshProのマテリアルプリセット。")]
    Material aiTurnWordMaterial;
    [SerializeField, Tooltip("二重アウトライン用Text。未設定なら子または同じCanvas内から名前で自動取得します。")]
    TMP_Text playerTurnOuterOutlineText;
    [SerializeField, Tooltip("「あなた」の外側アウトラインに適用するTextMeshProのマテリアルプリセット。")]
    Material playerTurnOuterOutlineMaterial;
    [SerializeField, Range(1f, 1.3f), Tooltip("外側アウトラインの各文字を中心から広げる倍率。フォントアトラスの余白が小さくても外側の線を見せられます。")]
    float playerTurnOuterOutlineScale = 1.12f;

    void InitializeTurnOutline()
    {
        if (!textTurnInfo) return;
        if (!playerTurnOuterOutlineText)
        {
            foreach (Transform child in textTurnInfo.transform)
            {
                if (!child.name.StartsWith("TextTurnInfo_back")) continue;
                playerTurnOuterOutlineText = child.GetComponent<TMP_Text>();
                if (playerTurnOuterOutlineText) break;
            }
            if (!playerTurnOuterOutlineText && textTurnInfo.transform.parent)
            {
                foreach (Transform sibling in textTurnInfo.transform.parent)
                {
                    if (!sibling.name.StartsWith("TextTurnInfo_back")) continue;
                    playerTurnOuterOutlineText = sibling.GetComponent<TMP_Text>();
                    if (playerTurnOuterOutlineText) break;
                }
            }
        }
        if (playerTurnOuterOutlineText)
        {
            playerTurnOuterOutlineText.OnPreRenderText -= ExpandPlayerTurnOuterOutline;
            playerTurnOuterOutlineText.OnPreRenderText += ExpandPlayerTurnOuterOutline;
        }
        ClearTurnInfo();
    }

    void SetTurnInfo()
    {
        if (!textTurnInfo) return;
        bool isAiTurn = nowPlayer == aiPlayer;
        string word = isAiTurn ? "AI" : "あなた";
        Material wordMaterial = isAiTurn ? aiTurnWordMaterial : playerTurnWordMaterial;
        textTurnInfo.richText = true;
        textTurnInfo.text = FormatTurnWord(word, wordMaterial) + "の番です";
        if (!playerTurnOuterOutlineText) return;
        playerTurnOuterOutlineText.gameObject.SetActive(!isAiTurn);
        if (isAiTurn) return;

        // 全文でレイアウトし、先頭の「あなた」だけを描画すると手前の文字位置と一致する。
        playerTurnOuterOutlineText.font = textTurnInfo.font;
        playerTurnOuterOutlineText.fontSize = textTurnInfo.fontSize;
        playerTurnOuterOutlineText.fontStyle = textTurnInfo.fontStyle;
        playerTurnOuterOutlineText.characterSpacing = textTurnInfo.characterSpacing;
        playerTurnOuterOutlineText.alignment = textTurnInfo.alignment;
        if (playerTurnOuterOutlineMaterial)
            playerTurnOuterOutlineText.fontSharedMaterial = playerTurnOuterOutlineMaterial;
        playerTurnOuterOutlineText.text = "あなたの番です";
        playerTurnOuterOutlineText.maxVisibleCharacters = 3;
    }

    void ClearTurnInfo()
    {
        if (textTurnInfo) textTurnInfo.text = "";
        if (playerTurnOuterOutlineText) playerTurnOuterOutlineText.gameObject.SetActive(false);
    }

    void ExpandPlayerTurnOuterOutline(TMP_TextInfo textInfo)
    {
        // 文字ごとの中心を基準に広げれば、全文の配置や文字間隔を変えずに外側の線だけ大きくできる。
        int count = Mathf.Min(3, textInfo.characterCount);
        for (int i = 0; i < count; i++)
        {
            TMP_CharacterInfo character = textInfo.characterInfo[i];
            if (!character.isVisible) continue;

            Vector3[] vertices = textInfo.meshInfo[character.materialReferenceIndex].vertices;
            int firstVertex = character.vertexIndex;
            Vector3 center = (vertices[firstVertex] + vertices[firstVertex + 2]) * 0.5f;
            for (int vertex = 0; vertex < 4; vertex++)
                vertices[firstVertex + vertex] = center +
                    (vertices[firstVertex + vertex] - center) * playerTurnOuterOutlineScale;
        }
    }

    static string FormatTurnWord(string word, Material material)
    {
        if (!material) return word;
        // TMPはタグ内の名前からマテリアルを探す。Inspector参照を事前登録する。
        int hash = TMP_TextUtilities.GetSimpleHashCode(material.name.ToUpperInvariant());
        if (!MaterialReferenceManager.TryGetMaterial(hash, out Material registered))
            MaterialReferenceManager.AddFontMaterial(hash, material);
        else if (registered != material)
        {
            Debug.LogWarning($"手番用TMPマテリアルの名前が重複しています: {material.name}");
            return word;
        }
        return $"<material=\"{material.name}\">{word}</material>";
    }
}
