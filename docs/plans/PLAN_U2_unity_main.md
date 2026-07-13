# 個人計画書 U2: Unity本体担当 (完全詳細版 / 10週)

> 親文書: `TEAM_PLAN.md` / データ: U1の `GameEvents` 経由 / モデル契約: `naming.md` (**週1にB1/B2と三者で作成**)
> この文書だけで作業を始められるよう、ノード構成・コード・設定値まで全て記載する。

## 役割

画面に映るものすべて: 感情シェーダ、駒アニメ、関係線VFX、オーラ、
会議吹き出し、実況字幕、HUD、転生/成り演出、音。
**JSONは触らない**。U1の `GameEvents` (C#イベント) を購読して数値を受け取るだけ。
Unityバージョンはチーム標準 **6000.3.8f1 (LTS)** に固定 (U1のプロジェクトを
そのまま開く。自分で別バージョンを入れない)。

## 優先順位 (10週の生命線。上から作る)

```
【必須】 1.感情シェーダ 2.移動/捕獲アニメ 3.形勢バー+凡例 4.関係線
        5.会議吹き出し 6.実況字幕 7.転生/成り演出 8.駒音
        9.キャリアUI (最小構成: 駒の戦績一覧表 + 対局MVP表示。§5.6)
【任意】 10.駒クリック詳細パネル (欲求レーダー) 11.BGM/ポスプロ盛り
```
リプレイ操作ボタンはU1担当。迷ったら必須リストの上に戻る。

---

## 1. 受け取るデータ (これだけ知っていれば作れる)

U1の `GameEvents` を購読する。自分でファイルや通信を読まない。

```csharp
void OnEnable() {
    GameEvents.OnStateUpdated  += HandleState;    // 毎手: 全駒のmood/alpha/relations等
    GameEvents.OnMovePlayed    += HandleMove;     // 移動 (from,to,piece_id)
    GameEvents.OnPieceCaptured += HandleCapture;  // 転生演出のトリガ
    GameEvents.OnPiecePromoted += HandlePromote;
    GameEvents.OnNarration     += HandleNarration;
}
void OnDisable() { /* 必ず -= で解除 (リークとNull例外防止) */ }
```

| フィールド | 範囲 | 使い道 |
|---|---|---|
| mood.fear | 0-1 | 青ざめ+震え |
| mood.aggression | 0-1 | 赤熱発光 |
| mood.valence | -1〜+1 | 彩度 (機嫌) |
| alpha | 0-1 | 発言力オーラ |
| relations[].r | 0-1 | 絆の線の太さ/輝度 |
| council[round].proposals[].bid | 実数 | 吹き出しの主張の強さ |
| narration | 文字列 | 字幕 (表示するだけ。生成はAI側) |

---

## 2. 感情シェーダ (Shader Graph 完全レシピ)

### 2.1 作成手順
1. Project右クリック → Create → Shader Graph → URP → **Lit Shader Graph**、
   名前 `SG_KomaEmotion`
2. Graph Inspector → Properties に追加 (Reference名は完全一致で):

| Property | 型 | Reference | Default |
|---|---|---|---|
| Fear | Float (0-1 Slider) | `_Fear` | 0 |
| Aggression | Float (0-1 Slider) | `_Aggr` | 0 |
| Valence | Float (-1..1 Slider) | `_Valence` | 0 |
| BaseMap | Texture2D | `_BaseMap` | (駒テクスチャ) |
| FearColor | Color | `_FearColor` | #6FA8DC (青ざめ) |
| AggrColor | Color (HDR) | `_AggrColor` | #FF3B30 強度2 |

3. ノード構成 (Fragment):
```
BaseMap ─ Sample Texture 2D ─┐
FearColor ───────────────────┴ Lerp (T = _Fear × 0.6) → Saturation
                                (Saturation量 = 1 + _Valence × 0.3) → Base Color
_AggrColor × _Aggr → Emission
```
4. ノード構成 (Vertex = 震え):
```
Time ─ Sine (Time × 40) ─ Multiply (_Fear × 0.002) ─ Multiply (法線方向)
  → Position に加算 (Object Space)
```
   ※ 40=震えの速さ、0.002=振幅 (メートル)。駒サイズ0.05に対して控えめに

### 2.2 C#から値を流す (MaterialPropertyBlock — 重要)
`renderer.material.SetFloat` は**マテリアルが駒ごとに複製されて重くなる**ので、
40駒あるこのプロジェクトでは必ず PropertyBlock を使う:
```csharp
static readonly int FearId = Shader.PropertyToID("_Fear");
static readonly int AggrId = Shader.PropertyToID("_Aggr");
MaterialPropertyBlock mpb = new();

public void ApplyMood(Renderer r, Mood m) {
    r.GetPropertyBlock(mpb);
    mpb.SetFloat(FearId, m.fear);
    mpb.SetFloat(AggrId, m.aggression);
    r.SetPropertyBlock(mpb);
}
```

### 2.3 シェイプキー (B1の駒が来たら)
FBXのシェイプキーはUnityでは BlendShape。
```csharp
var smr = piece.GetComponentInChildren<SkinnedMeshRenderer>();
int fearIdx = smr.sharedMesh.GetBlendShapeIndex("mood_fear"); // naming.md準拠
smr.SetBlendShapeWeight(fearIdx, mood.fear * 100f);           // 0-100スケール注意
```

---

## 3. アニメーション (DOTween レシピ)

導入: Asset Store から DOTween (無料) → Tools → DOTween Utility Panel → Setup。

```csharp
// 移動 (0.35秒、放物線ぽく)
piece.transform.DOMove(targetPos, 0.35f).SetEase(Ease.OutQuad);
piece.transform.DOJump(targetPos, jumpPower: 0.03f, numJumps: 1, 0.35f); // 跳ね版

// 捕獲: 縮んで消える (転生演出の前半)
captured.transform.DOScale(0f, 0.25f).SetEase(Ease.InBack)
        .OnComplete(() => SpawnSoul(captured));  // 魂パーティクルへ

// 成り: 半回転して裏面へ
piece.transform.DORotate(new Vector3(0,0,180), 0.4f, RotateMode.LocalAxisAdd);
```
連続手の事故対策: 新しい手が来たら `piece.transform.DOKill(true);` で
前のTweenを完了状態で殺してから次を再生。

---

## 4. 関係線とオーラ

### 4.1 関係線 (LineRenderer)
- 空GameObject `RelationLines` にプール (最大60本のLineRendererを事前生成、使い回し)
- 設定: Width = `0.002 + r * 0.006`、Material = URP/Unlit + 加算合成 (Additive)、
  Color = シアン系で `alpha = r`
- 更新は**0.2秒に1回** (毎フレーム更新は不要で重い):
  `InvokeRepeating(nameof(UpdateLines), 0f, 0.2f);`
- r < 0.3 の関係は描かない (画面が線だらけになるのを防ぐ足切り)

### 4.2 発言力オーラ
- 各駒の足元に Particle System (Shape: Circle, Emission: rate = alpha × 20,
  Start Size 0.01, Start Color HDR金色, Simulation Space: World)
- または簡易版: 足元に発光デカール (Quad + Additiveマテリアル, scale = alpha)

---

## 5. UI (UGUI 完全手順)

### 5.1 日本語フォント設定 (週1に必ず)
1. Google Fonts から **Noto Sans JP** をダウンロード (OFLライセンス、明記)
2. Window → TextMeshPro → Font Asset Creator:
   - Source Font: NotoSansJP-Regular
   - Sampling Point Size: Auto / Atlas: **4096×4096**
   - Character Set: **Characters from File** → 常用漢字+かな+記号を入れた
     txt (「TextMeshPro 日本語 文字セット」で検索して入手) を指定
   - Generate → Save。以降全TextはこのFont Assetを使う

### 5.2 形勢バー
- Screen Space Canvas 上部に Slider (操作不可) を置き、Fill を赤/青2色
- `eval` (-1..+1) → `slider.DOValue((eval+1)/2, 0.5f)` でヌルッと追従

### 5.3 会議吹き出し (World Space Canvas)
- 吹き出しPrefab: World Space Canvas (scale 0.001) + Image(9-slice) + TMP Text
- 表示ロジック: `council` の各roundを 0.6秒間隔で逐次再生。
  round内は **bid上位3件のみ**。テキスト例 `▲2四飛! ` + bid値で文字サイズを
  `24 + bid*4` に (主張が強いほどデカい)
- 駒の頭上に出す: `bubble.position = piece.position + Vector3.up * 0.08f;`
  カメラ正対: `bubble.forward = cam.forward;` (LateUpdate)

### 5.4 実況字幕 (タイプライター)
```csharp
IEnumerator TypeText(TMP_Text label, string s) {
    label.maxVisibleCharacters = 0; label.text = s;
    for (int i = 0; i <= s.Length; i++) {
        label.maxVisibleCharacters = i;
        yield return new WaitForSeconds(0.03f);
    }
}
```
過去2件をログ表示 (VerticalLayoutGroup)。narrationが来たら再生。

### 5.5 凡例
画面隅に固定: 「青=恐怖 / 赤=闘志 / 金の輪=発言力 / 光の線=絆」。
**シェーダの色と完全に同じ色コード**を使う (観客が対応を取れることが全て)。

### 5.6 キャリアUI (週8着手・週9完成、最小構成)
- データ: Aが週9頭に配信するcareerメッセージ (またはStreamingAssetsの
  career.json):
```jsonc
{"type": "career", "pieces": [
  {"piece_id": "P77_gen0_0003", "species": "FU", "games": 120,
   "survival_rate": 0.42, "promotions": 18, "mvp_count": 3}]}
```
- 画面: Screen Space Canvas のパネル1枚。ScrollView + 行Prefab
  (駒種アイコン | piece_id | 対局数 | 生存率 | 成り | MVP)。
  ソートはMVP降順固定でよい (ソート機能は任意)
- 対局終了時: 「本局のMVP: ▲飛車 (貢献度 +2.3)」のトースト表示を1つ
- **これ以上作らない** (レーダー・履歴グラフは任意タスク11に置く)

---

## 6. 転生・成り演出 (デモのハイライト、週7)

転生シーケンス (捕獲イベントで開始):
1. 駒が縮む (DOScale 0, 0.25s)
2. 魂パーティクル生成 (小さな光球: Particle System 1個 + Trail)
3. 魂が相手駒台へ DOMove (1.0s, Ease.InOutSine, 弧を描くなら DOPath)
4. 駒台で着地フラッシュ → 持ち駒表示 (小さい駒を駒台に置く)
5. 打たれたら (OnPieceDropped) 逆再生: 光球→盤上で駒がScale 0→1

素材 (魂のメッシュ・駒の裏面) はB1が週6に納品予定。

---

## 7. サウンド (週8)

- 駒音: 「将棋 駒音 フリー効果音」で入手 (例: 効果音ラボ等)。
  **`Assets/_Project/Audio/LICENSES.md` に出典と利用条件を必ず記録**
- 実装: `AudioSource.PlayOneShot(komaOto)` を OnMovePlayed で。
  音量ゆらぎ `source.pitch = Random.Range(0.95f, 1.05f)` で自然に
- 会議ポップ音 (吹き出し出現)、転生の音 (キラーン系) も同様

---

## 8. パフォーマンス (常時ルール)

- 目標 **60fps** (Game viewのStatsで常時確認、週5にProfilerで本計測)
- Window → Analysis → Profiler → CPU/Renderingを見る。SetPass Calls が
  100超えたらマテリアル共有を疑う
- ルール: MaterialPropertyBlock必須 / 関係線更新0.2秒 / パーティクルは
  Max Particles 50以下/個 / 演出プレハブはプール (Instantiate連打禁止)
- 週9: Post-processing (Volume → Bloom, Intensity 0.5〜1) は最後に足す。
  先に足すと重さの原因切り分けができなくなる

---

## 9. 週次計画 (10週)

| 週 | やること | 完了条件 |
|---|---|---|
| 1 | Shader Graphチュートリアル1本 / SG_KomaEmotion試作 (Cube) / フォント設定 / naming.md合意 / UIワイヤー(手描き可) | Cubeが青ざめて震える |
| 2 | 感情シェーダ完成 (Cube) / MaterialPropertyBlock適用クラス / B1先行2駒で検証 | 歩と王に感情が乗る |
| 3 | 移動/捕獲/成りアニメ (GameEvents購読) / 形勢バー+凡例 | リプレイで駒が滑らかに動く |
| 4 | ★統合点①: 統合検証・調整 | 演出+HUD同期のリプレイ |
| 5 | 関係線プール / オーラ / **Profiler初計測** | 40駒+線で60fps |
| 6 | 実データで感情レンジ調整 (下記※) / 会議吹き出けv0 | 実データが「読める」 |
| 7 | ★統合点② + ユーザーテスト / 転生/成り演出・実況字幕 | デモのハイライト完成 |
| 8 | 駒音 / キャリアUI着手 (§5.6) | 戦績一覧が仮データで表示 |
| 9 | キャリアUI完成 (実データ接続) / 最適化 (60fps死守) / Bloom | 統合リハ通過 |
| 10 | ★統合点③ / デモ動画のカメラ・演出担当 | 撮って出しできる絵 |

※実データのレンジ調整: 実際のfearは0.2〜0.5に固まりがち。シェーダ側で
`remap`: `fear01 = saturate((fear - 0.2) / 0.3)` のようにC#側で引き伸ばして
から流す (AIの値は変えない。見た目の責任はこちら、が契約)。

## 10. 困ったときは
- BlendShapeが効かない → FBX Import設定のBlendShapesにチェックがあるか →
  それでもダメならB1へ (書き出し設定)
- イベントが来ない → U1へ (まずDebug.Logで購読確認)
- タスクが溢れる → **週次MTGで即申告**。必須リストの下から切る判断はU1と行う
