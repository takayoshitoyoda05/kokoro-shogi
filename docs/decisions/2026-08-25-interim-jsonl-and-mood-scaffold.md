# 決定: 中間版モデルJSONLの納品方式と感情GRUの土台

- 日付: 2026-08-25
- 決めた人: Taka (A)
- 関連: `DESIGN.md` §3(9) [A] / `docs/INTERFACE.md` §3 / Phase 3 /
  [`2026-07-26-phase2-desire-structure.md`](./2026-07-26-phase2-desire-structure.md)

## 何を決めたか

### 1. 週6の実データ納品を「中間版」と「完全版」に分ける

感情GRU [A] の学習を待たずに、**いま学習済みの範囲だけ本物にしたJSONL**を
`sample_data/model/` へ先行納品する (`scripts/export_model_jsonl.py`)。

| フィールド | 中間版 (今回) | 完全版 (Phase 3 後) |
|---|---|---|
| 指し手・SFEN | モデル自己対戦 (温度0.25) | 同じ |
| `eval` | valueヘッド (手番視点→先手視点へ変換) | 同じ |
| `desire` | 欲求ヘッド (下記§2の縮約) | 同じ |
| `alpha` | 単調mixing | 同じ |
| `mood` | ヒューリスティック継続 | 感情GRU + MoodProjection |
| `relations` | ヒューリスティック継続 | 関係性バイアス [C] |

チェックポイントは `desire_lambda0.1.pt` (説明率91.5%・一致率低下-0.42ptの点)。
U2はこれで desire / alpha / eval のレンジ調整を前倒しできる。

### 2. 手ごとの欲求 $d_i(a)$ から駒ごとの `desire` への縮約は方策重み付け平均

INTERFACE.md の `pieces[].desire` は駒ごと、欲求ヘッドの出力は手ごと。

$$d_i = \frac{\sum_{a \in \mathcal{A}_i} \pi(a)\, d_i(a)}{\sum_{a \in \mathcal{A}_i} \pi(a)}$$

= 「その駒が指しそうな手に込めた欲求」。単純平均にしなかったのは、ほぼ指さない
非現実的な手 (端歩の犠打など) の欲求が混ざって表示がぼやけるため。

手番でない側の駒は合法手を持たないので、**手番を反転した盤面でもう1回 forward**
して同じ縮約を行う (王手放置で反転局面が不正になる場合は全162手の単純平均へ
フォールバック)。1手あたり forward 2回のコストは許容 (エクスポートはオフライン)。

### 3. 実測レンジ (U2への申し送り)

10局・1,321レコードの実測: `desire.survive` は平均0.82に高止まり、
attack/promote/redeploy は平均0.06以下。`alpha` の局面内最大値は平均0.066。
**PLAN_U2 §9※のremap方式を desire / alpha にも適用しないと画面上で読めない**。
AI側は値を加工しない (見た目の責任はUnity側、が契約)。

### 4. 感情GRU [A] は「モジュール先行・学習後追い」で入れる

`model/mood.py` を実装した (イベント特徴8軸 / `MoodGRU` / `MoodProjection`)。
合流点は2つ (DESIGN.md §3(1)(4)):

- trunk 埋め込みへの $W_m m_i$ … **零初期化** + features.mood フラグOFF時は
  モジュール自体を作らない。Phase 1-2 チェックポイントの state_dict と完全互換で、
  ONにした直後もOFFと同じ出力から学習が始まる (effect_bias と同じ流儀)
- `PersonalityWeights` への連結 (d_theta 16 → 16+32)。こちらは形が変わるので
  ウォームスタート時は personality.project だけ再初期化する

イベント特徴で決めたこと:

- 被取イベントは DESIGN.md の例のとおり **L1距離減衰** ($\beta=2$) で全駒に届く。
  「失った側」は取られた駒の**現所有者の反対** (tracker は指した後の状態で、
  取られた駒の owner は既に捕獲側に書き換わっているため。ここを間違えると
  味方被弾と敵撃破が逆転する — `tests/test_mood.py` で固定)
- 持ち駒には was_captured 以外届かない (盤上の出来事から切り離す)
- 王手フラグは「王手されている側の全駒」に立てる (cshogi.is_check は手番側の玉)

### 5. 系列学習の実装と結果 (2026-08-27 追記)

計画していた系列学習まで実施した。決定事項:

- **イベント特徴はシャードに足さず、dataset 側で局を再生して復元する**
  (`data/sequence.py`)。シャードの `move` 列に生の指し手が入っているので
  cshogi + PieceIdTracker の再生で正確に作れる。シャード形式の変更なし
- **局の境界は game_index の変化点だけでは決められない**。game_index は
  棋譜ファイル内の連番で、1局だけのファイルが連続すると隣接する別の局が
  同じ番号になり融合する (実データでクラッシュを確認)。境界は
  「変化点 ∪ 初期局面の行」とし、初期局面から始まらない断片は捨てる
- **学習は truncated BPTT 16手** (`train/mood_distill.py`)。Phase 2 の
  desire_lambda0.1 からウォームスタート、lr 2e-4
- **MoodProjection は学習後のリッジ回帰でアンカー**
  (`scripts/fit_mood_projection.py`)。sigmoid/tanh のリンク空間で
  ヒューリスティック mood に回帰し、軸の向きを人間の言葉に固定する

結果 (data/shards/2024 全6,903局・2エポック、計85分/RTX GPU):

- val 一致率 **35.1%** (Phase 2 の34.3%と同水準 = 感情を入れても棋力を
  壊さない、設計原則1)。説明率 71%
- 射影の R² (リンク空間、31.4万駒局面): fear **0.40** / aggression 0.27 /
  valence 0.22。GRU状態が脅威情報を実際に保持している
- `--mood-checkpoint` 付き再納品 (1,343レコード): fear は 0.01〜1.00 に分布
  (平均0.11, sd 0.16)。駒取りの局面で取られた側の平均 fear が上がる反応を確認。
  **valence はレンジが狭い** (-0.39〜0.10, sd 0.045) — 形勢情報が $m_i$ に
  乏しいことを示す。イベント特徴に形勢差分を足すのが次の改善候補

## 検証

- `tests/test_mood.py` (10件): イベントの意味論 (距離減衰の順序・王手の到達範囲・
  持ち駒の隔離)、GRU/射影の形状と値域、フラグOFF時のチェックポイント互換、
  零初期化の同値性
- `sample_data/model/` 全レコードが `logging/jsonl.py` スキーマと cshogi の
  SFENパースを通過、alpha 総和1、全体 236→246 テスト通過
