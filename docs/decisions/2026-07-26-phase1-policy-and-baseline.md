# 決定: Phase 1 の行動空間・方策ヘッド・CNNベースラインの取り方

- 日付: 2026-07-26
- 決めた人: Taka (A)
- 関連: `DESIGN.md` §3(1)(2)(7) §4a §5出典表 / Phase 1 Gate1 /
  [`2026-07-26-phase0-data-pipeline.md`](./2026-07-26-phase0-data-pipeline.md)

## 何を決めたか

### 1. 行動空間は (駒トークン, 移動先, 成り) の3つ組 = 6480通り

$40 \times 81 \times 2$。マス起点 (from-square) ではなく**駒トークン起点**にした。

理由は Phase 0 のトークン並び (piece_id 昇順固定) をそのまま行動空間に使えるため。
マス起点にすると、Phase 3 の感情GRU [A] や Phase 5 の個体性格 [B] が「駒 $i$ の
手」を指すときに毎回マス→駒の変換が要る。DESIGN.md §3(6) の会議も
「駒 $i$ が手 $a$ を主張する」形なので、駒起点のほうが後続Phaseに素直に繋がる。

**打ちの駒の選び方**: 同種の持ち駒が複数あるとき、どのトークンが動いたかは
一意でない。`PieceIdTracker._take_from_hand` が piece_id 順で最初の1枚を選ぶので、
合法手マスク側 (`data/dataset.legal_move_mask`) も同じ規則にした。
ここがずれると教師手がマスクの外に出て、損失が $-10^9$ のセルを指す。
`tests/test_dataset.py::test_teacher_move_is_inside_the_legal_mask` で固定した。

### 2. Phase 1 の方策ヘッドは内積型 (MLPではない)

DESIGN.md §3(3) の $u_{i,a} = [h_i \| E_{pos}[dst(a)] \| \rho(a)]$ をMLPに通す形は、
6480通り全てに対してMLPを走らせることになり素の方策には重すぎる。Phase 1 は

$$s_{i,a} = \langle W_p h_i,\; E_{move}[dst(a), \rho(a)] \rangle + b_{dst(a),\rho(a)}$$

にした (DESIGN.md §5 出典表の「src-dst方策ヘッド」= Lc0 と同じ形)。

Phase 2 の欲求ヘッドは**合法手だけ**を相手にすればよく (1局面あたり平均80手ほど)、
そちらは DESIGN.md どおりMLPで実装できる。出力の形 `(B, 40, 81, 2)` は変えないので、
`policy.py` から先 (マスク・softmax・損失・一致率) は書き換えずに移行できる。

### 3. Phase 1 に欲求・性格・自由項・単調mixingを入れない

DESIGN.md Phase 表の Phase 1 は「駒トークンtrunk + **通常policyヘッド** で蒸留」。
Gate1 が問うているのは「駒をトークンにして棋力が落ちないか」なので、
欲求 (3)・性格重み (4)・自由項 (5)・単調mixing (8) を混ぜると
何を測ったのか分からなくなる。これらは Phase 2 (週5) から。

損失も $L = L_{policy} + c_1 L_{value}$ の2項のみ。$L_{desire}$ / $L_g$ / $H(\pi)$ は
ヘッドが入ってから。

### 4. CNNベースラインは「trunkだけを差し替えた」対照実験にする

Gate1 の比較対象 `model/baseline_cnn.py` は、盤面を畳み込んだ特徴マップから
**各駒トークンのいるマスの特徴を取り出して**、`KokoroPolicy` と同じ
`PolicyHead` / `ValueHead` に渡す (持ち駒トークンは学習可能な駒台ベクトル)。

素の dlshogi 型 (マス起点の方策ヘッド) にしなかったのは、行動空間もヘッドも
違う2つを比べると「トークン化の是非」と「ヘッドの違い」が分離できないため。
違いを trunk (盤面CNN vs 駒トークンTransformer) だけに絞った。

入力プレーンは 9×9 × 43面 (盤上 駒種14×所有2 / 持ち駒 生駒7×所有2 / 手番1)。

**留保**: Gate1 の数字に疑義が出た場合は、マス起点の方策ヘッドを持つ完全な
dlshogi 型ベースラインを別途立てて再測定する。

### 5. 手番は全トークンに加算する埋め込みで与える

Phase 0 のADR §2 でトークナイザは先後の視点反転をしないと決めた。そのため
モデルに手番を知らせる手段が他にないので、$E_{turn}$ を全トークンに加算している
(DESIGN.md §3(1) の式には現れない項)。

視点反転版を試すなら、盤面を `rotate_sfen` してからトークン化する経路を
別に用意して比較する (トークナイザ内では反転しない)。

### 6. Transformer層は自前で持つ (PyTorch の不具合を踏んだため)

`nn.TransformerEncoderLayer` は使わず `model/trunk.py` の `KokoroEncoderLayer` を使う。

**PyTorch 2.13.0 の `nn.TransformerEncoderLayer` は `norm_first=True` のとき、
非ゼロの float `src_mask` を渡すと出力が NaN になる**。最小再現:

```python
layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True, norm_first=True).eval()
x = torch.randn(2, 5, 32)
layer(x, src_mask=torch.zeros(8, 5, 5))        # → 有限
layer(x, src_mask=torch.full((8, 5, 5), 0.01))  # → NaN
layer.self_attn(x, x, x, attn_mask=torch.full((8,5,5), 0.01))  # → 有限
```

全要素が同じ定数の加算マスクは softmax で打ち消えるので、本来は
`src_mask=None` と同じ結果になるはずのケースで壊れる。float マスクが bool として
解釈され全キーが塞がれている挙動に見える。`nn.MultiheadAttention` を直接呼ぶ
経路では正しく加算されるので、pre-norm層を自前で書いて回避した。

DESIGN.md §3(2) の $B_{利き}$ と Phase 3 の関係性 $r_{ij}$ [C] はどちらも
「attentionスコアへの加算」なので、この経路は今後も使い続ける。
回帰テスト: `tests/test_trunk.py::test_nonzero_attention_bias_stays_finite`
(素の `nn.TransformerEncoderLayer` に戻すと落ちる)。

**この不具合は実害が大きかった**: 学習中の損失は正常に下がるのに検証だけ NaN、
という形で出る (学習時は利きバイアスがまだ0に近く、学習が進んで0から離れた
時点から壊れる)。

### 7. train/val はシャード単位、valは先頭から取る

局面単位で分けると同一局の局面が両側に入ってリークする。シャードは棋譜順に
詰まっているのでシャード単位で分ける。

val を**先頭から**取るのは、最後のシャードが端数になりやすいため
(1,000,069局面を10万ずつ詰めると最後は69局面しかなく検証セットにならない)。

## なぜ

- 行動空間とヘッドの出力形を Phase 1 で固定しておくと、Phase 2 以降は
  スコアの計算式だけ差し替えれば済む。DESIGN.md 設計原則1「機能を外しても
  素の強いpolicy netが残る」を、コードの構造として担保する形。
- ベースラインを「trunkだけ違う」形にしたのは、Gate1 の -2% という基準が
  トークン化の是非を測るためのものだから。

## 影響

- A: Phase 2 では `heads.py` に欲求・性格・自由項・単調mixingを足し、
  `policy.py` のスコア計算を差し替える。`dataset.py` と `distill.py` は無変更でよい。
- 他メンバー: 影響なし (INTERFACE.md は変更していない)。

## Gate1 判定: 通過 (2026-08-10 追記)

floodgate 2024 の 1,000,069局面 (train 900,053 / val 100,016、シャード単位分割) で
2モデルを同一設定・同一分割で学習した。

```
uv run python scripts/run_gate1.py --shard-dir data/shards/2024 \
    --epochs 10 --batch-size 512 --device cuda
```

| モデル | パラメータ | ベスト検証一致率 (エポック) |
|---|---|---|
| CNNベースライン (192ch/10block) | 7.01M | 31.28% (6) |
| 駒トークンTransformer | 5.01M | **34.27%** (7) |

**差 +2.98pt で通過** (合格ライン -2pt 以内)。-2%の許容どころか、駒トークン側が
少ないパラメータで CNN を明確に上回った。「駒をトークンにして棋力が落ちないか」
という Gate1 の問いには**落ちない**が答え。§9 の対策 (利きバイアス強化 /
ハイブリッドトークン化) は不要。

補足:

- 実行環境は開発機 (WSL2 + RTX 3050 6GB)。torch を cu128 ビルドに固定したことで
  WSL でも GPU が使えるようになった (pyproject.toml の `[tool.uv.index]` 参照)。
  VRAM の都合で batch は 1024 でなく 512 (両モデル同一なので比較は有効)。
  1エポックは CNN 880秒 / kokoro 707秒。
- 両モデルとも 6-7 エポックで過学習に入る (kokoro 最終: train 68.5% / val 32.6%)。
  次に精度を積むなら、エポックを増やすより 2024 以外の年次シャード追加が先。
- 生データ: `checkpoints/gate1.json` / 学習ログ: `checkpoints/gate1_run.log` /
  ベスト重み: `checkpoints/{cnn,kokoro}_gate1_best.pt`

### CPU での予備計測 (判定ではない・履歴として残す)

20,000局面 (全体の2%) ・2エポック・CNNは96ch/6block という縮小条件:

| 学習率 | CNN | 駒トークン | 差 |
|---|---|---|---|
| 定数 1e-3 (1エポック) | 14.87% | 10.46% | -4.41% |
| ウォームアップ+コサイン (2エポック) | 16.26% | 13.45% | **-2.80%** |

**この数字は Gate1 の判定ではない**。データ量もエポック数も桁違いに足りず、
両モデルとも強く過学習している (CNN: train 36% / val 16%)。

読み取れるのは次の2点:

1. **学習率スケジュールで差が 1.6pt 縮まった**。Transformer は定数学習率だと
   序盤が不安定で、ウォームアップなしの比較は Transformer に不利。
   Gate1 は必ず同一スケジュールで測る (`scripts/run_gate1.py` がそれを強制する)。
2. 現時点では駒トークン側が負けている。ただしこの規模では Transformer が
   不利になりやすい (パラメータ 5.01M 対 1.30M で、データが2万局面しかない)。
   1M局面での再測定が必要。それでも届かない場合は DESIGN.md §9 の対策
   (利きバイアス強化 / マス+駒ハイブリッドトークン化) に進む。
