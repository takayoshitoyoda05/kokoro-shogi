# Kokoro-Shogi (こころ将棋)

**駒が心を持つ将棋 AI** — 盤面ではなく 40 枚の駒それぞれをエージェントとして扱い、
駒の欲求・性格・感情から一手を決め、その内面を Unity の 3D 盤面で観戦できます。

> 歩が怯えて青ざめ、金と玉が絆の光で結ばれ、駒たちの会議が吹き出しで交わされ、
> AI の実況が流れる。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](./pyproject.toml)

---

## 何が普通の将棋 AI と違うのか

従来の将棋 AI は盤面全体をひとつの評価関数で見ます。Kokoro-Shogi は
**駒 40 枚をそれぞれ 1 人の登場人物として扱います**。

- 各駒が **欲求 6 軸**（生存・攻撃・成り・守備・前進・再登場）を持つ
- 駒種ごとの **性格** が、どの欲求を重視するかを決める
- 対局中の出来事で **感情** が動く（仲間が近くで取られると強く動揺する）
- 駒どうしが **絆** を育てる（長く守り合った金と玉）
- 一手は駒たちの **会議**（主張の競り合い）で決まり、議事録が残る
- 取られた駒は消えず、**相手の駒として転生**して盤に戻る

指し手は「どの駒が、どこへ」という形で出力され、**40 駒 × 81 マス × 成り = 6,480 通りを
まとめて 1 つの softmax** にかけます。駒ごとに正規化しないので、「ある駒の主張が強くなれば
別の駒の確率が下がる」という競合が、そのまま 1 つの微分可能な式になります。

**推論時に先読み（木探索）を一切しません。** 局面を 1 回計算して 1 手を返します。

これらの「内面」はすべて**零初期化された足し算**として組み込まれているので、機能フラグを
切ると素の駒トークン Transformer と数値的に完全に一致します。

---

## どのくらい強いのか

正直に書きます。**やねうら王 + Háo（NNUE）を探索 1 手に制限した相手に、最も強いモデルで
勝率 0.230** です（floodgate の序盤 500 種から 100 局、先後各 50、SE ≈ 0.04）。
アマチュアの将棋 AI としては弱い部類で、市販ソフトには遠く及びません。
なお同梱しているのは文化リーグのモデル（0.10 〜 0.16）で、0.230 を出した蒸留 + PPO のモデルは
サイズの都合で同梱していません。

2026 年 9 月に初めて外部エンジンを基準にして測り直したところ、**それまで「効いている」と
思っていた施策のほとんどが、外では効いていませんでした**。

| 施策 | 動かしたパラメータ | 外部勝率の変化 |
|---|---:|---:|
| 棋譜からの蒸留（3 エポック） | 全体 | **0.220**（ここが土台） |
| 自己対戦 PPO 9,200 反復 | 全体 | +0.01 |
| 文化リーグ（性格の進化） | 224 | −0.07 〜 −0.13 |
| 感情 GRU を学習で動かす | 4,128 | **−0.18**（用量依存） |
| エンジン教師蒸留 20,000 局 | 全体 | ±0（p = 0.84） |
| 蒸留 4 エポック目 | 全体 | ±0（p = 0.88） |

パラメータ 500 万個のうち **95% が共有 Transformer、「こころ」の部分は 0.09%** しかありません。
内面をいじっても強さが動かないのは、この比率を見れば当然でした。

現在の律速は**データ量**と診断しています（学習データの一致率 47.3% に対し検証データ 41.5%、
その差がエポックごとに拡大＝過学習）。測定方法・失敗した施策・次の方針を含めた全記録は
**[技術報告](./docs/TECHNICAL_REPORT.md)** にあります（機械学習の予備知識がなくても読める形で
書いてあります）。

---

## 動かしてみる

Python 3.12 と [uv](https://docs.astral.sh/uv/) が必要です。

```bash
git clone https://github.com/takayoshitoyoda05/kokoro-shogi.git
cd kokoro-shogi
uv sync --group train        # PyTorch を含む依存一式
```

学習済みモデルが 2 つ同梱されているので、**clone すればすぐ対局できます**。

### まず動くか確かめる（Unity 不要・数十秒）

```bash
uv run python scripts/play_server.py --selfcheck
```

乱択相手に 1 局を終局まで指します。起動直後の 1 行目に、どのモデルと機能が載ったかが出ます。

```
checkpoint: league.pt / culture: culture1 / device: cpu / tau: 0.0 / mood: 感情GRU / relations: r_ij状態 / council: ON
  ply   2 eval +0.053  2ラウンドの議論で形勢が動いた。当初は△玉の主張が通りかけたが、最後は△歩が8c8dを押し切った。
  ...
selfcheck: 42 手で終局 (人間の手 21 回, 0.9s, 0.02s/手) / career 40 駒 / MVP B22_gen0_0006
```

末尾 3 つが `感情GRU` / `r_ij状態` / `ON` になっていれば内面つきで動いています。
`ply` の行に出ているのは、**その手を決めた会議の実況**です。

### Unity の 3D 盤面で観戦・対局する

Python を先に起動してから Unity を Play してください（逆順では繋がりません）。

```bash
uv run python scripts/play_server.py --host 0.0.0.0
```

1 手ごとに、40 駒それぞれの感情・欲求・発言力・絆、会議の議事録、自然言語の実況が
WebSocket で流れます。データ形式の仕様は [`docs/INTERFACE.md`](./docs/INTERFACE.md) です。

Unity プロジェクトは `unity/KokoroShogi/`（Unity 6000.3.8f1）にあります。

---

## 同梱モデルと「文化」

| ファイル | 中身 | 対 Háo depth1 |
|---|---|---:|
| `checkpoints/league_E2b_grace/league.pt`（既定） | 文化リーグ（猶予つき淘汰） | 0.10 〜 0.13 |
| `checkpoints/league_E7_ema/league.pt` | 文化リーグ（適応度 EMA）。**同梱 2 つでは強い方** | 0.16 |

```bash
uv run python scripts/play_server.py --checkpoint checkpoints/league_E7_ema/league.pt
```

`league.pt` は**ひとつの共有ネットワークと 6 つの「文化」**（駒種の性格パラメータ）を持っていて、
どの文化で指すかを選べます。同じ盤面でも、生存を重んじる文化と攻撃を重んじる文化では違う手を指します。

```bash
uv run python scripts/play_server.py --culture culture2
```

> **注意**: 文化リーグは自分の過去のモデルとの対戦では強くなったように見えましたが、
> 外部エンジンを基準にすると蒸留だけのモデルより弱くなりました（0.220 → 0.10 〜 0.13）。
> **自己対戦の勝率は、自分の系列の中でしか意味を持ちません。**
> このため強さの判定は外部エンジンでのみ行っています
> （[技術報告 §9.1〜9.2](./docs/TECHNICAL_REPORT.md)）。

その他の主な引数: `--checkpoint`（モデル差し替え）/ `--human white`（人間が後手）/
`--tau 0.1`（手を揺らす。0 なら常に最善手）/ `--device cuda` / `--port` / `--max-plies`。

---

## 自分で学習させる

学習データは [floodgate](http://wdoor.c.u-tokyo.ac.jp/shogi/)（コンピュータ将棋の対局サーバ）の
公開棋譜です。

```bash
uv run python scripts/download_floodgate.py --year 2024      # 年次アーカイブ（約 350MB）
uv run python scripts/make_labels.py --csa-dir data/floodgate/csa   # 棋譜 → 学習用シャード
uv run python -m kokoro_shogi.train.mood_distill \
    --shard-dir data/shards --relations --council --amp --epochs 1
```

欲求 6 軸の教師ラベルは**棋譜からルールで自動生成**されます（人手のアノテーションはゼロ）。
1 エポック（15.6 万局）に GPU で約 13 時間かかります。

---

## 構成

```
src/kokoro_shogi/     モデル・学習・データ処理
  model/              Transformer trunk、欲求/性格/価値ヘッド、会議ループ、感情 GRU
  train/              蒸留・自己対戦 PPO・文化リーグ
  data/               floodgate 棋譜の読み込み、欲求ラベル生成、シャード
  server/             対局セッション
scripts/              対局サーバ、棋譜の取得と前処理、可視化
unity/KokoroShogi/    Unity プロジェクト（3D 盤面・演出・UI）
docs/                 技術報告・データ契約・設計書・判断記録
tests/                18 ファイル、330 テスト
```

主な文書:

| 文書 | 内容 |
|---|---|
| [`docs/TECHNICAL_REPORT.md`](./docs/TECHNICAL_REPORT.md) | **技術報告** — 表現・アーキテクチャ・学習・推論・評価と、効かなかった施策の実測 |
| [`DESIGN.md`](./DESIGN.md) | 設計書（数式仕様・出典） |
| [`docs/INTERFACE.md`](./docs/INTERFACE.md) | Python ↔ Unity のデータ契約（JSON スキーマ） |
| [`docs/decisions/`](./docs/decisions/) | 判断記録（ADR）— 何を試して何が効かなかったかの一次記録 |
| [`TEAM_PLAN.md`](./TEAM_PLAN.md) | チーム運営の計画（役割・マイルストーン） |

---

## このプロジェクトについて

学生チーム（AI 1 名 / Unity 2 名 / Blender 2 名）の作品です。「駒に心を持たせたら
面白いのでは」という発想から始まり、**面白さの機能を全部切っても素の将棋 AI が残る**
という原則で設計されています。

技術的に主張できるのは、強さよりも次の 3 点だと考えています。

1. **ターン制完全情報ゲームでの駒粒度マルチエージェント学習** — 駒をトークンにする将棋 AI の
   先行例が見つかりませんでした（Leela・Chessformer・Ruoss らはいずれもマスか文字列）
2. **持ち駒 = エージェントの転生** — 捕獲でエージェントの所有権が敵に移る現象は、将棋にしか
   存在しません
3. **解釈可能性と強さのトレードオフの定量化** — 「欲求でどれだけ説明できるか」と一致率の
   曲線を、正則化係数を振って描けます

そして**否定的な結果も同じ重みで公開しています**。内面を強化学習で育てる路線は有害でした。
自己対戦の勝率は系統の外では通用しませんでした。これらは技術報告の第 9 章に検定値つきで
記録してあります。

---

## ライセンス / クレジット

本体は [MIT License](./LICENSE) です。

- **棋譜データ**: [floodgate (wdoor)](http://wdoor.c.u-tokyo.ac.jp/shogi/) — 利用条件に従ってください
- **同梱の学習済みモデル**: floodgate 棋譜から学習したもの。MIT の下で自由に使えます
- **実況 LLM**: [Qwen2.5](https://github.com/QwenLM/Qwen2.5)（Apache 2.0）via [Ollama](https://ollama.com/)
- **将棋ライブラリ**: [cshogi](https://github.com/TadaoYamaoka/cshogi)
- **外部基準に使ったエンジン**: [やねうら王](https://github.com/yaneurao/YaneuraOu) + Háo（NNUE 評価関数、GPLv3）— 別途入手が必要です
- **効果音・BGM・フォント**: `unity/KokoroShogi/Assets/_Project/Audio/LICENSES.md`
- **テクスチャ素材**: `unity/KokoroShogi/Assets/_Project/Textures/LICENSES_textures.md`

Unity プロジェクト内のアセットは、それぞれ上記のライセンス一覧に従います。
MIT が適用されるのは本リポジトリのソースコードと同梱モデルです。
