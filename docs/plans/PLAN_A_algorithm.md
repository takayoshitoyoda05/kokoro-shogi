# 個人計画書 A: アルゴリズム担当 (Taka)

> 親文書: `TEAM_PLAN.md` / 技術正本: `DESIGN.md`

## ミッション
AIエンジン全体 (蒸留→機能追加→自己対戦RL) の実装と、Unity側への
データ供給 (スキーマ策定・サンプルJSONL・WebSocketサーバ)。

## 技術スタック

| 領域 | 採用技術 | 備考 |
|---|---|---|
| 言語 | Python 3.12 | uvで管理 (既存環境流用) |
| 深層学習 | PyTorch 2.x (CUDA) | trunk/ヘッド/GRU全部 |
| 将棋環境 | cshogi | 高速。合法手生成・SFEN・棋譜パース |
| 棋譜データ | floodgate CSA棋譜 | wdoor.c.u-tokyo.ac.jp から取得 |
| 実験管理 | 自作実験ログ (claude-ml-template流) + TensorBoard | λ_g掃引・アブレーション表 |
| ハイパラ探索 | Optuna | 経験あり。c1,c2,λ_g,τの調整 |
| RL | 自作MAPPO (→必要ならMiniZero参考にGumbel AZ) | DESIGN.md §4b/§4c |
| 永続化 | SQLite (標準ライブラリ) | θ_ind, career, lineage |
| 実況LLM | Ollama + Qwen2.5 7B Instruct (Q4_K_M) | ローカル・無料。CPU推論可 |
| 通信 | websockets (Python) + JSON | server.py |
| スキーマ | JSON Schema + INTERFACE.md | 変更はA承認制 |
| CI/品質 | pytest, ruff, GitHub Actions | claude-ml-templateのCI流用 |
| 開発環境 | WSL2 / gmk02 (Linux), nvim, Claude Code | 既存2台体制 |

## 週次タスク (TEAM_PLAN.md v4 / 10週・全機能版)

| 週 | タスク | 成果物 |
|---|---|---|
| 1 | INTERFACE.md策定 / **サンプルJSONL納品** / Phase 0着手 | schema v1.0, sample/*.jsonl |
| 2 | Phase 0完了: floodgate前処理, トークン化, 欲求ラベル | data pipeline + tests |
| 3 | Phase 1: 蒸留。**Gate1: 一致率CNN比-2%以内** | baseline + レポート |
| 4 | Phase 2: 欲求/性格/自由項, λ_g掃引 (3点に縮小) | 簡易トレードオフ曲線 |
| 5 | Phase 3: 感情GRU+関係バイアス / server.py稼働 | server v1 |
| 6 | Phase 4: 会議調停 + Ollama実況 → **実データJSONL納品 (会議・実況込み)** | real/*.jsonl |
| 7 | Phase 5: COMA功績配分, θ_ind更新, SQLite, キャリア統計 | career DB |
| 8 | Phase 7: **5五将棋でMAPPO自己対戦RL** + E忠誠3案の少数対局比較 | 学習曲線 + E比較表 |
| 9 | Phase 6: **5五将棋で文化リーグ (4文化)** / careerデータ配信 (U2のキャリアUI用) / 安定化 | 分化の観測ログ + career API |
| 10 | 統合点③: フルデモ用モデル・デモ動画/発表資料 | — |

**縮退の原則**: 週8-9のRL・リーグ・忠誠は**5五将棋+小型モデル**で「動く・
兆候が見える」までが10週の合格ライン。本将棋フルスケールは継続タスク。
週次Gate判断: RLが週8末で「レート上昇」未達ならデモは蒸留モデル+学習曲線
グラフに切替(TEAM_PLAN §3注記)。

**実装効率の生命線**: claude-ml-templateのPlanner/Generator/Evaluatorループを
フル活用。週7以降は毎週1 Phaseの強行軍なので、各Phaseの初日にplan承認まで
終わらせるリズムを守る。

## 他メンバーとの境界
- Unity側との会話は **U1経由 + INTERFACE.md** のみ。個別実装相談は受けない
  (受けるとスキーマが口約束で崩れる)
- 納期死守は2点のみ: 週1サンプルJSONL / 週9-10実データJSONL

## リスク対応 (A固有)
- Gate1不通過 → 利きバイアス強化 / ハイブリッドトークン化 (DESIGN.md §9)
- 学習が重い → 5五将棋に縮退して機能検証を先行、本将棋は後追い
- server.pyに時間を食われる → リプレイ(JSONL)だけでも統合点②の大半は成立
