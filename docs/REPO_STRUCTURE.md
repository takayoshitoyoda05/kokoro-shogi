# Kokoro-Shogi リポジトリ構成 (Git管理ガイド / モノレポ版)

> 対象: チーム全員 (初学者含む)。リポジトリの構成・ブランチの使い方・
> 置いてよいもの/いけないものを定義する。迷ったらこの文書に従う。

---

## 1. 全体像: モノレポ1つ + 契約書2枚

リポジトリは **`kokoro-shogi` の1つだけ** (モノレポ)。AI側 (Python)、
Unity側、Blender素材が同居する。世界をつなぐ契約書は2枚:

- `docs/INTERFACE.md` — AI⇄Unityのデータ契約 (JSONスキーマ)。変更はA承認制
- `unity/KokoroShogi/Assets/_Project/docs/naming.md` — Blender⇄Unityの
  アセット契約 (シェイプキー名・スケール・ピボット)。B1/B2/U2の三者管理

モノレポの利点: 契約書のコピー同期が不要 / 全員が全体を1回のcloneで持てる。
欠点 (Unityの大ファイルでcloneが重い) はGit LFSで抑える。

## 2. ディレクトリ構成

```
kokoro-shogi/
├── README.md                # プロジェクト概要 (最初に読む)
├── DESIGN.md                # AIアルゴリズム設計書 v2 (数式仕様)
├── TEAM_PLAN.md             # チーム全体計画 (v4: 10週・全機能)
├── PROMPT_initial_repo.md   # 初期リポジトリ生成に使ったプロンプト (記録)
├── pyproject.toml           # Python依存 (uv管理)
├── .gitignore / .gitattributes (LFS)
│
├── docs/
│   ├── INTERFACE.md         # ★データ契約の正本 (schema 1.0)
│   ├── REPO_STRUCTURE.md    # この文書
│   ├── plans/               # 個人計画書5通 (PLAN_A, U1, U2, B1, B2)
│   └── decisions/           # 設計判断の記録 (1判断1ファイル)
│
├── src/kokoro_shogi/        # AI本体 (担当: A)
│   ├── core/  model/  train/  data/
│   ├── persist/  logging/  viz/  server/
├── scripts/                 # download_floodgate / make_labels / gen_sample_jsonl
├── configs/                 # base.yaml / features.yaml
├── tests/
├── sample_data/             # サンプルJSONL (週1納品物。Unityが読む)
│
├── blender/                 # .blend原本 (担当: B1/B2, LFS管理)
│
└── unity/                   # Unity側 (担当: U1が親、U2が中身)
    ├── README.md            # セットアップ案内
    └── KokoroShogi/         # Unityプロジェクト (6000.3.8f1 LTS / URP)
        ├── Assets/
        │   ├── _Project/    # ★自作物は全部この下
        │   │   ├── Scenes/     Replay / Live / Play        (U1)
        │   │   ├── Scripts/Core, Net, Replay               (U1)
        │   │   ├── Scripts/Fx, UI                          (U2)
        │   │   ├── Prefabs/  Materials/  VFX/  UI/  Audio/ (U2)
        │   │   ├── Models/   Textures/                     (B1/B2)
        │   │   └── docs/naming.md                          (B1/B2/U2)
        │   ├── Plugins/     # NativeWebSocket, DOTween 等
        │   └── StreamingAssets/sample_data/  # JSONLのコピー
        ├── Packages/        # manifest.json (コミットする)
        └── ProjectSettings/ # コミットする
```

**フォルダ = 担当**。自分のフォルダの外を触るときはDiscordで一言。
これでコンフリクトの大半は防げる。

## 3. 何をコミットし、何をしないか

### コミットする
- 全ソースコード・設定・文書 / Unityの `Assets/ Packages/ ProjectSettings/`
- FBX・テクスチャ・音・.blend (すべてLFS経由)
- サンプルJSONL (sample_data/)

### コミットしない (.gitignore済み)
- Python: `__pycache__/ .venv/ data/ checkpoints/ runs/ *.db *.pth .env`
  (棋譜生データと学習済みモデルは巨大。共有が必要なモデルはGitHub Releasesへ)
- Unity: `Library/ Temp/ Logs/ UserSettings/ obj/ Build/` (自動生成・個人環境)
- Blender: `*.blend1 *.blend2` (自動バックアップ)

### Git LFS (.gitattributes 登録済み)
```
*.fbx *.png *.psd *.wav *.mp3 *.blend
```
**全員、clone より前に1回だけ `git lfs install` を実行** (忘れるとモデルが
「ポインタファイル」になって開けない — 初学者事故の第1位)。

## 4. ブランチ運用 (ルールは3つだけ)

```
main ────●────●────●──→  常に「動く」状態。直接push禁止 (保護設定)
          \    \
           \    feature/council-ui   ← U2の作業
            feature/replay-v0        ← U1の作業
```
1. 作業は必ず `feature/内容` ブランチを切ってから
2. 終わったらPull Request。説明に「何を・なぜ・確認方法」。
   マージ担当: Python側=A、Unity/Blender側=U1
3. mainが壊れたら全作業より優先で直す

コミットメッセージは日本語一行でよい (例: `駒の移動アニメを追加`)。
1日の終わりに必ずpush。

## 5. Unity特有の事故防止

- Edit → Project Settings → Editor → Asset Serialization: **Force Text**
  (U1が初週に設定。シーンをテキスト化しマージ事故を減らす)
- `.unity` と `.prefab` はマージ不可能と考える。シーンは担当者以外編集しない、
  同じPrefabを触るときは事前宣言
- Unityのバージョンは **6000.3.8f1 (LTS) に固定**。他バージョンで開かない
  (上位版で開くと下位版に戻せなくなる)
- pullの前にUnityを閉じる (開いたままpullするとLibraryが壊れることがある)

## 6. INTERFACE.md / naming.md の変更手順

INTERFACE.md (データ契約): INTERFACE.md §7 の儀式に従う
(U1がIssue化 → A承認 → バージョン更新 → サンプル再生成 → 全員通知)。

naming.md (アセット契約): B1/B2/U2の3人が合意 → 更新 → Discord通知。
シェイプキー名など「既に出荷したFBXに影響する変更」は原則しない
(するなら全FBX再書き出しをB1/B2が引き受ける覚悟で)。

## 7. セットアップの入口

| 誰 | 読むもの |
|---|---|
| 全員 | ルートREADME → `git lfs install` → clone |
| A | README → `uv sync` |
| U1/U2 | unity/README → Unity Hub で 6000.3.8f1 を入れて開く |
| B1/B2 | Blender 4.x + 各自の個人計画書 §1 |

「READMEの手順通りで動かなければ、それはREADMEのバグ」としてIssueを立てる
(初学者が自分を責めて止まるのを防ぐ文化)。
