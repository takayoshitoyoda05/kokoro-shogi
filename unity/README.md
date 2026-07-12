# unity/

ここに U1 が Unity Hub で **KokoroShogi** プロジェクトを作成する。

- Unity バージョン: **6000.3.8f1 LTS**
- テンプレート: **Universal 3D (URP)**
- 作成場所: このディレクトリ直下 (`unity/KokoroShogi/`)

## 注意

- **clone 前に `git lfs install` を必ず実行すること** (fbx / png / wav などは LFS 管理。`.gitattributes` 参照)
- `Library/` `Temp/` `Logs/` `UserSettings/` `obj/` `Build/` は `.gitignore` 済み。コミットしないこと。
- Python 側との通信は JSONL / WebSocket (`docs/INTERFACE.md`, schema 1.0) のみ。
