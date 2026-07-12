"""機能フラグ (configs/features.yaml) の性質を検証する (予定)。

検証予定の性質:
- 全フラグ false の最小構成でも forward が例外なく通る
- フラグを1つずつ true にしても出力の形状・正規化が保たれる

参照: configs/features.yaml / DESIGN.md §3
実装: 週2 (以降フラグ追加のたびに拡張) / 担当: A
"""
