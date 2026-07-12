"""全駒トークンを文脈化する共有Transformer trunk。

実装予定:
- KokoroTrunk: 駒トークン列を受け、利きバイアス・関係性バイアス付き
  self-attentionで文脈化された駒表現を返す共有エンコーダ
- features.yaml の relations フラグによるバイアス項の有効/無効切り替え

参照: DESIGN.md §3(2) trunkと利きバイアス
実装: 週1-2 / 担当: A
"""
