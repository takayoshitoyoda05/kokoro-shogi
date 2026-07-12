"""JSONL出力 (logging/jsonl.py) の性質を検証する (予定)。

検証予定の性質:
- 出力レコードが schema 1.0 (docs/INTERFACE.md) に準拠する
- 必須キーの欠落・型不一致を検証エラーとして検出できる
- gen_sample_jsonl.py の出力もこの検証を通る

参照: docs/INTERFACE.md (schema 1.0)
実装: 週1 / 担当: A
"""
