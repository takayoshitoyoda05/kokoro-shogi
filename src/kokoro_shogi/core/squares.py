"""cshogi のマス番号と INTERFACE.md の "筋段" 文字列を相互変換する。

cshogi のマス番号は `筋index * 9 + 段index` (筋index = 筋-1, 段index = 段-1)。
INTERFACE.md §6 の表記は "76" = 7筋6段。例: 7g7f の from=60 → "77", to=59 → "76"。

盤外の表現も INTERFACE.md に従ってここに集約する:
- 持ち駒のマス: "hand"
- 打つ手の移動元: "00"
"""

from __future__ import annotations

HAND_SQUARE = "hand"
DROP_FROM = "00"

NUM_SQUARES = 81


def sq_to_str(square: int) -> str:
    """cshogi のマス番号 → "筋段" 文字列。"""
    if not 0 <= square < NUM_SQUARES:
        raise ValueError(f"マス番号は 0〜80 の範囲です: {square}")
    return f"{square // 9 + 1}{square % 9 + 1}"


def str_to_sq(text: str) -> int:
    """"筋段" 文字列 → cshogi のマス番号。"""
    if len(text) != 2 or not text.isdigit():
        raise ValueError(f'マスは "筋段" の2桁で指定してください: {text!r}')
    file = int(text[0])
    rank = int(text[1])
    if not (1 <= file <= 9 and 1 <= rank <= 9):
        raise ValueError(f"筋・段は 1〜9 の範囲です: {text!r}")
    return (file - 1) * 9 + (rank - 1)
