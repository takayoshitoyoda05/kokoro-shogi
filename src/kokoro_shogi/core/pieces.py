"""駒種の語彙と、cshogi / SFEN / INTERFACE.md の3系統の対応表。

同じ「歩」が文脈によって3通りの表記を持つ:

- cshogi の駒コード: 先手 1〜14 / 後手 17〜30 (`cshogi.PIECE_SYMBOLS` の索引 + 16)
- SFEN の1文字: ``P`` (piece_id の接頭辞にも使う)
- INTERFACE.md の species: ``FU`` (Unityへ送る2文字コード)

この3つの対応をリポジトリ全体でここ1箇所だけに持つ。トークン化 (core/tokenizer.py) と
JSONL出力 (logging/jsonl.py) の双方がここを参照する。
"""

from __future__ import annotations

from typing import Literal

#: INTERFACE.md §3 の species。生駒8種 + 成駒6種
Species = Literal[
    "FU", "KY", "KE", "GI", "KI", "KA", "HI", "OU",  # 生駒
    "TO", "NY", "NK", "NG", "UM", "RY",              # 成駒
]

#: 学習で使う駒種インデックス (0-13)。順序は上の Literal と一致させる
SPECIES_ORDER: tuple[Species, ...] = (
    "FU", "KY", "KE", "GI", "KI", "KA", "HI", "OU",
    "TO", "NY", "NK", "NG", "UM", "RY",
)
SPECIES_TO_INDEX: dict[Species, int] = {
    species: index for index, species in enumerate(SPECIES_ORDER)
}

#: SFEN の1文字 → species。piece_id の接頭辞もこの文字を使う
SFEN_LETTER_TO_SPECIES: dict[str, Species] = {
    "P": "FU", "L": "KY", "N": "KE", "S": "GI",
    "G": "KI", "B": "KA", "R": "HI", "K": "OU",
}
SPECIES_TO_SFEN_LETTER: dict[Species, str] = {
    species: letter for letter, species in SFEN_LETTER_TO_SPECIES.items()
}

#: 成る前 → 成った後
PROMOTION: dict[Species, Species] = {
    "FU": "TO", "KY": "NY", "KE": "NK", "GI": "NG", "KA": "UM", "HI": "RY",
}
#: 成った後 → 成る前 (捕獲時に生駒へ戻すのに使う)
UNPROMOTION: dict[Species, Species] = {promoted: base for base, promoted in PROMOTION.items()}

#: cshogi の駒コード (先手 1〜14) → species。`cshogi.PIECE_SYMBOLS` の並びに対応する
_CSHOGI_PIECE_ORDER: tuple[Species, ...] = (
    "FU", "KY", "KE", "GI", "KA", "HI", "KI", "OU",  # 1-8:  p l n s b r g k
    "TO", "NY", "NK", "NG", "UM", "RY",              # 9-14: +p +l +n +s +b +r
)

#: 持ち駒の索引 → species。`cshogi.HAND_PIECE_SYMBOLS` (p l n s g b r) の並び。
#: 盤上の駒コード順 (…b r g…) とは金・角飛の順序が違うので別表にする。
HAND_INDEX_TO_SPECIES: tuple[Species, ...] = ("FU", "KY", "KE", "GI", "KI", "KA", "HI")
SPECIES_TO_HAND_INDEX: dict[Species, int] = {
    species: index for index, species in enumerate(HAND_INDEX_TO_SPECIES)
}

#: 後手の駒コードは先手 + 16 (cshogi の内部表現)
WHITE_PIECE_OFFSET = 16

BLACK = 0
WHITE = 1


def piece_code_to_species(code: int) -> Species:
    """cshogi の駒コード → species。空マス (0) はエラー。"""
    index = code - WHITE_PIECE_OFFSET if code > WHITE_PIECE_OFFSET else code
    if not 1 <= index <= len(_CSHOGI_PIECE_ORDER):
        raise ValueError(f"駒コードが不正です: {code}")
    return _CSHOGI_PIECE_ORDER[index - 1]


def piece_code_to_owner(code: int) -> int:
    """cshogi の駒コード → 所有者 (0=先手, 1=後手)。"""
    if code <= 0:
        raise ValueError(f"空マスには所有者がありません: {code}")
    return WHITE if code > WHITE_PIECE_OFFSET else BLACK


def base_species(species: Species) -> Species:
    """成駒なら生駒へ戻す。生駒はそのまま。"""
    return UNPROMOTION.get(species, species)


def promoted_species(species: Species) -> Species:
    """生駒なら成駒へ。成れない駒 (金・玉) や成駒済みはエラー。"""
    if species not in PROMOTION:
        raise ValueError(f"成れない駒です: {species}")
    return PROMOTION[species]
