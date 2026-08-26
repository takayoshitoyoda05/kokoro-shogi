"""会議ログから実況テキストを生成する (DESIGN.md §3(6) 議事録 → 実況)。

LLMを使わないテンプレート実況 (`TemplateNarrator`)。素材は council の
ラウンドごとの top-k 提案 (piece_id, move, bid) と、実際に指された手。
見どころの検出は3パターン:

- **逆転**: 第1ラウンドの首位提案と最終ラウンドの首位が違う駒
- **初志貫徹**: 全ラウンド同じ提案が首位のまま通った
- **接戦**: 最終ラウンドの1位と2位のbid差が小さい

OllamaNarrator (ローカルLLM) は後続。narration の受け手 (U2の字幕) は
文字列しか見ないので、差し替えてもインターフェースは変わらない。
"""

from __future__ import annotations

from kokoro_shogi.logging.jsonl import CouncilRound

#: 駒種の日本語名 (SFEN準拠コード → 表示名)
SPECIES_JA = {
    "FU": "歩", "KY": "香", "KE": "桂", "GI": "銀", "KI": "金",
    "KA": "角", "HI": "飛", "OU": "玉",
    "TO": "と金", "NY": "成香", "NK": "成桂", "NG": "成銀", "UM": "馬", "RY": "竜",
}

#: 接戦とみなす1位と2位のbid差
CLOSE_MARGIN = 0.15


def _piece_label(piece_id: str, species: str, owner: int) -> str:
    """▲飛 / △歩 のような表示名。"""
    side = "▲" if owner == 0 else "△"
    return side + SPECIES_JA.get(species, species)


class TemplateNarrator:
    """council (INTERFACE.md §3) から1文の実況を作る。

    `pieces_meta` は piece_id → (species, owner)。会議OFF (空ログ) なら空文字。
    """

    def narrate(
        self,
        council: list[CouncilRound],
        chosen_piece_id: str | None,
        pieces_meta: dict[str, tuple[str, int]],
    ) -> str:
        if not council or not council[0].proposals:
            return ""

        first = council[0].proposals
        last = council[-1].proposals
        rounds = len(council)

        def label(piece_id: str) -> str:
            species, owner = pieces_meta.get(piece_id, (piece_id, 0))
            return _piece_label(piece_id, species, owner)

        leader_first = first[0]
        leader_last = last[0]

        # 実際に指した駒が最後まで首位だったか
        chosen_label = label(chosen_piece_id) if chosen_piece_id else label(leader_last.piece_id)

        if leader_first.piece_id != leader_last.piece_id:
            return (
                f"{rounds}ラウンドの議論で形勢が動いた。"
                f"当初は{label(leader_first.piece_id)}の主張が通りかけたが、"
                f"最後は{chosen_label}が{leader_last.move}を押し切った。"
            )

        if len(last) >= 2 and (last[0].bid - last[1].bid) < CLOSE_MARGIN:
            return (
                f"{chosen_label}の{leader_last.move}と"
                f"{label(last[1].piece_id)}の{last[1].move}が最後まで競り合う接戦。"
                f"僅差で{chosen_label}に軍配が上がった。"
            )

        return (
            f"{chosen_label}が{leader_last.move}を強く主張し、"
            f"{rounds}ラウンドを通して揺るがなかった。"
        )


__all__ = ["SPECIES_JA", "TemplateNarrator"]
