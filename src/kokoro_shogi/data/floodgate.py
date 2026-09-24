"""floodgate の CSA 棋譜をパースして学習サンプルにする。

floodgate (wdoor) の棋譜は 1ファイル1局の CSA 形式で、
`http://wdoor.c.u-tokyo.ac.jp/shogi/archive/wdoorYYYY.7z` に年単位でまとまっている
(取得は scripts/download_floodgate.py)。

強さの経路を壊さないため、蒸留に使う棋譜は **強いエンジン同士の決着した対局** に
絞る (dlshogi の前処理と同じ考え方)。既定のしきい値はこのモジュールの定数を参照。

CSA のパースそのものは cshogi.Parser に任せ、このモジュールは
「どの棋譜を採用するか」と「学習に必要な形へ整えるか」だけを担当する。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cshogi

#: 両者にこのレート以上を要求する。floodgateはレート不明を 0.0 で出すので自動的に落ちる
DEFAULT_MIN_RATING = 3000.0
#: レート線形重み (2026-09-21、arXiv:2603.29761)。足切りで捨てるより、勾配の寄与を
#: 下げて残すほうが良い。3000 足切りは全体の 42.9% しか残さず、trunk が過学習している
#: (train 一致率 47.3% / val 41.5%、差が epoch ごとに拡大) いま、データ量が律速。
#: w(e) = clip((e - WEIGHT_FLOOR) / (WEIGHT_TOP - WEIGHT_FLOOR), WEIGHT_MIN, 1.0)
#: 論文の「線形・強さの比 20:1」に合わせる (指数 200:1 は検証損失が下がるのに性能が壊れた)
WEIGHT_FLOOR = 2000.0
WEIGHT_TOP = 4000.0
WEIGHT_MIN = 0.05
#: 短すぎる対局 (接続直後の投了など) を捨てる
DEFAULT_MIN_MOVES = 50
#: Max_Moves 到達などの異常に長い対局も捨てる
DEFAULT_MAX_MOVES = 400

#: 中断・不正手で終わった対局は勝敗が信用できない
REJECTED_ENDGAMES = frozenset({"%CHUDAN", "%ILLEGAL_MOVE", "%ILLEGAL_ACTION", "%ERROR"})


@dataclass(frozen=True)
class GameRecord:
    """1局ぶんの棋譜。"""

    path: Path
    start_sfen: str
    moves: tuple[int, ...]
    #: cshogi.BLACK_WIN(1) / WHITE_WIN(2) / DRAW(0)
    win: int
    endgame: str
    ratings: tuple[float, float]
    names: tuple[str, str]

    def __len__(self) -> int:
        return len(self.moves)

    @property
    def result_for_black(self) -> float:
        """先手から見た勝敗 z ∈ {+1, 0, -1} (DESIGN.md §1)。"""
        if self.win == cshogi.BLACK_WIN:
            return 1.0
        if self.win == cshogi.WHITE_WIN:
            return -1.0
        return 0.0

    def result_at(self, ply: int) -> float:
        """`ply` 手目を指す側から見た勝敗。value ヘッドの教師に使う。"""
        return self.result_for_black if ply % 2 == 0 else -self.result_for_black


#: 指し手行 `+7776FU` と消費時間行 `T12`
_MOVE_LINE = re.compile(rb"^[+-]\d{4}[A-Z]{2}\s*$")
_TIME_LINE = re.compile(rb"^T\d+\s*$")


def looks_wellformed(path: Path | str) -> bool:
    """cshogi のパーサに渡す前の構造検査 (2026-09-21)。

    CSA では消費時間行 `T…` は必ず指し手行の直後に来る。floodgate の古い棋譜には
    改行が落ちて指し手がコメント行の末尾に連結したものがあり
    (例: ``'rating:…+2726FU`` の直後に ``T1``)、そのまま `cshogi.Parser` に渡すと
    **SIGSEGV でプロセスごと落ちる**。Python 側では捕まえられないので、
    読む前にこの不変条件で弾く。
    """
    previous_was_move = False
    try:
        with Path(path).open("rb") as handle:
            for raw in handle:
                line = raw.rstrip(b"\r\n")
                if not line or line.startswith(b"'"):
                    continue
                if _TIME_LINE.match(line):
                    if not previous_was_move:
                        return False
                    previous_was_move = False
                else:
                    previous_was_move = bool(_MOVE_LINE.match(line))
    except OSError:
        return False
    return True


def parse_csa(path: Path | str) -> GameRecord | None:
    """CSAファイル1つを読む。壊れていれば None。"""
    file_path = Path(path)
    if not looks_wellformed(file_path):
        return None
    parser = cshogi.Parser()
    try:
        parser.parse_csa_file(str(file_path))
    except Exception:
        return None

    if not parser.moves:
        return None

    ratings = [float(value) for value in parser.ratings[:2]] + [0.0, 0.0]
    names = [str(value) for value in parser.names[:2]] + ["", ""]

    return GameRecord(
        path=file_path,
        start_sfen=str(parser.sfen),
        moves=tuple(int(move) for move in parser.moves),
        win=int(parser.win),
        endgame=str(parser.endgame or ""),
        ratings=(ratings[0], ratings[1]),
        names=(names[0], names[1]),
    )


def accept_game(
    record: GameRecord,
    *,
    min_rating: float = DEFAULT_MIN_RATING,
    min_moves: int = DEFAULT_MIN_MOVES,
    max_moves: int = DEFAULT_MAX_MOVES,
    allow_draw: bool = False,
) -> bool:
    """蒸留に使う棋譜かどうかを判定する。"""
    if record.endgame in REJECTED_ENDGAMES:
        return False
    if not allow_draw and record.win not in (cshogi.BLACK_WIN, cshogi.WHITE_WIN):
        return False
    if not min_moves <= len(record.moves) <= max_moves:
        return False
    if min(record.ratings) < min_rating:
        return False
    # 平手初期局面から始まらない棋譜は駒の初期IDが振れない (PieceIdTracker の前提)
    return record.start_sfen.split(" ")[0] == cshogi.STARTING_SFEN.split(" ")[0]


def rating_weight(
    record: GameRecord,
    *,
    floor: float = WEIGHT_FLOOR,
    top: float = WEIGHT_TOP,
    minimum: float = WEIGHT_MIN,
) -> float:
    """棋譜の学習重み ∈ [minimum, 1.0] (2026-09-21)。

    両者の低い方のレートを線形に写す。`floor` 以下は `minimum`、`top` 以上は 1.0。
    レート不明 (floodgate は 0.0 で出す) は重み付けできないので 0.0 を返し、
    呼び出し側が捨てる。
    """
    lowest = min(record.ratings)
    if lowest <= 0.0:
        return 0.0
    scaled = (lowest - floor) / (top - floor)
    return float(min(1.0, max(minimum, scaled)))


def iter_csa_files(root: Path | str) -> Iterator[Path]:
    """ディレクトリ以下の .csa を名前順に列挙する。"""
    return iter(sorted(Path(root).rglob("*.csa")))


def load_games(
    paths: Iterable[Path | str],
    *,
    min_rating: float = DEFAULT_MIN_RATING,
    min_moves: int = DEFAULT_MIN_MOVES,
    max_moves: int = DEFAULT_MAX_MOVES,
    allow_draw: bool = False,
    limit: int | None = None,
) -> Iterator[GameRecord]:
    """条件を満たす棋譜だけを順に返す。"""
    produced = 0
    for path in paths:
        if limit is not None and produced >= limit:
            return
        record = parse_csa(path)
        if record is None:
            continue
        if not accept_game(
            record,
            min_rating=min_rating,
            min_moves=min_moves,
            max_moves=max_moves,
            allow_draw=allow_draw,
        ):
            continue
        produced += 1
        yield record
