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

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cshogi

#: 両者にこのレート以上を要求する。floodgateはレート不明を 0.0 で出すので自動的に落ちる
DEFAULT_MIN_RATING = 3000.0
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


def parse_csa(path: Path | str) -> GameRecord | None:
    """CSAファイル1つを読む。壊れていれば None。"""
    file_path = Path(path)
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
