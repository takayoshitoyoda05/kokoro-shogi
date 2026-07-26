"""欲求ラベル (6軸) の事前計算 (DESIGN.md §3(3) の表)。

蒸留の補助損失 $L_{desire} = \\sum_{i,k} \\mathrm{BCE}(d^{(k)}_i(a_t), y^{(k)}_i)$ の
教師 $y$ を、棋譜から**ルールベースで自動生成**する。人手のアノテーションは要らない。

DESIGN.md の表は定義が一文なので、実装上の読み方をここで確定させる
(先読み幅は既定 $k=4$ 手):

| 軸 | DESIGN.md の定義 | この実装での判定 |
|---|---|---|
| survive  | k手で取られない | k手後に盤上にいるか |
| attack   | 捕獲/王手に関与 | k手以内にこの駒が捕獲または王手をかけたか |
| promote  | 4手以内に成る | k手以内にこの駒が成ったか |
| defend   | 味方(特に玉)への利き貢献 | 今の局面で紐を付けている味方の数 (玉は重み2) を正規化 |
| advance  | 敵陣への接近 | k手後に敵陣側へ進んでいるか |
| redeploy | (持ち駒専用) 打たれる | 持ち駒のとき、k手以内に打たれたか |

survive/attack/promote/advance/redeploy は 0/1、defend だけ [0,1] の連続値。
BCE はソフトターゲットを受け付けるので混在して問題ない。

トークンの並びは `core/tokenizer.py` が tracker 付きで作る並び (piece_id昇順) と
同一なので、ラベル配列はトークン配列とそのまま添字で対応する。
"""

from __future__ import annotations

from dataclasses import dataclass

import cshogi
import numpy as np

from kokoro_shogi.core.effects import attack_lists
from kokoro_shogi.core.piece_state import PieceIdTracker
from kokoro_shogi.core.squares import HAND_SQUARE, str_to_sq
from kokoro_shogi.core.tokenizer import MAX_PIECES

#: 欲求6軸。INTERFACE.md の desire フィールドと同じ並び
DESIRE_AXES: tuple[str, ...] = ("survive", "attack", "promote", "defend", "advance", "redeploy")
NUM_AXES = len(DESIRE_AXES)

SURVIVE, ATTACK, PROMOTE, DEFEND, ADVANCE, REDEPLOY = range(NUM_AXES)

#: 先読み幅 k (DESIGN.md §3(3))
DEFAULT_LOOKAHEAD = 4

#: defend の正規化に使う「これだけ守っていれば満点」の重み合計
DEFEND_SATURATION = 4.0
#: 玉を守る利きの重み
KING_DEFENSE_WEIGHT = 2.0


@dataclass(frozen=True)
class GameLabels:
    """1局ぶんの欲求ラベル。

    `labels[t, i, k]` = 局面 t (t手目を指す**前**) における駒 i の軸 k の教師。
    """

    labels: np.ndarray  # (T, MAX_PIECES, NUM_AXES) float32
    mask: np.ndarray  # (T, MAX_PIECES) bool
    piece_ids: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def _advancement(square: int, owner: int) -> float:
    """敵陣への進み具合 [0, 1]。自陣最奥が0。"""
    rank = square % 9 + 1
    return (9 - rank) / 8 if owner == 0 else (rank - 1) / 8


def _defend_scores(board: cshogi.Board, tracker: PieceIdTracker, order: list[str]) -> np.ndarray:
    """各駒が味方に付けている紐の量 [0, 1]。玉を守る利きは重い。"""
    scores = np.zeros(MAX_PIECES, dtype=np.float32)
    index_of = {piece_id: index for index, piece_id in enumerate(order)}
    lists = attack_lists(board)

    for square, targets in lists.items():
        piece_id = tracker.piece_id_at(square)
        if piece_id is None:
            continue
        state = tracker.get(piece_id)

        weight = 0.0
        for target in targets:
            target_id = tracker.piece_id_at(target)
            if target_id is None:
                continue
            other = tracker.get(target_id)
            if other.owner != state.owner:
                continue
            weight += KING_DEFENSE_WEIGHT if other.species == "OU" else 1.0

        scores[index_of[piece_id]] = min(weight / DEFEND_SATURATION, 1.0)

    return scores


def compute_desire_labels(
    moves: list[int],
    *,
    lookahead: int = DEFAULT_LOOKAHEAD,
    start_sfen: str | None = None,
) -> GameLabels:
    """棋譜1局から欲求ラベルを作る。

    `moves` は cshogi の指し手列。返るのは `len(moves)` 局面ぶんのラベル
    (最終局面は指し手が無いのでラベルを持たない)。
    """
    board = cshogi.Board()
    if start_sfen is not None:
        board.set_sfen(start_sfen)
    tracker = PieceIdTracker(board)

    order = sorted(tracker.states)
    index_of = {piece_id: index for index, piece_id in enumerate(order)}
    total = len(moves)

    # --- 1周目: 各局面のスナップショットと、指し手が何をしたかを記録する ---
    on_board = np.zeros((total + 1, MAX_PIECES), dtype=bool)
    advancement = np.zeros((total + 1, MAX_PIECES), dtype=np.float32)
    defend = np.zeros((total, MAX_PIECES), dtype=np.float32)

    mover = np.full(total, -1, dtype=np.int16)
    did_capture = np.zeros(total, dtype=bool)
    did_check = np.zeros(total, dtype=bool)
    did_promote = np.zeros(total, dtype=bool)
    did_drop = np.zeros(total, dtype=bool)

    def snapshot(ply: int) -> None:
        for piece_id, state in tracker.states.items():
            index = index_of[piece_id]
            if state.square == HAND_SQUARE:
                on_board[ply][index] = False
                advancement[ply][index] = 0.0
            else:
                on_board[ply][index] = True
                advancement[ply][index] = _advancement(str_to_sq(state.square), state.owner)

    for ply, move in enumerate(moves):
        snapshot(ply)
        defend[ply] = _defend_scores(board, tracker, order)

        record = tracker.apply_move(board, move)
        board.push(move)

        mover[ply] = index_of[record.piece_id]
        did_capture[ply] = record.capture
        did_promote[ply] = record.promote
        did_drop[ply] = record.drop
        did_check[ply] = board.is_check()

    snapshot(total)

    # --- 2周目: 先読みしてラベルを埋める ---
    labels = np.zeros((total, MAX_PIECES, NUM_AXES), dtype=np.float32)
    mask = np.ones((total, MAX_PIECES), dtype=bool)
    mask[:, len(order) :] = False

    for ply in range(total):
        horizon = min(ply + lookahead, total)

        labels[ply, :, SURVIVE] = on_board[horizon]
        labels[ply, :, DEFEND] = defend[ply]
        labels[ply, :, ADVANCE] = (advancement[horizon] > advancement[ply]).astype(np.float32)

        for future in range(ply, horizon):
            index = mover[future]
            if did_capture[future] or did_check[future]:
                labels[ply, index, ATTACK] = 1.0
            if did_promote[future]:
                labels[ply, index, PROMOTE] = 1.0
            # redeploy は「持ち駒だった駒が打たれる」ときだけ立てる
            if did_drop[future] and not on_board[ply][index]:
                labels[ply, index, REDEPLOY] = 1.0

    return GameLabels(labels=labels, mask=mask, piece_ids=tuple(order))
