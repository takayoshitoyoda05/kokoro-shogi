"""ダミー対局のJSONL生成スクリプト (週1の納品物)。

目的は **Unity側 (U1/U2) がAI本体の完成を待たずに実装を始められること**。
盤面と指し手は cshogi のランダム合法手プレイアウトで作るので SFEN は必ず妥当で、
Unity の `BoardState.LoadSfen` に対する検証も兼ねる。

内面データ (mood / desire / alpha / relations) は **学習済みモデルの出力ではなく、
このスクリプト内のヒューリスティック**である。値域と分布が本物らしくなるように
作ってあるので、U2の感情シェーダや関係線VFXの見た目調整には使えるが、
将棋的な意味を読み取ってはいけない。本物は週6の実データ納品で差し替わる。

会議 (council) と実況 (narration) は configs/features.yaml の council が false の間、
INTERFACE.md §3 の規定どおり空配列 / 空文字で出す。

使い方::

    uv run python scripts/gen_sample_jsonl.py
    uv run python scripts/gen_sample_jsonl.py --games 3 --max-plies 60
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cshogi

from kokoro_shogi.config import REPO_ROOT, load_config
from kokoro_shogi.core.effects import attack_counts, attack_lists
from kokoro_shogi.core.piece_state import MoveRecord, PieceIdTracker, PieceState
from kokoro_shogi.core.pieces import BLACK, WHITE, Species
from kokoro_shogi.core.squares import str_to_sq
from kokoro_shogi.logging.jsonl import (
    CareerMessage,
    CareerMvp,
    CareerPiece,
    Desire,
    JsonlWriter,
    LastMove,
    Mood,
    PieceInfo,
    Relation,
    StateUpdate,
    to_json_line,
)

DEFAULT_GAMES = 10
DEFAULT_MAX_PLIES = 140

#: 形勢と発言力の重み付けに使う駒の価値
PIECE_VALUE: dict[Species, float] = {
    "FU": 1, "KY": 3, "KE": 3, "GI": 5, "KI": 6, "KA": 8, "HI": 10, "OU": 15,
    "TO": 6, "NY": 6, "NK": 6, "NG": 6, "UM": 12, "RY": 13,
}

#: これ以上は成れない駒
UNPROMOTABLE: frozenset[Species] = frozenset({"KI", "OU", "TO", "NY", "NK", "NG", "UM", "RY"})



def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def round3(value: float) -> float:
    """JSONの桁を抑える。Unity側は float なので3桁で十分。"""
    return round(value, 3)


def file_rank(square: int) -> tuple[int, int]:
    """cshogi のマス番号 → (筋, 段) の1始まり。"""
    return square // 9 + 1, square % 9 + 1


def distance(a: int, b: int) -> int:
    """2マスのチェビシェフ距離。"""
    file_a, rank_a = file_rank(a)
    file_b, rank_b = file_rank(b)
    return max(abs(file_a - file_b), abs(rank_a - rank_b))


def advancement(square: int, owner: int) -> float:
    """敵陣への進み具合 [0, 1]。自陣最奥が0、敵陣最奥が1。"""
    _, rank = file_rank(square)
    return (9 - rank) / 8 if owner == BLACK else (rank - 1) / 8


def material_eval(board: cshogi.Board, tracker: PieceIdTracker) -> float:
    """駒得だけの雑な形勢。**常に先手視点** (INTERFACE.md §3)。"""
    score = 0.0
    for state in tracker.states.values():
        value = PIECE_VALUE[state.species]
        if state.in_hand:
            value *= 0.9  # 持ち駒は少し割り引く
        score += value if state.owner == BLACK else -value
    # 除数が小さいと歩1枚の取り合いで形勢が振り切れる。飛車1枚差で 0.5 前後になる目安。
    return round3(math.tanh(score / 30.0))


class MoodHeuristics:
    """1局面ぶんの内面データを作る。学習済みモデルの代役。"""

    def __init__(self, board: cshogi.Board, tracker: PieceIdTracker, evaluation: float) -> None:
        self.board = board
        self.tracker = tracker
        self.evaluation = evaluation
        self.counts = attack_counts(board)
        self.lists = attack_lists(board)
        self.king_square = {
            BLACK: board.king_square(BLACK),
            WHITE: board.king_square(WHITE),
        }

    # --- 個々の指標 -------------------------------------------------------

    def _threat(self, state: PieceState, square: int) -> tuple[int, int]:
        """(敵の利き数, 味方の紐の数)。"""
        enemy = WHITE if state.owner == BLACK else BLACK
        return int(self.counts[enemy][square]), int(self.counts[state.owner][square])

    def _targets(self, state: PieceState, square: int) -> int:
        """その駒が取れる敵駒の数。"""
        enemy = WHITE if state.owner == BLACK else BLACK
        return sum(
            1
            for target in self.lists.get(square, ())
            if self.board.piece(target) != 0
            and self.tracker.get(self.tracker.piece_id_at(target)).owner == enemy  # type: ignore[arg-type]
        )

    def _king_distance(self, square: int, owner: int) -> int:
        king = self.king_square[owner]
        return distance(square, king) if king is not None and king >= 0 else 9

    def _enemy_proximity(self, state: PieceState, square: int) -> float:
        """一番近い敵駒との距離から作る「緊張」[0, 1]。隣接で1、5マス離れて0。

        取られる直前でなくても敵陣に近い駒は落ち着かない、という程度の味付け。
        これが無いと開戦前の駒が全て同じ見た目になり、U2の感情シェーダの確認に使えない。
        """
        nearest = min(
            (
                distance(square, str_to_sq(other.square))
                for other in self.tracker.states.values()
                if not other.in_hand and other.owner != state.owner
            ),
            default=9,
        )
        return clamp(1.0 - (nearest - 1) / 4.0)

    # --- 出力 -------------------------------------------------------------

    def mood(self, state: PieceState) -> Mood:
        if state.in_hand:
            # 持ち駒は盤上の危険から離れている。寝返った駒だけ気分が沈む [E]
            return Mood(
                fear=0.05,
                aggression=0.25,
                valence=round3(-0.3 if state.is_defector else 0.05),
            )

        square = str_to_sq(state.square)
        attacked, defended = self._threat(state, square)
        targets = self._targets(state, square)
        enemy = WHITE if state.owner == BLACK else BLACK

        fear = clamp(
            0.05
            + 0.25 * min(attacked, 3)
            - 0.08 * min(defended, 3)
            + (0.2 if attacked > defended else 0.0)
            + 0.25 * self._enemy_proximity(state, square)
        )
        aggression = clamp(
            0.15
            + 0.20 * min(targets, 2)
            + 0.40 * clamp(1.0 - self._king_distance(square, enemy) / 6.0)
        )
        side_eval = self.evaluation if state.owner == BLACK else -self.evaluation
        valence = clamp(side_eval - 0.4 * fear + 0.2 * aggression, -1.0, 1.0)

        return Mood(fear=round3(fear), aggression=round3(aggression), valence=round3(valence))

    def desire(self, state: PieceState, mood: Mood) -> Desire:
        if state.in_hand:
            return Desire(
                survive=1.0, attack=0.2, promote=0.0, defend=0.1, advance=0.0, redeploy=0.9
            )

        square = str_to_sq(state.square)
        forward = advancement(square, state.owner)
        promote = 0.0 if state.species in UNPROMOTABLE else clamp(forward * 1.2)

        return Desire(
            survive=round3(clamp(0.30 + 0.5 * mood.fear)),
            attack=round3(mood.aggression),
            promote=round3(promote),
            defend=round3(clamp(1.0 - self._king_distance(square, state.owner) / 6.0)),
            advance=round3(forward),
            redeploy=0.0,
        )

    def relations(self, state: PieceState) -> list[Relation]:
        """紐を付けている味方との絆。

        「近くにいる味方すべて」にすると1駒あたり5本×40駒で関係線が200本になり、
        U2の関係線VFXでは見分けがつかない。実際に取り返せる関係 (紐) だけに絞ると
        囲いの形がそのまま線として浮かび上がる。相互に support し合っていれば強い絆。
        """
        if state.in_hand:
            return []

        square = str_to_sq(state.square)
        candidates: list[tuple[float, str]] = []

        for target in self.lists.get(square, ()):
            target_id = self.tracker.piece_id_at(target)
            if target_id is None:
                continue
            other = self.tracker.get(target_id)
            if other.owner != state.owner:
                continue

            mutual = square in self.lists.get(target, ())
            strength = 0.9 if mutual else 0.6
            # 玉を守る関係は特別扱い (READMEの「金と玉の絆」)
            if other.species == "OU" or state.species == "OU":
                strength = clamp(strength + 0.1)
            candidates.append((strength, target_id))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [
            Relation(to=piece_id, r=round3(strength))
            for strength, piece_id in candidates[:5]
            if strength >= 0.3
        ]


def build_pieces(heuristics: MoodHeuristics, tracker: PieceIdTracker) -> list[PieceInfo]:
    """全40駒の PieceInfo を作る。alpha は総和1になるよう正規化する。"""
    moods = {state.piece_id: heuristics.mood(state) for state in tracker.states.values()}

    weights = {
        state.piece_id: PIECE_VALUE[state.species] * (0.5 + moods[state.piece_id].aggression)
        for state in tracker.states.values()
    }
    total = sum(weights.values()) or 1.0

    pieces: list[PieceInfo] = []
    for state in sorted(tracker.states.values(), key=lambda item: item.piece_id):
        mood = moods[state.piece_id]
        pieces.append(
            PieceInfo(
                piece_id=state.piece_id,
                species=state.species,
                owner=state.owner,
                square=state.square,
                mood=mood,
                desire=heuristics.desire(state, mood),
                alpha=round3(weights[state.piece_id] / total),
                relations=heuristics.relations(state),
            )
        )
    return pieces


def build_state_update(
    board: cshogi.Board,
    tracker: PieceIdTracker,
    ply: int,
    record: MoveRecord | None,
) -> StateUpdate:
    """`board` は指した **後** の局面。`record` は直前の1手 (ply=0 では None)。"""
    evaluation = material_eval(board, tracker)
    heuristics = MoodHeuristics(board, tracker, evaluation)

    last_move = None
    if record is not None:
        last_move = LastMove(
            **{"from": record.from_square},
            to=record.to_square,
            piece_id=record.piece_id,
            capture=record.capture,
            promote=record.promote,
            drop=record.drop,
        )

    return StateUpdate(
        ply=ply,
        sfen=board.sfen(),
        last_move=last_move,
        eval=evaluation,
        pieces=build_pieces(heuristics, tracker),
        council=[],      # 会議機能は features.yaml で false (INTERFACE.md §3)
        narration="",    # 実況も週6の実データ納品から
    )


def generate_game(
    path: Path, seed: int, max_plies: int
) -> tuple[dict[str, PieceState], int]:
    """1局を生成して JSONL に書き出す。戻り値は (終局時の駒の状態, 手数)。"""
    rng = random.Random(seed)
    board = cshogi.Board()
    tracker = PieceIdTracker(board)

    with JsonlWriter(path) as writer:
        writer.write(build_state_update(board, tracker, ply=0, record=None))

        ply = 0
        while ply < max_plies and not board.is_game_over():
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            record = tracker.apply_move(board, move)
            board.push(move)
            ply += 1
            writer.write(build_state_update(board, tracker, ply=ply, record=record))

    return tracker.states, ply


def build_career(
    results: list[dict[str, PieceState]], promotions: dict[str, int]
) -> CareerMessage:
    """生成した全局を集計して career メッセージを作る (INTERFACE.md §5)。"""
    games: dict[str, int] = {}
    survivals: dict[str, int] = {}
    species: dict[str, Species] = {}

    for states in results:
        for state in states.values():
            games[state.piece_id] = games.get(state.piece_id, 0) + 1
            species[state.piece_id] = state.species
            if not state.in_hand:  # 終局時に盤上に残っていれば生存
                survivals[state.piece_id] = survivals.get(state.piece_id, 0) + 1

    pieces = [
        CareerPiece(
            piece_id=piece_id,
            species=species[piece_id],
            games=count,
            survival_rate=round3(survivals.get(piece_id, 0) / count),
            promotions=promotions.get(piece_id, 0),
            mvp_count=0,  # MVPは Phase 5 の功績配分 (COMA) が出す
        )
        for piece_id, count in sorted(games.items())
    ]

    best = max(pieces, key=lambda piece: (piece.promotions, piece.survival_rate), default=None)
    mvp = (
        CareerMvp(piece_id=best.piece_id, contribution=round3(best.promotions + best.survival_rate))
        if best is not None
        else None
    )
    return CareerMessage(pieces=pieces, last_game_mvp=mvp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="生成する対局数")
    parser.add_argument(
        "--max-plies", type=int, default=DEFAULT_MAX_PLIES, help="1局の最大手数"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "sample_data", help="出力先ディレクトリ"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="乱数シード (既定: configs/base.yaml の seed)"
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else load_config().seed
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, PieceState]] = []
    promotions: dict[str, int] = {}

    for index in range(1, args.games + 1):
        path = args.out_dir / f"game{index:02d}.jsonl"
        states, plies = generate_game(path, seed=seed + index, max_plies=args.max_plies)
        results.append(states)
        for state in states.values():
            if state.is_promoted:
                promotions[state.piece_id] = promotions.get(state.piece_id, 0) + 1
        print(f"{path.name}: {plies + 1} レコード ({plies} 手)")

    career_path = args.out_dir / "career.json"
    career_path.write_text(
        json.dumps(json.loads(to_json_line(build_career(results, promotions))), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"{career_path.name}: {args.games} 局を集計")


if __name__ == "__main__":
    main()
