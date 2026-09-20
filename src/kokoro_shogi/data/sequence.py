"""1局を通した系列サンプル (Phase 3 感情GRU [A] の学習入口)。

`ShardDataset` (data/dataset.py) は局面を独立に返すが、感情GRUの学習には
$m_i^{(t+1)} = \\mathrm{GRU}(u^{ev}_i(s_t, a_t), m_i^{(t)})$ を1局通して
回すための **時系列** が要る。シャードは局の境界で切れており (make_labels の
ShardBuilder が保証)、`move` 列に生の指し手が入っているので、
cshogi + PieceIdTracker で各局を再生してイベント特徴を正確に復元できる。

イベントの時刻合わせ: 行 $t$ (局面 $s_t$) に付くイベントは **$s_t$ を作った手**
$a_{t-1}$ のもの。つまりモデルが $s_t$ を見る時点の感情は
$m^{(t)} = \\mathrm{GRU}(u^{ev}(t), m^{(t-1)})$、$m^{(-1)} = 0$、
$u^{ev}(0)$ は初期局面の盤面項 (threatened等) のみ。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cshogi
import numpy as np

from kokoro_shogi.core.effects import piece_effect_matrix
from kokoro_shogi.core.piece_state import PieceIdTracker
from kokoro_shogi.core.squares import NUM_SQUARES
from kokoro_shogi.core.tokenizer import MAX_PIECES, PieceTokenizer
from kokoro_shogi.data.dataset import (
    NUM_MOVE_TO,
    NUM_PROMOTE,
    SHARD_COLUMNS,
    TEACHER_COLUMNS,
    TEACHER_K,
    action_index,
    legal_move_mask,
)
from kokoro_shogi.model.mood import NUM_EVENT_FEATURES, build_event_features


def _initial_row() -> dict[str, np.ndarray]:
    """平手初期局面のトークン行 (make_labels と同じ tracker 経由のトークン化)。"""
    board = cshogi.Board()
    tokens = PieceTokenizer().tokenize(board, PieceIdTracker(board))
    return {
        "species": tokens.species,
        "position": tokens.position,
        "owner": tokens.owner,
        "promoted": tokens.promoted,
    }


@dataclass(frozen=True)
class GameSequence:
    """1局ぶんの学習サンプル (numpy、全て先頭次元が手数 T)。"""

    species: np.ndarray  # (T, 40) int64
    position: np.ndarray  # (T, 40) int64
    owner: np.ndarray  # (T, 40) int64
    promoted: np.ndarray  # (T, 40) int64
    mask: np.ndarray  # (T, 40) bool
    turn: np.ndarray  # (T,) int64
    effect: np.ndarray  # (T, 40, 40) int64
    legal: np.ndarray  # (T, 81*... ) → (T, 40, 81, 2) bool
    action: np.ndarray  # (T,) int64
    result: np.ndarray  # (T,) float32
    labels: np.ndarray  # (T, 40, 6) float32
    events: np.ndarray  # (T, 40, 9) float32
    #: エンジン教師 (scripts/ops/join_teacher.py の側面ファイル)。無い手は NaN / -1
    teacher_value: np.ndarray  # (T,) float32
    teacher_actions: np.ndarray  # (T, K) int64
    teacher_cps: np.ndarray  # (T, K) float32

    @property
    def length(self) -> int:
        return len(self.turn)


class SequenceDataset:
    """npz シャード群を1局=1サンプルで返す Dataset。

    行の複製は持たず、シャードの列をそのまま保持して局の切れ目
    (`game_index` が変わる位置) だけを索引化する。`__getitem__` のたびに
    その局を cshogi で再生し、イベント特徴・合法手・利き行列を作る
    (1局 100〜200ms 程度。学習は系列単位なので許容)。
    """

    def __init__(self, paths: list[Path], *, max_games: int | None = None) -> None:
        if not paths:
            raise ValueError("シャードが1つも指定されていません。")

        columns: dict[str, list[np.ndarray]] = {
            name: [] for name in SHARD_COLUMNS + TEACHER_COLUMNS
        }
        boundaries: list[tuple[int, int]] = []
        offset = 0
        initial = _initial_row()
        #: 初期局面から始まらず捨てた断片の数 (0でなければデータ側の要調査サイン)
        self.dropped = 0
        #: エンジン教師の側面ファイルがあったシャード数
        self.teacher_shards = 0

        for path in paths:
            with np.load(path) as shard:
                index = shard["game_index"]
                # 局の始まり = game_index の変化点 ∪ 初期局面の行。
                # game_index は棋譜**ファイル内**の連番なので、1局だけのファイルが
                # 連続すると隣接する別の局が同じ番号になり、変化点だけでは融合する
                # (実データで確認済み)。全ての局は初期局面から始まる
                # (make_labels.encode_game) ので、初期局面の行を境界に加えると
                # この取りこぼしがなくなる
                changed = np.diff(index, prepend=index[0] - 1) != 0
                at_initial = np.logical_and.reduce(
                    [
                        (shard[name].reshape(len(index), -1) == initial[name].ravel()).all(axis=1)
                        for name in ("species", "position", "owner", "promoted")
                    ]
                ) & (shard["turn"] == 0)
                starts = np.flatnonzero(changed | at_initial)
                ends = np.append(starts[1:], len(index))
                for start, end in zip(starts, ends, strict=True):
                    # 初期局面から始まらない断片 (古いシャードの局跨ぎflush等) は
                    # 再生できないので捨てる
                    if not at_initial[start]:
                        self.dropped += 1
                        continue
                    boundaries.append((offset + int(start), offset + int(end)))
                for name in SHARD_COLUMNS:
                    columns[name].append(shard[name])
                count = len(index)
                offset += count
            side = path.with_suffix(".teacher.npz")
            if side.exists():
                with np.load(side) as teacher:
                    for name in TEACHER_COLUMNS:
                        columns[name].append(teacher[name][:count])
                self.teacher_shards += 1
            else:
                columns["teacher_value"].append(np.full(count, np.nan, dtype=np.float32))
                columns["teacher_actions"].append(
                    np.full((count, TEACHER_K), -1, dtype=np.int64)
                )
                columns["teacher_cps"].append(np.full((count, TEACHER_K), np.nan, dtype=np.float32))
            if max_games is not None and len(boundaries) >= max_games:
                break

        if max_games is not None:
            boundaries = boundaries[:max_games]
        self._columns = {name: np.concatenate(values) for name, values in columns.items()}
        self._boundaries = boundaries

    def __len__(self) -> int:
        return len(self._boundaries)

    def __getitem__(self, game: int) -> GameSequence:
        start, end = self._boundaries[game]
        rows = slice(start, end)
        length = end - start
        column = self._columns

        species = column["species"][rows].astype(np.int64)
        position = column["position"][rows].astype(np.int64)
        owner = column["owner"][rows].astype(np.int64)
        promoted = column["promoted"][rows].astype(np.int64)
        mask = column["mask"][rows]
        moves = column["move"][rows]

        effect = np.zeros((length, MAX_PIECES, MAX_PIECES), dtype=np.int64)
        legal = np.zeros((length, MAX_PIECES, NUM_MOVE_TO, NUM_PROMOTE), dtype=bool)
        events = np.zeros((length, MAX_PIECES, NUM_EVENT_FEATURES), dtype=np.float32)

        board = cshogi.Board()
        tracker = PieceIdTracker(board)
        record = None
        for t in range(length):
            events[t] = build_event_features(board, tracker, record)
            squares = [
                int(square) if flag and square < NUM_SQUARES else -1
                for square, flag in zip(position[t], mask[t], strict=True)
            ]
            try:
                effect[t] = piece_effect_matrix(board, squares).astype(np.int64)
                legal[t] = legal_move_mask(board, position[t], owner[t], species[t], mask[t])
                move = int(moves[t])
                record = tracker.apply_move(board, move)
                board.push(move)
            except (ValueError, KeyError) as error:
                # 再生盤面が行データとズレた = 局の境界検出の取りこぼし。
                # どの局か分かるように包み直す
                raise ValueError(
                    f"game {game} (行 {start + t}) で再生が行データと不整合: {error}"
                ) from error

        action = np.array(
            [
                action_index(
                    int(column["move_token"][start + t]),
                    int(column["move_to"][start + t]),
                    int(column["move_promote"][start + t]),
                )
                for t in range(length)
            ],
            dtype=np.int64,
        )

        return GameSequence(
            species=species,
            position=position,
            owner=owner,
            promoted=promoted,
            mask=mask,
            turn=column["turn"][rows].astype(np.int64),
            effect=effect,
            legal=legal,
            action=action,
            result=column["result"][rows].astype(np.float32),
            labels=column["labels"][rows].astype(np.float32) / 255.0,
            events=events,
            teacher_value=column["teacher_value"][rows].astype(np.float32),
            teacher_actions=column["teacher_actions"][rows].astype(np.int64),
            teacher_cps=column["teacher_cps"][rows].astype(np.float32),
        )

    def __iter__(self):
        for game in range(len(self)):
            yield self[game]


def collate_sequences(batch: list[GameSequence]) -> dict:
    """局のリストを最大手数へ0詰めした torch バッチにする。

    返す辞書の各テンソルは `(G, T, ...)`。`steps` `(G, T)` が有効手数のマスクで、
    詰め物の手は損失から除外する。
    """
    import torch

    longest = max(item.length for item in batch)

    def pad(name: str, dtype, fill: float = 0.0) -> torch.Tensor:
        arrays = []
        for item in batch:
            array = getattr(item, name)
            width = [(0, longest - item.length)] + [(0, 0)] * (array.ndim - 1)
            arrays.append(np.pad(array, width, constant_values=fill))
        return torch.from_numpy(np.stack(arrays)).to(dtype)

    steps = torch.zeros(len(batch), longest, dtype=torch.bool)
    for row, item in enumerate(batch):
        steps[row, : item.length] = True

    return {
        "species": pad("species", torch.long),
        "position": pad("position", torch.long),
        "owner": pad("owner", torch.long),
        "promoted": pad("promoted", torch.long),
        "mask": pad("mask", torch.bool),
        "turn": pad("turn", torch.long),
        "effect": pad("effect", torch.long),
        "legal": pad("legal", torch.bool),
        "action": pad("action", torch.long),
        "result": pad("result", torch.float32),
        "labels": pad("labels", torch.float32),
        "events": pad("events", torch.float32),
        # 教師の詰め物は「教師なし」を表す値にする (0 を入れると偽の教師になる)
        "teacher_value": pad("teacher_value", torch.float32, fill=np.nan),
        "teacher_actions": pad("teacher_actions", torch.long, fill=-1),
        "teacher_cps": pad("teacher_cps", torch.float32, fill=np.nan),
        "steps": steps,
    }


__all__ = ["GameSequence", "SequenceDataset", "collate_sequences"]
