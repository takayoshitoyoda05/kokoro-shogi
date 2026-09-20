"""駒ごとの感情状態を逐次更新する感情GRU (DESIGN.md §3(9) [A])。

$$m_i^{(t+1)} = \\mathrm{GRU}\\big(u^{ev}_i(s_t, a_t),\\, m_i^{(t)}\\big)$$

構成は3部品:

- `build_event_features`: 1手 (`MoveRecord`) と指した後の盤面から、駒ごとの
  イベント特徴 $u^{ev}_i \\in \\mathbb{R}^{9}$ を作る (torch非依存の前処理)。
  DESIGN.md の例 $u^{ev}_{i,1} = \\sum_{j\\in\\text{被取}} e^{-\\|p(i)-p(j)\\|_1/\\beta}$
  のとおり、駒取りイベントは距離減衰付きで全駒に伝わる —
  **事件現場に近い駒ほど強く動揺する**。
- `MoodGRU`: イベント特徴で感情状態 (d_mood次元) を1手ぶん更新する GRUCell。
- `MoodProjection`: 32次元の $m_i$ を INTERFACE.md §3 の3軸
  (fear, aggression ∈ [0,1] / valence ∈ [-1,+1]) へ射影する。Unityへ送るのは
  この出力で、$m_i$ 本体は PersonalityWeights と trunk 埋め込みに入る。

学習は Phase 3: 蒸留損失をゲーム内の局面列に沿って流す (シャードは棋譜順なので
`game_index` で系列を復元できる)。trunk 側の合流点 (`W_m m_i`) は零初期化なので、
Phase 2 チェックポイントからのウォームスタートを壊さない。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

from kokoro_shogi.config import ModelConfig
from kokoro_shogi.core.effects import attack_counts
from kokoro_shogi.core.piece_state import MoveRecord, PieceIdTracker
from kokoro_shogi.core.squares import str_to_sq
from kokoro_shogi.core.tokenizer import MAX_PIECES

if TYPE_CHECKING:
    import cshogi

#: 距離減衰の尺度 β (DESIGN.md §3(9))。L1距離2マスで 1/e
BETA = 2.0

#: イベント特徴の並び。教師なし (GRUへの入力のみ) なので追加は自由だが、
#: 並びを変えると学習済みGRUと非互換になる
EVENT_FEATURES = (
    "moved",           # 自分が動いた (打ちを含む)
    "captured",        # 自分が敵駒を取った
    "was_captured",    # 自分が取られて持ち駒になった
    "promoted",        # 自分が成った
    "ally_lost",       # 味方が取られた (距離減衰)
    "enemy_lost",      # 敵駒が取られた = 味方の戦果 (距離減衰)
    "threatened",      # 自分のマスへの敵の利き数 (0-3を正規化)
    "king_in_check",   # 自玉が王手をかけられている
    "material_swing",  # 駒得の増減 (自軍が得なら+、損なら−)。valence の形勢信号
)
NUM_EVENT_FEATURES = len(EVENT_FEATURES)

(
    MOVED,
    CAPTURED,
    WAS_CAPTURED,
    PROMOTED,
    ALLY_LOST,
    ENEMY_LOST,
    THREATENED,
    KING_IN_CHECK,
    MATERIAL_SWING,
) = range(NUM_EVENT_FEATURES)

#: 駒得スイングの重み付けに使う駒の価値 (取られた時点では生駒に戻っている)
_PIECE_VALUE = {"FU": 1, "KY": 3, "KE": 3, "GI": 5, "KI": 6, "KA": 8, "HI": 10, "OU": 0}
#: tanh(価値/8): 歩0.12 〜 飛0.85 におさめるスケール
_SWING_SCALE = 8.0


def _l1_distance(a: int, b: int) -> int:
    """cshogi マス番号2つのL1 (マンハッタン) 距離。"""
    return abs(a // 9 - b // 9) + abs(a % 9 - b % 9)


def build_event_features(
    board_after: cshogi.Board,
    tracker: PieceIdTracker,
    record: MoveRecord | None,
    *,
    max_pieces: int = MAX_PIECES,
) -> np.ndarray:
    """1手ぶんのイベント特徴 `(max_pieces, NUM_EVENT_FEATURES)` を作る。

    `board_after` / `tracker` は **指した後** の状態。トークンの並びは
    tokenizer と同じ piece_id 昇順。`record=None` (初期局面) では
    盤面由来の項 (threatened / king_in_check) だけが立つ。
    """
    features = np.zeros((max_pieces, NUM_EVENT_FEATURES), dtype=np.float32)
    states = sorted(tracker.states.values(), key=lambda item: item.piece_id)

    counts = attack_counts(board_after)
    #: 王手は「これから指す側の玉に掛かっている」(cshogi.is_check の定義)
    checked_owner = int(board_after.turn) if board_after.is_check() else None

    capture_square: int | None = None
    losing_owner: int | None = None
    swing = 0.0
    if record is not None and record.capture and record.captured_piece_id is not None:
        capture_square = str_to_sq(record.to_square)
        captured = tracker.get(record.captured_piece_id)
        # tracker は指した後の状態なので、取られた駒の owner は既に捕獲側。
        # 駒を失った側はその反対
        losing_owner = 1 - captured.owner
        swing = math.tanh(_PIECE_VALUE.get(captured.species, 5) / _SWING_SCALE)

    for index, state in enumerate(states):
        if index >= max_pieces:
            break

        if record is not None:
            if state.piece_id == record.piece_id:
                features[index, MOVED] = 1.0
                features[index, CAPTURED] = float(record.capture)
                features[index, PROMOTED] = float(record.promote)
            if state.piece_id == record.captured_piece_id:
                features[index, WAS_CAPTURED] = 1.0

        if state.in_hand:
            continue  # 持ち駒は盤上の出来事から切り離される (was_captured だけ届く)

        square = str_to_sq(state.square)
        enemy = 1 - state.owner
        features[index, THREATENED] = min(float(counts[enemy][square]), 3.0) / 3.0
        if checked_owner is not None and state.owner == checked_owner:
            features[index, KING_IN_CHECK] = 1.0

        if capture_square is not None and state.piece_id != record.captured_piece_id:
            decay = math.exp(-_l1_distance(square, capture_square) / BETA)
            axis = ALLY_LOST if state.owner == losing_owner else ENEMY_LOST
            features[index, axis] = decay
            # 形勢信号は距離に依らず陣営全体へ (valence の教師なし素材)
            features[index, MATERIAL_SWING] = -swing if state.owner == losing_owner else swing

    return features


class MoodGRU(nn.Module):
    """イベント特徴から感情状態 $m_i$ を1手ごとに更新する。

    全駒が同じGRUを共有する (駒ごとに分けると個体差が $\\theta^{ind}$ [B] と
    重複するため)。個体差は入力側の $u^{ev}_i$ と初期状態で生まれる。
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.d_mood = config.d_mood
        self.cell = nn.GRUCell(NUM_EVENT_FEATURES, self.d_mood)

    def initial_state(
        self, batch: int, tokens: int = MAX_PIECES, device: torch.device | None = None
    ) -> Tensor:
        """対局開始時の $m_i^{(0)} = 0$。`(B, N, d_mood)`。"""
        return torch.zeros(batch, tokens, self.d_mood, device=device)

    def forward(self, events: Tensor, state: Tensor) -> Tensor:
        """`events` `(B, N, 9)` と `state` `(B, N, d_mood)` → 更新後の状態。"""
        batch, tokens, _ = events.shape
        updated = self.cell(
            events.reshape(batch * tokens, NUM_EVENT_FEATURES),
            state.reshape(batch * tokens, self.d_mood),
        )
        return updated.view(batch, tokens, self.d_mood)


class MoodProjection(nn.Module):
    """$m_i$ → INTERFACE.md §3 の (fear, aggression, valence)。

    fear / aggression は sigmoid で [0,1]、valence は tanh で [-1,+1]。
    学習信号は感情そのものへの教師ではなく、$m_i$ が PersonalityWeights と
    trunk を経由して方策・価値に効くことで間接的に流れる (DRQN系のGRU stateと同じ)。
    """

    #: 出力軸の並び (INTERFACE.md の mood フィールドに対応)
    AXES = ("fear", "aggression", "valence")

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.project = nn.Linear(config.d_mood, len(self.AXES))

    def forward(self, mood: Tensor) -> Tensor:
        """`(B, N, d_mood)` → `(B, N, 3)` (fear, aggression, valence)。"""
        raw = self.project(mood)
        bounded_01 = torch.sigmoid(raw[..., :2])
        signed = torch.tanh(raw[..., 2:])
        return torch.cat([bounded_01, signed], dim=-1)


__all__ = [
    "BETA",
    "EVENT_FEATURES",
    "NUM_EVENT_FEATURES",
    "MoodGRU",
    "MoodProjection",
    "build_event_features",
]
