"""感情GRUの状態 $m_i$ から3軸 (fear/aggression/valence) への射影を回帰で当てる。

MoodProjection は蒸留損失に教師が無く、学習後も乱数のままになる (ADR 2026-08-25)。
そこで学習済みGRUで対局を再生しながら $(m_i, \\text{ヒューリスティックmood})$ の
ペアを集め、**リッジ回帰**で射影の重みを合わせる:

- fear / aggression: sigmoid リンク → 教師を logit 変換して線形回帰
- valence: tanh リンク → atanh 変換して線形回帰

教師のヒューリスティック (gen_sample_jsonl.MoodHeuristics) は「利きと距離から
見た危険・攻勢」の素朴な定義なので、これで得られるのは *ヒューリスティックの
GRU平滑化版* ではなく「**GRU状態のうちヒューリスティックと相関する読み出し**」。
軸の向きが人間の言葉 (恐怖・闘志・機嫌) に固定されるのが狙いで、R² が低い軸は
GRUがその概念を持っていないことを意味する (それ自体が計測値)。

結果は mood_distill.pt の `mood_projection` を**上書き**し、R² を表示する。

使い方::

    uv run python scripts/fit_mood_projection.py --games 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cshogi
import numpy as np
import torch

from gen_sample_jsonl import MoodHeuristics, material_eval
from kokoro_shogi.config import REPO_ROOT, load_config
from kokoro_shogi.core.piece_state import PieceIdTracker
from kokoro_shogi.core.tokenizer import MAX_PIECES
from kokoro_shogi.data.dataset import find_shards
from kokoro_shogi.data.sequence import SequenceDataset
from kokoro_shogi.model.mood import MoodGRU, MoodProjection
from kokoro_shogi.train.distill import DEFAULT_SHARD_DIR

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "mood_distill.pt"

#: リンク関数の発散を避けるため教師を [eps, 1-eps] に寄せる
EPS = 1e-3


def collect_pairs(
    dataset: SequenceDataset, gru: MoodGRU, games: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """(m_i の列 (K, d_mood), ヒューリスティック3軸 (K, 3)) を集める。"""
    states_out: list[np.ndarray] = []
    targets_out: list[np.ndarray] = []

    for game_index in range(min(games, len(dataset))):
        game = dataset[game_index]
        board = cshogi.Board()
        tracker = PieceIdTracker(board)
        mood = gru.initial_state(1, MAX_PIECES, device=device)

        moves = _game_moves(dataset, game_index)
        with torch.no_grad():
            for t in range(game.length):
                events = torch.from_numpy(game.events[t])[None].to(device)
                mood = gru(events, mood)

                # ヒューリスティック教師。valence は形勢に依存するので、
                # 駒得ベースの形勢 (material_eval) を渡す (0にすると教師の
                # valence から形勢成分が消え、GRUの形勢信号を測れない)
                heuristics = MoodHeuristics(board, tracker, material_eval(board, tracker))
                ordered = sorted(tracker.states.values(), key=lambda item: item.piece_id)
                target = np.array(
                    [
                        (lambda m: (m.fear, m.aggression, m.valence))(heuristics.mood(state))
                        for state in ordered
                    ],
                    dtype=np.float32,
                )

                states_out.append(mood[0].cpu().numpy())
                targets_out.append(target)

                move = moves[t]
                tracker.apply_move(board, move)
                board.push(move)

    return np.concatenate(states_out), np.concatenate(targets_out)


def _game_moves(dataset: SequenceDataset, game: int) -> list[int]:
    start, end = dataset._boundaries[game]  # noqa: SLF001 (再生のための内部参照)
    return [int(m) for m in dataset._columns["move"][start:end]]  # noqa: SLF001


def ridge_fit(features: np.ndarray, target: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """[X|1] w = y のリッジ解 (d_mood+1,)。"""
    design = np.concatenate([features, np.ones((len(features), 1), dtype=features.dtype)], axis=1)
    gram = design.T @ design + alpha * np.eye(design.shape[1], dtype=features.dtype)
    return np.linalg.solve(gram, design.T @ target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--games", type=int, default=64, help="回帰に使う対局数")
    parser.add_argument("--alpha", type=float, default=1.0, help="リッジの正則化")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = load_config()

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    gru = MoodGRU(config.model).to(device)
    gru.load_state_dict(state["mood_gru"])
    gru.eval()

    dataset = SequenceDataset(find_shards(args.shard_dir), max_games=args.games)
    features, targets = collect_pairs(dataset, gru, args.games, device)
    print(f"回帰サンプル: {len(features):,} 駒局面")

    projection = MoodProjection(config.model)
    weight = np.zeros((3, config.model.d_mood), dtype=np.float32)
    bias = np.zeros(3, dtype=np.float32)

    for axis, name in enumerate(MoodProjection.AXES):
        raw = np.clip(targets[:, axis], -1 + EPS, 1 - EPS)
        if name == "valence":
            linked = np.arctanh(raw)
        else:
            clipped = np.clip(raw, EPS, 1 - EPS)
            linked = np.log(clipped / (1 - clipped))

        solution = ridge_fit(features, linked, alpha=args.alpha)
        weight[axis], bias[axis] = solution[:-1], solution[-1]

        predicted = features @ solution[:-1] + solution[-1]
        residual = linked - predicted
        r2 = 1.0 - residual.var() / max(linked.var(), 1e-12)
        print(f"  {name:>10}: R² = {r2:.3f} (リンク空間)")

    with torch.no_grad():
        projection.project.weight.copy_(torch.from_numpy(weight))
        projection.project.bias.copy_(torch.from_numpy(bias))

    state["mood_projection"] = projection.state_dict()
    torch.save(state, args.checkpoint)
    print(f"{args.checkpoint.name} の mood_projection を更新しました。")


if __name__ == "__main__":
    main()
