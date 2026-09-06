"""対局後ループ [B] (DESIGN.md §4d): 自己対戦 → 功績配分 → θ_ind更新 → キャリア蓄積。

勾配ループ (蒸留/RL) の**外側**。trunk は凍結し、更新するのは個体性格 θ_ind だけ:

$$\\theta^{ind}_i \\leftarrow \\mathrm{clip}_{\\|\\cdot\\|\\le\\kappa}\\Big(
  \\theta^{ind}_i + \\eta_{ind}\\sum_{t:\\,\\text{駒}i\\text{が指した}}
  \\hat A_t\\,\\nabla_{\\theta^{ind}_i}\\log\\pi(a_t)\\Big)$$

功績 $\\hat A_t$ は **モンテカルロ advantage** $z_{\\text{手番}} - V(s_t)$ の近似
(DESIGN.md §9 の方針どおり、COMAの厳密なQは使わず $\\sum_t \\hat A_t$ の相対比較 =
MVP・貢献度にのみ使う)。credit は「その手を指した駒」に立てる。

実装は2パス: パス1は no_grad の自己対戦で軌跡 (トークン・mood・relation・V) を
収集し、パス2で全手をまとめて1回 forward して $\\log\\pi$ に勾配を流す
(1手ずつ計算グラフを保持すると140手ぶんの活性がVRAMに積もるため)。

結果は persist/store.py (SQLite) に蓄積され、`--career-out` に INTERFACE.md §5 の
career メッセージを書き出す (キャリアUIの実データ)。

忠誠 [E] は `--loyalty` でバリアントとして有効化 (ハンデ案: 寝返り駒の欲求を
$1-\\kappa_E$ 倍)。棋力評価には使わない。

使い方::

    uv run python scripts/selfplay_career.py --games 20 --checkpoint checkpoints/phase34.pt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cshogi
import numpy as np
import torch

from export_model_jsonl import build_action_map
from kokoro_shogi.config import REPO_ROOT, load_config
from kokoro_shogi.core.effects import piece_effect_matrix
from kokoro_shogi.core.piece_state import PieceIdTracker
from kokoro_shogi.core.tokenizer import MAX_PIECES, PieceTokenizer
from kokoro_shogi.data.dataset import NUM_PROMOTE, legal_move_mask
from kokoro_shogi.core.squares import NUM_SQUARES
from kokoro_shogi.model.mood import MoodGRU, build_event_features
from kokoro_shogi.model.policy import KokoroPolicy
from kokoro_shogi.model.relations import initial_relations, update_relations
from kokoro_shogi.persist.store import DEFAULT_DB, PieceStore

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "phase34.pt"
#: θ_ind の学習率 (η_ind ≪ η) と normクリップ κ (DESIGN.md §4d / §9)
ETA_IND = 0.02
KAPPA_NORM = 1.0


def load_policy(checkpoint: Path, device: torch.device, *, loyalty: bool) -> tuple:
    """チェックポイントを features.individual=True で読み込む。

    personality.project は [θ_sp | θ_ind | m] の順で連結されるため、
    学習済みの [θ_sp | m] 重みを対応する列へ移し、θ_ind の列には θ_sp の列を
    **複製**する (W_ind := W_sp)。これで w = softplus(W_sp(θ_sp + θ_ind) + W_m m) となり、
    θ_ind は「種の性格からの個体オフセット」として同じ空間で解釈できる。
    零初期化だと trunk 凍結下で ∂logπ/∂θ_ind ≡ 0 となり θ_ind が永遠に動かない
    (2026-09-04 に 230 局後も全駒ノルム 0 で発覚)。θ_ind 自体は 0 から始める。
    """
    config = load_config()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    saved = state.get("features", {})
    flags = replace(
        config.features,
        mood=True,
        relations=bool(saved.get("relations", False)),
        council=bool(saved.get("council", False)),
        individual=True,
        loyalty=loyalty,
    )
    model = KokoroPolicy(config.model, flags, tau=config.loss.tau, head="desire").to(device)

    own = model.state_dict()
    d_theta, d_mood = config.model.d_theta, config.model.d_mood
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state["model"].items():
        if key == "personality.project.weight" and value.shape[1] == d_theta + d_mood:
            widened = torch.zeros_like(own[key])
            widened[:, :d_theta] = value[:, :d_theta]  # θ_sp の列
            widened[:, d_theta : d_theta * 2] = value[:, :d_theta]  # θ_ind の列 = W_sp (勾配を生かす)
            widened[:, d_theta * 2 :] = value[:, d_theta:]  # m の列 (θ_ind ぶん右へ)
            remapped[key] = widened
        elif key in own and own[key].shape == value.shape:
            remapped[key] = value
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    assert not unexpected, unexpected
    model.eval()

    gru = MoodGRU(config.model).to(device)
    gru.load_state_dict(state["mood_gru"])
    gru.eval()
    return model, gru, flags


@torch.no_grad()
def play_one_game(model, gru, theta: torch.Tensor, device, tau: float, max_plies: int, seed: int):
    """パス1: 自己対戦して軌跡を集める。"""
    generator = torch.Generator().manual_seed(seed)
    board = cshogi.Board()
    tracker = PieceIdTracker(board)
    tokenizer = PieceTokenizer()

    mood = gru.initial_state(1, MAX_PIECES, device=device)
    relation = (
        initial_relations(1, MAX_PIECES, device=device) if model.features.relations else None
    )
    record = None
    steps: list[dict] = []

    while len(steps) < max_plies and not board.is_game_over():
        events = build_event_features(board, tracker, record)
        mood = gru(torch.from_numpy(events)[None].to(device), mood)
        tokens = tokenizer.tokenize(board, tracker)
        effect = piece_effect_matrix(board, tokens.squares()).astype(np.int64)
        if relation is not None:
            relation = update_relations(relation, torch.from_numpy(effect)[None].to(device))
        legal = legal_move_mask(board, tokens.position, tokens.owner, tokens.species, tokens.mask)
        if not legal.any():
            break

        loyalty_row = None
        if model.features.loyalty:
            states = sorted(tracker.states.values(), key=lambda item: item.piece_id)
            loyalty_row = np.array(
                [s.loyalty if s.is_defector else 0.0 for s in states], dtype=np.float32
            )

        step = {
            "species": tokens.species.astype(np.int64),
            "position": tokens.position.astype(np.int64),
            "owner": tokens.owner.astype(np.int64),
            "promoted": tokens.promoted.astype(np.int64),
            "mask": tokens.mask.copy(),
            "turn": tokens.turn,
            "effect": effect,
            "legal": legal,
            "mood": mood[0].cpu().numpy(),
            "relation": relation[0].cpu().numpy() if relation is not None else None,
            "loyalty": loyalty_row,
        }
        as_long = lambda a: torch.from_numpy(a)[None].to(device)
        output = model(
            species=as_long(step["species"]),
            position=as_long(step["position"]),
            owner=as_long(step["owner"]),
            promoted=as_long(step["promoted"]),
            mask=torch.from_numpy(step["mask"])[None].to(device),
            turn=torch.tensor([step["turn"]], device=device),
            effect=as_long(step["effect"]),
            legal=torch.from_numpy(step["legal"])[None].to(device),
            mood=mood,
            relation=relation,
            individual=theta[None],
            loyalty=torch.from_numpy(loyalty_row)[None].to(device)
            if loyalty_row is not None
            else None,
        )
        logits = output.logits[0]
        if tau <= 0:
            action = int(logits.argmax())
        else:
            action = int(
                torch.multinomial(torch.softmax(logits / tau, dim=-1).cpu(), 1, generator=generator)
            )
        step["action"] = action
        step["value"] = float(output.value[0])

        promote = action % NUM_PROMOTE
        rest = action // NUM_PROMOTE
        token, to_square = rest // NUM_SQUARES, rest % NUM_SQUARES
        step["mover"] = token
        move = build_action_map(board, tokens)[(token, to_square, promote)]
        record = tracker.apply_move(board, move)
        board.push(move)
        steps.append(step)

    if board.is_game_over():
        winner = 1 - int(board.turn)  # 手番側が詰まされている
    else:
        winner = None  # 打ち切り = 引き分け扱い
    return steps, winner, tracker


def batched_log_probs(model, steps: list[dict], theta: torch.Tensor, device) -> torch.Tensor:
    """パス2: 全手をまとめて forward し、指した手の logπ `(T,)` を返す (θに勾配)。"""
    stack = lambda key, dtype: torch.from_numpy(
        np.stack([s[key] for s in steps])
    ).to(device=device, dtype=dtype)

    kwargs = dict(
        species=stack("species", torch.long),
        position=stack("position", torch.long),
        owner=stack("owner", torch.long),
        promoted=stack("promoted", torch.long),
        mask=stack("mask", torch.bool),
        turn=torch.tensor([s["turn"] for s in steps], device=device),
        effect=stack("effect", torch.long),
        legal=stack("legal", torch.bool),
        mood=stack("mood", torch.float32),
        individual=theta.unsqueeze(0).expand(len(steps), -1, -1),
    )
    if steps[0]["relation"] is not None:
        kwargs["relation"] = stack("relation", torch.float32)
    if steps[0]["loyalty"] is not None:
        kwargs["loyalty"] = stack("loyalty", torch.float32)

    output = model(**kwargs)
    log_probs = output.log_probs(tau=1.0)
    actions = torch.tensor([s["action"] for s in steps], device=device)
    return log_probs[torch.arange(len(steps), device=device), actions]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--tau", type=float, default=0.4, help="自己対戦の温度 (探索用に高め)")
    parser.add_argument("--eta", type=float, default=ETA_IND)
    parser.add_argument("--loyalty", action="store_true", help="忠誠ハンデ [E] バリアント")
    parser.add_argument("--career-out", type=Path, default=REPO_ROOT / "sample_data" / "model" / "career.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = args.seed if args.seed is not None else load_config().seed
    model, gru, flags = load_policy(args.checkpoint, device, loyalty=args.loyalty)
    print(f"checkpoint: {args.checkpoint.name} / device: {device} / loyalty: {args.loyalty}")

    store = PieceStore(args.db)
    last_mvp: tuple[str, float] | None = None

    for game in range(1, args.games + 1):
        board = cshogi.Board()
        tracker = PieceIdTracker(board)
        piece_ids = sorted(tracker.states)
        species_map = {s.piece_id: s.species for s in tracker.states.values()}
        theta_np = store.load_theta(piece_ids, species_map)
        theta = torch.tensor(theta_np, device=device, requires_grad=True)

        steps, winner, tracker = play_one_game(
            model, gru, theta.detach(), device, args.tau, args.max_plies, seed + game
        )
        if not steps:
            continue

        # モンテカルロ advantage (手番側視点、γ=1)
        advantages = np.array(
            [
                (0.0 if winner is None else (1.0 if s["turn"] == winner else -1.0)) - s["value"]
                for s in steps
            ],
            dtype=np.float32,
        )

        # θ_ind 更新: J = Σ_t Â_t logπ(a_t) の勾配上昇 + normクリップ
        log_probs = batched_log_probs(model, steps, theta, device)
        objective = (torch.from_numpy(advantages).to(device) * log_probs).sum()
        objective.backward()
        with torch.no_grad():
            updated = theta + args.eta * theta.grad
            norms = updated.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            updated = updated * (norms.clamp(max=KAPPA_NORM) / norms)
        store.save_theta({pid: vec for pid, vec in zip(piece_ids, updated.cpu().numpy())})

        # 功績配分 → MVP・キャリア
        credit: dict[str, float] = {pid: 0.0 for pid in piece_ids}
        for step, adv in zip(steps, advantages, strict=True):
            credit[piece_ids[step["mover"]]] += float(adv)
        mvp_id = max(credit, key=lambda pid: credit[pid])
        store.record_game(
            survived={s.piece_id: not s.in_hand for s in tracker.states.values()},
            promoted={s.piece_id: s.is_promoted for s in tracker.states.values()},
            contribution=credit,
            mvp_id=mvp_id,
        )
        last_mvp = (mvp_id, credit[mvp_id])
        result = "先手勝ち" if winner == 0 else "後手勝ち" if winner == 1 else "引き分け"
        print(f"game {game:>3}: {len(steps)}手 {result} / MVP {mvp_id} (+{credit[mvp_id]:.2f})")

    careers = store.careers()
    message = {
        "schema": "1.0",
        "type": "career",
        "pieces": [
            {
                "piece_id": c["piece_id"],
                "species": c["species"],
                "games": c["games"],
                "survival_rate": round(c["survival_rate"], 3),
                "promotions": c["promotions"],
                "mvp_count": c["mvp_count"],
            }
            for c in careers
        ],
        "last_game_mvp": (
            {"piece_id": last_mvp[0], "contribution": round(last_mvp[1], 3)} if last_mvp else None
        ),
    }
    args.career_out.parent.mkdir(parents=True, exist_ok=True)
    args.career_out.write_text(json.dumps(message, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.career_out} を更新 ({len(careers)} 駒)")
    store.close()


if __name__ == "__main__":
    main()
