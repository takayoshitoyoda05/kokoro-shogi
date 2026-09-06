"""自己対戦PPO (Phase 7-α、DESIGN.md §4b MAPPO)。

疎報酬 $r_T = z$、GAE ($\\gamma=1, \\lambda=0.95$)、PPOクリップ ($\\epsilon=0.2$)。
将棋は手番が交互の零和なので、価値・advantage は**手番側視点**で持ち、
後続手の値は符号反転して伝播する (negamax流):

$$\\delta_t = r_t + \\gamma\\,(-V(s_{t+1})) - V(s_t), \\qquad
  \\hat A_t = \\delta_t + \\gamma\\lambda\\,(-\\hat A_{t+1})$$

損失は統一損失 (DESIGN.md §4) の RL 形:

$$L = L_{\\text{PPO}} + c_1 (V - \\hat R)^2 + c_2 L_{\\text{desire}}
    + \\lambda_g L_g - c_3 H(\\pi)$$

$L_{desire}$ の教師は自己対戦の棋譜から `data/labels.compute_desire_labels` で
その場で作る (蒸留と同じ「全フェーズ並走」)。

割り切り (10週縮退版):

- **感情GRUは既定で凍結**し、mood/relation は観測の一部として扱う (BPTTなし。
  GRU自体の更新は系列蒸留 train/mood_distill.py の担当)。2026-09-06 追加の
  ``--train-gru`` (E4) は truncated BPTT (``--bptt`` 手) で感情を勾配付きに
  再計算して GRU も更新し、凍結 GRU の感情との MSE (``--anchor``) で繋ぎ止める
- 打ち切り局は引き分け (z=0)
- Gate は「兆候が見える」= 凍結した開始時スナップショットとの対戦勝率

使い方::

    uv run python -m kokoro_shogi.train.selfplay_ppo --iterations 10 --games-per-iter 16
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import cshogi
import numpy as np
import torch
from torch import Tensor, nn

from kokoro_shogi.config import Config, load_config, set_global_seed
from kokoro_shogi.core.effects import piece_effect_matrix
from kokoro_shogi.core.piece_state import PieceIdTracker
from kokoro_shogi.core.squares import NUM_SQUARES
from kokoro_shogi.core.tokenizer import MAX_PIECES, PieceTokenizer
from kokoro_shogi.data.dataset import NUM_PROMOTE, legal_move_mask
from kokoro_shogi.data.labels import compute_desire_labels
from kokoro_shogi.model.mood import NUM_EVENT_FEATURES, MoodGRU, build_event_features
from kokoro_shogi.model.policy import KokoroPolicy
from kokoro_shogi.model.relations import initial_relations, update_relations
from kokoro_shogi.train.distill import DEFAULT_OUT_DIR, Metrics, desire_bce, free_term_l2, resolve_device

DEFAULT_CHECKPOINT = DEFAULT_OUT_DIR / "phase34.pt"

GAMMA = 1.0
LAMBDA = 0.95
CLIP_EPSILON = 0.2


class SelfPlayEnv:
    """1面ぶんの自己対戦状態 (盤 + tracker + mood/relation)。"""

    def __init__(
        self,
        model: KokoroPolicy,
        gru: MoodGRU,
        device: torch.device,
        *,
        gru_ref: MoodGRU | None = None,
        bptt: int = 0,
    ) -> None:
        self.board = cshogi.Board()
        self.tracker = PieceIdTracker(self.board)
        self.tokenizer = PieceTokenizer()
        self.device = device
        self.mood = gru.initial_state(1, MAX_PIECES, device=device)
        # GRU 解凍 (E4) 用: 凍結 GRU の感情列を並走させる (相手側 / anchor の参照)。
        # bptt > 0 なら再計算に必要な (K 手前の状態, その後のイベント列) を観測に残す
        self.gru_ref = gru_ref
        self.mood_ref = gru.initial_state(1, MAX_PIECES, device=device) if gru_ref else None
        self.bptt = bptt
        self.past_moods: list[np.ndarray] = [self.mood[0].cpu().numpy()]
        self.past_events: list[np.ndarray] = []
        self.relation = (
            initial_relations(1, MAX_PIECES, device=device) if model.features.relations else None
        )
        self.record = None
        self.moves: list[int] = []
        self.steps: list[dict] = []
        self.done = False
        self.winner: int | None = None

    def observe(self, gru: MoodGRU) -> dict | None:
        """観測を作る (mood/relation の状態遷移もここで進む)。終局なら None。"""
        if self.done:
            return None
        if self.board.is_game_over():
            self.done = True
            self.winner = 1 - int(self.board.turn)
            return None

        with torch.no_grad():
            events = build_event_features(self.board, self.tracker, self.record)
            events_tensor = torch.from_numpy(events)[None].to(self.device)
            self.mood = gru(events_tensor, self.mood)
            if self.gru_ref is not None:
                self.mood_ref = self.gru_ref(events_tensor, self.mood_ref)
        if self.bptt:
            self.past_events.append(events)
            self.past_moods.append(self.mood[0].cpu().numpy())
        tokens = self.tokenizer.tokenize(self.board, self.tracker)
        effect = piece_effect_matrix(self.board, tokens.squares()).astype(np.int64)
        if self.relation is not None:
            self.relation = update_relations(
                self.relation, torch.from_numpy(effect)[None].to(self.device)
            )
        legal = legal_move_mask(
            self.board, tokens.position, tokens.owner, tokens.species, tokens.mask
        )
        if not legal.any():
            self.done = True
            self.winner = 1 - int(self.board.turn)
            return None

        obs = {
            "species": tokens.species.astype(np.int64),
            "position": tokens.position.astype(np.int64),
            "owner": tokens.owner.astype(np.int64),
            "promoted": tokens.promoted.astype(np.int64),
            "mask": tokens.mask.copy(),
            "turn": tokens.turn,
            "effect": effect,
            "legal": legal,
            "mood": self.mood[0].cpu().numpy(),
            "relation": self.relation[0].cpu().numpy() if self.relation is not None else None,
        }
        if self.mood_ref is not None:
            obs["mood_ref"] = self.mood_ref[0].cpu().numpy()
        if self.bptt:
            obs.update(self.bptt_window())
        self._tokens = tokens
        return obs

    def bptt_window(self) -> dict:
        """直近 K 手の (開始状態, イベント列, 有効長)。K 手に満たない序盤は前詰めゼロ埋め。"""
        total = len(self.past_events)
        length = min(self.bptt, total)
        events = np.zeros((self.bptt, MAX_PIECES, NUM_EVENT_FEATURES), dtype=np.float32)
        events[self.bptt - length :] = np.stack(self.past_events[total - length :])
        return {
            "mood_start": self.past_moods[total - length],
            "events_window": events,
            "window_len": length,
        }

    def step(self, action: int, log_prob: float, value: float, obs: dict) -> None:
        promote = action % NUM_PROMOTE
        rest = action // NUM_PROMOTE
        token, to_square = rest // NUM_SQUARES, rest % NUM_SQUARES
        move = self._action_move(token, to_square, promote)

        obs["action"] = action
        obs["log_prob"] = log_prob
        obs["value"] = value
        self.steps.append(obs)
        self.moves.append(move)
        self.record = self.tracker.apply_move(self.board, move)
        self.board.push(move)

    def _action_move(self, token: int, to_square: int, promote: int) -> int:
        from kokoro_shogi.core.pieces import HAND_INDEX_TO_SPECIES, base_species, SPECIES_ORDER

        tokens = self._tokens
        turn = int(self.board.turn)
        for move in self.board.legal_moves:
            if cshogi.move_to(move) != to_square:
                continue
            if int(cshogi.move_is_promotion(move)) != promote:
                continue
            if cshogi.move_is_drop(move):
                held = HAND_INDEX_TO_SPECIES[cshogi.move_drop_hand_piece(move)]
                candidate = None
                for index in range(MAX_PIECES):
                    if (
                        tokens.mask[index]
                        and int(tokens.position[index]) >= NUM_SQUARES
                        and int(tokens.owner[index]) == turn
                        and base_species(SPECIES_ORDER[int(tokens.species[index])]) == held
                    ):
                        candidate = index
                        break
                if candidate == token:
                    return move
            else:
                source = cshogi.move_from(move)
                if int(tokens.position[token]) == source:
                    return move
        raise ValueError(f"action がどの合法手にも対応しません: token={token} to={to_square}")


def negamax_gae(values: np.ndarray, turns: np.ndarray, winner: int | None) -> np.ndarray:
    """手番交互の零和ゲーム用GAE (docstringの式)。`(T,)` の advantage を返す。

    価値・報酬・advantage は全て**その手番側の視点**。後続の価値と advantage は
    相手視点なので符号を反転して伝播する。終局後の価値は0、報酬は最終手のみ
    ±1 (勝者視点で+1)。引き分け (winner=None) は報酬0。
    """
    total = len(values)
    advantages = np.zeros(total, dtype=np.float32)
    following = 0.0
    next_value = 0.0
    for t in reversed(range(total)):
        reward = 0.0
        if t == total - 1 and winner is not None:
            reward = 1.0 if int(turns[t]) == winner else -1.0
        delta = reward + GAMMA * (-next_value) - values[t]
        advantages[t] = delta + GAMMA * LAMBDA * (-following)
        following = advantages[t]
        next_value = values[t]
    return advantages


def batch_forward(
    model: KokoroPolicy,
    observations: list[dict],
    device: torch.device,
    *,
    mood: Tensor | None = None,
    mood_key: str = "mood",
):
    """観測のリストをまとめて forward。

    `mood` を渡すと観測に保存された感情の代わりに使う (GRU 解凍時の勾配付き再計算)。
    `mood_key` は凍結 GRU 側の感情列 (`"mood_ref"`) で指させるとき用。
    """
    stack = lambda key, dtype: torch.from_numpy(
        np.stack([o[key] for o in observations])
    ).to(device=device, dtype=dtype)
    kwargs = dict(
        species=stack("species", torch.long),
        position=stack("position", torch.long),
        owner=stack("owner", torch.long),
        promoted=stack("promoted", torch.long),
        mask=stack("mask", torch.bool),
        turn=torch.tensor([o["turn"] for o in observations], device=device),
        effect=stack("effect", torch.long),
        legal=stack("legal", torch.bool),
        mood=mood if mood is not None else stack(mood_key, torch.float32),
    )
    if observations[0]["relation"] is not None:
        kwargs["relation"] = stack("relation", torch.float32)
    return model(**kwargs)


def recompute_mood(gru: MoodGRU, batch: list[dict], device: torch.device) -> Tensor:
    """truncated BPTT: K 手前の状態からイベント列を再生し、感情 `(B, N, d)` を勾配付きで返す。

    序盤で有効長が K 未満のステップは、前詰めのゼロ埋め部分で状態を据え置く。
    """
    state = torch.from_numpy(np.stack([s["mood_start"] for s in batch])).to(device)
    events = torch.from_numpy(np.stack([s["events_window"] for s in batch])).to(device)
    lengths = torch.tensor([s["window_len"] for s in batch], device=device)
    window = events.shape[1]
    for k in range(window):
        updated = gru(events[:, k], state)
        valid = (k >= window - lengths)[:, None, None]
        state = torch.where(valid, updated, state)
    return state


@torch.no_grad()
def collect_games(
    model: KokoroPolicy,
    gru: MoodGRU,
    games: int,
    device: torch.device,
    *,
    tau: float,
    max_plies: int,
    generator: torch.Generator,
    opponents: list[KokoroPolicy] | None = None,
    opponent_prob: float = 0.5,
    gru_ref: MoodGRU | None = None,
    bptt: int = 0,
) -> tuple[list[dict], dict]:
    """並列自己対戦で軌跡を収集し、(学習側のステップ, 統計) を返す。

    `opponents` (過去スナップショットのプール) を渡すと、確率 `opponent_prob` で
    片側を過去の自分が受け持つ (自己対戦相手への過適合対策、AlphaStarリーグの
    最小版)。相手側の手は方策が違うため**学習データには含めない** (on-policy)。
    GAEは局全体で計算する (相手手番の価値は凍結された旧valueヘッドの近似)。

    `gru_ref` (凍結 GRU) を渡すと相手側はその感情列で指す (GRU 解凍時に相手が
    分布外の感情を見せられないため)。`bptt` > 0 で再計算用の窓を各ステップに残す。
    """
    model.eval()
    envs = [SelfPlayEnv(model, gru, device, gru_ref=gru_ref, bptt=bptt) for _ in range(games)]
    opponent_mood_key = "mood_ref" if gru_ref is not None else "mood"
    matchup: list[tuple[KokoroPolicy | None, int]] = []  # (相手モデル, 学習側の手番)
    for _ in envs:
        use_opponent = (
            opponents
            and float(torch.rand((), generator=generator)) < opponent_prob
        )
        if use_opponent:
            pick = int(torch.randint(len(opponents), (1,), generator=generator))
            matchup.append((opponents[pick], int(torch.randint(2, (1,), generator=generator))))
        else:
            matchup.append((None, -1))

    for _ply in range(max_plies):
        pairs = [
            (index, env, env.observe(gru)) for index, env in enumerate(envs) if not env.done
        ]
        pairs = [(index, env, obs) for index, env, obs in pairs if obs is not None]
        if not pairs:
            break
        # どのモデルが指す番かでグループ化してまとめて forward
        groups: dict[int, tuple[KokoroPolicy, list[tuple[int, SelfPlayEnv, dict]]]] = {}
        for index, env, obs in pairs:
            opponent, learner_side = matchup[index]
            if opponent is not None and obs["turn"] != learner_side:
                acting = opponent
            else:
                acting = model
            groups.setdefault(id(acting), (acting, []))[1].append((index, env, obs))

        for acting, members in groups.values():
            output = batch_forward(
                acting, [obs for _, _, obs in members], device,
                mood_key="mood" if acting is model else opponent_mood_key,
            )
            log_probs = output.log_probs(tau=1.0)
            probs = torch.softmax(output.logits / tau, dim=-1)
            actions = torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(-1)
            for row, (index, env, obs) in enumerate(members):
                opponent, learner_side = matchup[index]
                obs["learner"] = opponent is None or obs["turn"] == learner_side
                action = int(actions[row])
                env.step(action, float(log_probs[row, action]), float(output.value[row]), obs)

    steps: list[dict] = []
    decided = 0
    lengths = []
    for env in envs:
        if not env.done:
            env.winner = None  # 打ち切り = 引き分け
        if env.winner is not None:
            decided += 1
        lengths.append(len(env.steps))
        if not env.steps:
            continue

        values = np.array([s["value"] for s in env.steps], dtype=np.float32)
        turns = np.array([s["turn"] for s in env.steps], dtype=np.int64)
        advantages = negamax_gae(values, turns, env.winner)

        labels = compute_desire_labels(env.moves).labels  # (T, 40, 6)
        for t, step in enumerate(env.steps):
            if not step.get("learner", True):
                continue  # 相手側 (過去スナップショット) の手はoff-policyなので捨てる
            step["advantage"] = float(advantages[t])
            step["return"] = float(advantages[t] + values[t])
            step["labels"] = labels[t]
            steps.append(step)

    stats = {
        "games": games,
        "decided": decided,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }
    return steps, stats


def ppo_update(
    model: KokoroPolicy,
    optimizer: torch.optim.Optimizer,
    steps: list[dict],
    config: Config,
    device: torch.device,
    *,
    epochs: int,
    minibatch: int,
    generator: torch.Generator,
    gru: MoodGRU | None = None,
    gru_ref: MoodGRU | None = None,
    anchor: float = 0.0,
) -> dict:
    """収集した軌跡でPPO更新。

    `gru` を渡すと GRU も更新する (E4)。感情は保存値ではなく truncated BPTT で
    再計算し (`recompute_mood`)、`anchor` > 0 なら凍結 GRU `gru_ref` の感情との
    MSE を足して蒸留した感情空間 (MoodProjection の意味) から離れすぎないようにする。
    """
    model.train()
    if gru is not None:
        gru.train()
    stats = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "clip_frac": 0.0, "batches": 0}
    if gru is not None:
        stats["anchor"] = 0.0

    # advantage の正規化 (バッチ全体)
    adv = np.array([s["advantage"] for s in steps], dtype=np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for s, a in zip(steps, adv, strict=True):
        s["advantage_norm"] = float(a)

    indices = np.arange(len(steps))
    for _epoch in range(epochs):
        permutation = torch.randperm(len(steps), generator=generator).numpy()
        for start in range(0, len(steps), minibatch):
            rows = indices[permutation[start : start + minibatch]]
            batch = [steps[i] for i in rows]
            mood = recompute_mood(gru, batch, device) if gru is not None else None
            output = batch_forward(model, batch, device, mood=mood)

            actions = torch.tensor([s["action"] for s in batch], device=device)
            old_log_probs = torch.tensor([s["log_prob"] for s in batch], device=device)
            advantages = torch.tensor([s["advantage_norm"] for s in batch], device=device)
            returns = torch.tensor([s["return"] for s in batch], device=device)

            log_probs = output.log_probs(tau=1.0)
            picked = log_probs[torch.arange(len(batch), device=device), actions]
            ratio = torch.exp(picked - old_log_probs)
            clipped = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

            value_loss = nn.functional.mse_loss(output.value, returns)

            probs = torch.softmax(output.logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            aux_batch = {
                "action": actions,
                "labels": torch.from_numpy(np.stack([s["labels"] for s in batch])).to(
                    device=device, dtype=torch.float32
                ),
                "mask": torch.from_numpy(np.stack([s["mask"] for s in batch])).to(device),
                "legal": torch.from_numpy(np.stack([s["legal"] for s in batch])).to(device),
            }
            desire_loss = desire_bce(output, aux_batch)
            g_penalty = free_term_l2(output, aux_batch["legal"])

            loss = (
                policy_loss
                + config.loss.c1 * value_loss
                + config.loss.c2 * desire_loss
                + config.loss.lambda_g * g_penalty
                - config.loss.c3 * entropy
            )
            if gru is not None and anchor > 0 and gru_ref is not None:
                with torch.no_grad():
                    mood_ref = recompute_mood(gru_ref, batch, device)
                anchor_loss = nn.functional.mse_loss(mood, mood_ref)
                loss = loss + anchor * anchor_loss
                stats["anchor"] += float(anchor_loss.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if gru is not None:
                nn.utils.clip_grad_norm_(gru.parameters(), max_norm=1.0)
            optimizer.step()

            stats["policy"] += float(policy_loss.detach())
            stats["value"] += float(value_loss.detach())
            stats["entropy"] += float(entropy.detach())
            stats["clip_frac"] += float(
                ((ratio.detach() - 1).abs() > CLIP_EPSILON).float().mean()
            )
            stats["batches"] += 1

    for key in ("policy", "value", "entropy", "clip_frac", "anchor"):
        if key in stats:
            stats[key] /= max(stats["batches"], 1)
    if gru is not None:
        gru.eval()
    return stats


@torch.no_grad()
def play_match(
    challenger: KokoroPolicy,
    incumbent: KokoroPolicy,
    gru: MoodGRU,
    games: int,
    device: torch.device,
    *,
    tau: float = 0.1,
    max_plies: int = 200,
    seed: int = 0,
    gru_incumbent: MoodGRU | None = None,
) -> float:
    """challenger 対 incumbent の勝率 (先後を交互に持つ。引き分けは0.5)。

    `gru_incumbent` を渡すと incumbent はその GRU の感情列で指す (GRU 解凍時の公平な評価)。
    """
    generator = torch.Generator().manual_seed(seed)
    challenger.eval()
    incumbent.eval()
    score = 0.0

    for game in range(games):
        challenger_side = game % 2  # 偶数局は先手
        env = SelfPlayEnv(challenger, gru, device, gru_ref=gru_incumbent)
        for _ in range(max_plies):
            obs = env.observe(gru)
            if obs is None:
                break
            is_challenger = int(env.board.turn) == challenger_side
            model = challenger if is_challenger else incumbent
            mood_key = "mood" if is_challenger or gru_incumbent is None else "mood_ref"
            output = batch_forward(model, [obs], device, mood_key=mood_key)
            probs = torch.softmax(output.logits / tau, dim=-1)
            action = int(torch.multinomial(probs.cpu(), 1, generator=generator))
            env.step(action, 0.0, 0.0, obs)
        if env.winner is None and env.board.is_game_over():
            env.winner = 1 - int(env.board.turn)
        if env.winner is None:
            score += 0.5
        elif env.winner == challenger_side:
            score += 1.0
    return score / games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-name", default="ppo")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iter", type=int, default=16)
    parser.add_argument("--max-plies", type=int, default=180)
    parser.add_argument("--tau", type=float, default=0.8, help="収集時の温度 (探索)")
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument(
        "--snapshot-every", type=int, default=50,
        help="この間隔で凍結スナップショットを相手プールへ追加 (0で無効)",
    )
    parser.add_argument("--pool-size", type=int, default=5, help="相手プールの最大数")
    parser.add_argument(
        "--opponent-prob", type=float, default=0.5, help="片側を過去の自分が受け持つ確率"
    )
    parser.add_argument(
        "--culture-pool", type=Path, nargs="*", default=(),
        help="文化リーグの league.pt。各文化 (共有trunk + θ_sp) を凍結して相手プールに常駐させる",
    )
    parser.add_argument(
        "--train-gru", action="store_true",
        help="感情 GRU も更新する (E4)。truncated BPTT + 凍結 GRU への anchor 損失",
    )
    parser.add_argument("--bptt", type=int, default=8, help="GRU 再計算の窓 (手数)")
    parser.add_argument("--anchor", type=float, default=1.0, help="凍結 GRU の感情との MSE の重み")
    parser.add_argument("--gru-lr", type=float, default=None, help="GRU の学習率 (既定: --lr)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config()
    set_global_seed(config.seed)
    device = resolve_device(args.device)
    generator = torch.Generator().manual_seed(config.seed)

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    saved = state.get("features", {})
    flags = replace(
        config.features,
        mood=True,
        relations=bool(saved.get("relations", False)),
        council=bool(saved.get("council", False)),
    )
    model = KokoroPolicy(config.model, flags, tau=config.loss.tau, head="desire").to(device)
    model.load_state_dict(state["model"])
    gru = MoodGRU(config.model).to(device)
    gru.load_state_dict(state["mood_gru"])
    gru.eval()  # 既定では GRU は凍結 (docstring参照)。--train-gru のときも収集中は eval
    gru_ref: MoodGRU | None = None
    if args.train_gru:
        gru_ref = copy.deepcopy(gru)  # 凍結 GRU: 相手側の感情列 + anchor の参照
        for parameter in gru_ref.parameters():
            parameter.requires_grad_(False)
        gru_initial = [p.detach().clone() for p in gru.parameters()]

    snapshot = copy.deepcopy(model)  # Gate: 開始時スナップショットとの対戦勝率
    for parameter in snapshot.parameters():
        parameter.requires_grad_(False)
    snapshot.eval()
    pool: list[KokoroPolicy] = [snapshot]  # 相手プール (先頭は常に開始時)
    for league_path in args.culture_pool:
        # 文化リーグの成果を相手の多様性として使う (E5)。trunk はリーグ側のものを使う
        league_state = torch.load(league_path, map_location=device, weights_only=True)
        for culture in league_state["cultures"].values():
            member = KokoroPolicy(config.model, flags, tau=config.loss.tau, head="desire")
            member.to(device)
            member.load_state_dict(league_state["model"])
            member.personality.theta_species.weight.data.copy_(culture["theta_sp"].to(device))
            for parameter in member.parameters():
                parameter.requires_grad_(False)
            member.eval()
            pool.append(member)
        print(f"culture pool: {league_path} から {len(league_state['cultures'])} 文化")
    fixed = len(pool)  # 開始時 + 文化は落とさない。自分のスナップショットだけ入れ替える

    param_groups = [{"params": model.parameters()}]
    if args.train_gru:
        param_groups.append({"params": gru.parameters(), "lr": args.gru_lr or args.lr})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=1e-2)
    print(f"warm start: {args.checkpoint.name} / device: {device} / features: {flags}")
    if args.train_gru:
        gru_lr = args.gru_lr or args.lr
        print(f"train-gru: bptt {args.bptt} / anchor {args.anchor} / gru-lr {gru_lr}")

    history = []
    for iteration in range(1, args.iterations + 1):
        started = time.perf_counter()
        steps, collect_stats = collect_games(
            model, gru, args.games_per_iter, device,
            tau=args.tau, max_plies=args.max_plies, generator=generator,
            opponents=pool if args.snapshot_every else None,
            opponent_prob=args.opponent_prob,
            gru_ref=gru_ref, bptt=args.bptt if args.train_gru else 0,
        )
        if not steps:
            print(f"iter {iteration}: 軌跡が空 (全局即終了?)")
            continue
        update_stats = ppo_update(
            model, optimizer, steps, config, device,
            epochs=args.ppo_epochs, minibatch=args.minibatch, generator=generator,
            gru=gru if args.train_gru else None, gru_ref=gru_ref, anchor=args.anchor,
        )
        elapsed = time.perf_counter() - started

        entry = {"iteration": iteration, **collect_stats, **update_stats, "seconds": elapsed}
        if args.train_gru:
            # GRU の漂流量: 初期パラメータからの L2 距離 (anchor と合わせて崩壊の兆候を見る)
            entry["gru_delta"] = float(
                torch.sqrt(
                    sum(
                        ((p.detach() - p0) ** 2).sum()
                        for p, p0 in zip(gru.parameters(), gru_initial, strict=True)
                    )
                )
            )
        if iteration % args.eval_every == 0 or iteration == args.iterations:
            entry["win_rate_vs_start"] = play_match(
                model, snapshot, gru, args.eval_games, device, seed=config.seed + iteration,
                gru_incumbent=gru_ref,
            )
        if args.snapshot_every and iteration % args.snapshot_every == 0:
            frozen = copy.deepcopy(model)
            for parameter in frozen.parameters():
                parameter.requires_grad_(False)
            frozen.eval()
            pool.append(frozen)
            if len(pool) - fixed + 1 > args.pool_size:
                pool.pop(fixed)  # 開始時・文化は残し、自分のスナップショットの最古を落とす

        history.append(entry)
        win = entry.get("win_rate_vs_start")
        print(
            f"iter {iteration:>3}: {len(steps)}steps 決着{collect_stats['decided']}/{collect_stats['games']}"
            f" 平均{collect_stats['mean_length']:.0f}手 policy {update_stats['policy']:+.4f}"
            f" value {update_stats['value']:.4f} H {update_stats['entropy']:.2f}"
            f" clip {update_stats['clip_frac']:.2f}"
            + (
                f" anchor {update_stats['anchor']:.4f} Δgru {entry['gru_delta']:.3f}"
                if args.train_gru else ""
            )
            + (f" | 対開始時勝率 {win:.2f}" if win is not None else "")
            + f" ({elapsed:.0f}秒)"
        )

        torch.save(
            {
                "model": model.state_dict(),
                "mood_gru": gru.state_dict(),
                "mood_projection": state.get("mood_projection", {}),
                "features": saved,
                "ppo_iteration": iteration,
            },
            args.out_dir / f"{args.out_name}.pt",
        )
        (args.out_dir / f"{args.out_name}_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()


__all__ = [
    "SelfPlayEnv", "collect_games", "negamax_gae", "play_match", "ppo_update", "recompute_mood",
]
