"""学習済みモデルの自己対戦から state_update JSONL を書き出す (週6実データ納品の中間版)。

`gen_sample_jsonl.py` (全部ヒューリスティックのダミー) と週6の完全版の中間に位置する:

- **本物 (モデル出力)**: 指し手 (方策からのサンプリング)、`eval` (valueヘッド)、
  `pieces[].desire` (欲求ヘッドを、その駒の合法手について方策確率で重み付け平均)、
  `pieces[].alpha` (単調mixingの発言力)
- **ダミー継続**: `mood` / `relations` は gen_sample_jsonl のヒューリスティックのまま
  (感情GRU [A] と関係性 [C] は Phase 3)。`council` / `narration` は機能OFFのため空。

欲求ヘッド $d_i(a)$ は手ごとの値なので、駒ごとの `desire` へは
$d_i = \\sum_{a \\in \\mathcal{A}_i} \\pi(a) d_i(a) / \\sum_{a \\in \\mathcal{A}_i} \\pi(a)$
と縮約する (「その駒が指しそうな手に込めた欲求」)。手番でない側の駒は合法手を
持たないため、手番を反転した盤面でもう1回 forward して同じ縮約を行う。

使い方 (GPU推奨だがCPUでも動く)::

    uv run python scripts/export_model_jsonl.py
    uv run python scripts/export_model_jsonl.py --games 3 --checkpoint checkpoints/desire_lambda0.01.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cshogi
import numpy as np
import torch

from gen_sample_jsonl import MoodHeuristics, build_career, round3
from kokoro_shogi.config import REPO_ROOT, load_config
from kokoro_shogi.core.effects import piece_effect_matrix
from kokoro_shogi.core.piece_state import MoveRecord, PieceIdTracker, PieceState
from kokoro_shogi.core.pieces import BLACK, base_species, SPECIES_ORDER, HAND_INDEX_TO_SPECIES
from kokoro_shogi.core.squares import NUM_SQUARES
from kokoro_shogi.core.tokenizer import MAX_PIECES, PieceTokenizer, TokenizedPosition, to_sfen
from kokoro_shogi.data.dataset import NUM_PROMOTE, legal_move_mask
from kokoro_shogi.data.labels import DESIRE_AXES
from kokoro_shogi.logging.jsonl import (
    Desire,
    JsonlWriter,
    LastMove,
    PieceInfo,
    StateUpdate,
    to_json_line,
)
from kokoro_shogi.model.policy import KokoroPolicy

DEFAULT_GAMES = 10
DEFAULT_MAX_PLIES = 140
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "desire_lambda0.1.pt"
DEFAULT_OUT_DIR = REPO_ROOT / "sample_data" / "model"

#: 自己対戦のサンプリング温度。0.1 (対局時の既定) だと10局がほぼ同じ進行になるので
#: 少し上げて散らす。argmax にしたいときは 0 を渡す。
DEFAULT_TAU = 0.25


def build_action_map(
    board: cshogi.Board, tokens: TokenizedPosition
) -> dict[tuple[int, int, int], int]:
    """(駒トークン, 移動先, 成り) → cshogi の move。

    対応付けの規則は data/dataset.legal_move_mask と同一
    (打ちは piece_id 最小の持ち駒トークン)。
    """
    square_to_token: dict[int, int] = {}
    hand_token: dict[str, int] = {}
    turn = int(board.turn)

    for index in range(MAX_PIECES):
        if not tokens.mask[index]:
            continue
        square = int(tokens.position[index])
        if square < NUM_SQUARES:
            square_to_token[square] = index
        elif int(tokens.owner[index]) == turn:
            held = base_species(SPECIES_ORDER[int(tokens.species[index])])
            hand_token.setdefault(held, index)

    mapping: dict[tuple[int, int, int], int] = {}
    for move in board.legal_moves:
        to_square = cshogi.move_to(move)
        promote = int(cshogi.move_is_promotion(move))
        if cshogi.move_is_drop(move):
            held = HAND_INDEX_TO_SPECIES[cshogi.move_drop_hand_piece(move)]
            token = hand_token.get(held)
        else:
            token = square_to_token.get(cshogi.move_from(move))
        if token is not None:
            mapping[(token, to_square, promote)] = move
    return mapping


class ModelRunner:
    """1局面ぶんのモデル出力 (desire / alpha / eval / 方策) を取り出す。"""

    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        config = load_config()
        self.model = KokoroPolicy.from_config(config, head="desire").to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.device = device
        self.tokenizer = PieceTokenizer()

    @torch.no_grad()
    def forward(self, board: cshogi.Board, tokens: TokenizedPosition):
        """モデルを1回走らせて (PolicyOutput, 合法手マスク) を返す。"""
        legal = legal_move_mask(
            board, tokens.position, tokens.owner, tokens.species, tokens.mask
        )
        effect = piece_effect_matrix(board, tokens.squares()).astype(np.int64)

        as_long = lambda array: torch.from_numpy(array.astype(np.int64))[None].to(self.device)
        output = self.model(
            species=as_long(tokens.species),
            position=as_long(tokens.position),
            owner=as_long(tokens.owner),
            promoted=as_long(tokens.promoted),
            mask=torch.from_numpy(tokens.mask)[None].to(self.device),
            turn=torch.tensor([tokens.turn], dtype=torch.long, device=self.device),
            effect=torch.from_numpy(effect)[None].to(self.device),
            legal=torch.from_numpy(legal)[None].to(self.device),
        )
        return output, legal

    def piece_desires(self, output, legal: np.ndarray) -> dict[int, np.ndarray]:
        """手番側の各駒トークンの desire 6軸。合法手を方策確率で重み付け平均する。"""
        probs = output.probs(tau=1.0)[0].view(MAX_PIECES, NUM_SQUARES, NUM_PROMOTE)
        desire = output.desire[0].view(MAX_PIECES, NUM_SQUARES, NUM_PROMOTE, len(DESIRE_AXES))

        result: dict[int, np.ndarray] = {}
        for index in range(MAX_PIECES):
            if not legal[index].any():
                continue
            weight = probs[index]  # (81, 2)。非合法手は softmax でほぼ0
            total = float(weight.sum())
            if total < 1e-9:
                # 方策がこの駒の手をまったく指す気がない場合は合法手の単純平均
                mask = torch.from_numpy(legal[index]).to(weight.device)
                weight = mask.to(weight.dtype)
                total = float(weight.sum())
            averaged = (weight.unsqueeze(-1) * desire[index]).sum(dim=(0, 1)) / total
            result[index] = averaged.cpu().numpy()
        return result


def desires_for_all_pieces(
    runner: ModelRunner, board: cshogi.Board, tokens: TokenizedPosition, output, legal: np.ndarray
) -> list[Desire]:
    """全40駒の desire。手番でない側は手番を反転した盤面でもう1回 forward する。"""
    by_token = runner.piece_desires(output, legal)

    flipped = TokenizedPosition(
        species=tokens.species,
        position=tokens.position,
        owner=tokens.owner,
        promoted=tokens.promoted,
        mask=tokens.mask,
        turn=1 - tokens.turn,
        move_number=tokens.move_number,
        piece_ids=tokens.piece_ids,
    )
    try:
        flipped_board = cshogi.Board(to_sfen(flipped))
        flipped_output, flipped_legal = runner.forward(flipped_board, flipped)
        for index, value in runner.piece_desires(flipped_output, flipped_legal).items():
            by_token.setdefault(index, value)
    except Exception:
        pass  # 反転局面が不正 (王手放置など) なら下のフォールバックに任せる

    desires: list[Desire] = []
    fallback = output.desire[0].mean(dim=1).cpu().numpy()  # (40, 6) 全162手の単純平均
    for index in range(MAX_PIECES):
        values = by_token.get(index, fallback[index])
        desires.append(
            Desire(**{axis: round3(float(np.clip(v, 0.0, 1.0))) for axis, v in zip(DESIRE_AXES, values)})
        )
    return desires


def build_state_update(
    runner: ModelRunner,
    board: cshogi.Board,
    tracker: PieceIdTracker,
    ply: int,
    record: MoveRecord | None,
    output,
    legal: np.ndarray,
    tokens: TokenizedPosition,
) -> StateUpdate:
    """`board` は指した後の局面。desire/alpha/eval がモデル出力になる。"""
    # value は手番側視点 (make_labels の z と同じ) なので先手視点へ直す
    value = float(output.value[0])
    evaluation = round3(value if tokens.turn == BLACK else -value)

    heuristics = MoodHeuristics(board, tracker, evaluation)
    desires = desires_for_all_pieces(runner, board, tokens, output, legal)
    alpha = output.alpha[0].cpu().numpy()

    pieces: list[PieceInfo] = []
    states = sorted(tracker.states.values(), key=lambda item: item.piece_id)
    for index, state in enumerate(states):  # トークンも piece_id 順なので index が一致
        mood = heuristics.mood(state)
        pieces.append(
            PieceInfo(
                piece_id=state.piece_id,
                species=state.species,
                owner=state.owner,
                square=state.square,
                mood=mood,
                desire=desires[index],
                alpha=round3(float(alpha[index])),
                relations=heuristics.relations(state),
            )
        )

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
        pieces=pieces,
        council=[],
        narration="",
    )


def sample_move(
    output, action_map: dict[tuple[int, int, int], int], tau: float, generator: torch.Generator
) -> int:
    """方策から1手選ぶ。tau=0 で argmax。"""
    logits = output.logits[0]
    if tau <= 0:
        index = int(logits.argmax())
    else:
        probs = torch.softmax(logits / tau, dim=-1)
        index = int(torch.multinomial(probs.cpu(), 1, generator=generator))
    promote = index % NUM_PROMOTE
    rest = index // NUM_PROMOTE
    token, to_square = rest // NUM_SQUARES, rest % NUM_SQUARES
    return action_map[(token, to_square, promote)]


def generate_game(
    runner: ModelRunner, path: Path, seed: int, max_plies: int, tau: float
) -> tuple[dict[str, PieceState], int]:
    """モデル同士の1局を生成して JSONL に書き出す。"""
    generator = torch.Generator().manual_seed(seed)
    board = cshogi.Board()
    tracker = PieceIdTracker(board)

    ply = 0
    record: MoveRecord | None = None
    with JsonlWriter(path) as writer:
        while True:
            tokens = runner.tokenizer.tokenize(board, tracker)
            output, legal = runner.forward(board, tokens)
            writer.write(
                build_state_update(runner, board, tracker, ply, record, output, legal, tokens)
            )
            if ply >= max_plies or board.is_game_over() or not legal.any():
                break
            move = sample_move(output, build_action_map(board, tokens), tau, generator)
            record = tracker.apply_move(board, move)
            board.push(move)
            ply += 1

    return tracker.states, ply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="自己対戦の温度 (0でargmax)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    seed = args.seed if args.seed is not None else load_config().seed
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runner = ModelRunner(args.checkpoint, device)
    print(f"checkpoint: {args.checkpoint.name} / device: {device}")

    results: list[dict[str, PieceState]] = []
    promotions: dict[str, int] = {}
    for index in range(1, args.games + 1):
        path = args.out_dir / f"game{index:02d}.jsonl"
        states, plies = generate_game(
            runner, path, seed=seed + index, max_plies=args.max_plies, tau=args.tau
        )
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
