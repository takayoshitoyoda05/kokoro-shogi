"""floodgate棋譜からの蒸留学習 (DESIGN.md §4a)。

$$L_{\\text{policy}} = -\\log\\pi(a^\\ast\\mid s), \\qquad L_{\\text{value}} = (V(s)-z)^2$$

統一損失 (DESIGN.md §4) は

$$L = L_{\\text{policy}} + c_1 L_{\\text{value}} + c_2 L_{\\text{desire}}
      + \\lambda_g L_g - c_3 H(\\pi)$$

で、**Phase 1 で有効なのは前2項だけ**。$L_{desire}$ と $L_g$ は欲求ヘッド・自由項が
入る Phase 2 から、$H(\\pi)$ のエントロピーボーナスは自己対戦RL (Phase 7) から効く。

Gate1 は「一致率が CNN ベースライン比 -2%以内」。同じデータ・同じ損失で
`KokoroPolicy` と `BaselineCNN` を学習し、`evaluate` の一致率で比べる::

    uv run python -m kokoro_shogi.train.distill --model kokoro --epochs 2
    uv run python -m kokoro_shogi.train.distill --model cnn    --epochs 2

学習を回すマシンにGPUがあれば `--device cuda`。既定は自動判定。
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from kokoro_shogi.config import REPO_ROOT, Config, load_config, set_global_seed
from kokoro_shogi.data.dataset import ShardDataset, collate, find_shards, split_shards
from kokoro_shogi.model.baseline_cnn import BaselineCNN
from kokoro_shogi.model.policy import HEAD_KINDS, KokoroPolicy, PolicyOutput

DEFAULT_SHARD_DIR = REPO_ROOT / "data" / "shards"
DEFAULT_OUT_DIR = REPO_ROOT / "checkpoints"


#: 平均を取る対象 (positions で重み付けする)
_AVERAGED = (
    "loss",
    "policy_loss",
    "value_loss",
    "desire_loss",
    "free_term_penalty",
    "accuracy",
    "explained_ratio",
)


@dataclass
class Metrics:
    """1エポック (または検証1回) の集計。"""

    loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    desire_loss: float = 0.0
    free_term_penalty: float = 0.0
    accuracy: float = 0.0
    #: 欲求説明率 E<w,d>² / E s² (DESIGN.md §4a)。head="desire" のときだけ
    explained_ratio: float = 0.0
    positions: int = 0

    def update(self, other: Metrics) -> None:
        total = self.positions + other.positions
        if not total:
            return
        for name in _AVERAGED:
            merged = (
                getattr(self, name) * self.positions + getattr(other, name) * other.positions
            ) / total
            setattr(self, name, merged)
        self.positions = total

    def format(self) -> str:
        text = (
            f"loss {self.loss:.4f} (policy {self.policy_loss:.4f} / "
            f"value {self.value_loss:.4f}"
        )
        if self.desire_loss:
            text += f" / desire {self.desire_loss:.4f} / g² {self.free_term_penalty:.4f}"
        text += f")  一致率 {self.accuracy * 100:.2f}%"
        if self.explained_ratio:
            text += f"  説明率 {self.explained_ratio * 100:.1f}%"
        return text


def resolve_device(name: str | None = None) -> torch.device:
    """`--device` の指定を解決する。既定はCUDAがあれば使う。"""
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
    kind: str, config: Config, *, channels: int = 192, blocks: int = 10, head: str = "plain"
) -> nn.Module:
    if kind == "kokoro":
        return KokoroPolicy.from_config(config, head=head)
    if kind == "cnn":
        return BaselineCNN.from_config(config, channels=channels, blocks=blocks)
    raise ValueError(f"model は kokoro / cnn のどちらかです: {kind}")


def compute_loss(
    output: PolicyOutput, batch: dict[str, Tensor], config: Config
) -> tuple[Tensor, Metrics]:
    """統一損失 (DESIGN.md §4) のうち、蒸留で有効な項を足して返す。

    $$L = L_{policy} + c_1 L_{value} + c_2 L_{desire} + \\lambda_g L_g$$

    `head="plain"` なら前2項のみ。エントロピーボーナス $-c_3 H(\\pi)$ は
    自己対戦RL (Phase 7) で入る。
    """
    action = batch["action"]
    policy_loss = nn.functional.cross_entropy(output.logits, action)
    value_loss = nn.functional.mse_loss(output.value, batch["result"])
    loss = policy_loss + config.loss.c1 * value_loss

    desire_loss = torch.zeros((), device=action.device)
    free_term_penalty = torch.zeros((), device=action.device)
    explained_ratio = 0.0

    if output.desire is not None:
        desire_loss = desire_bce(output, batch)
        free_term_penalty = free_term_l2(output, batch["legal"])
        loss = loss + config.loss.c2 * desire_loss + config.loss.lambda_g * free_term_penalty
        with torch.no_grad():
            explained_ratio = explanation_ratio(output, batch["legal"])

    with torch.no_grad():
        accuracy = (output.logits.argmax(dim=-1) == action).float().mean()

    return loss, Metrics(
        loss=float(loss.detach()),
        policy_loss=float(policy_loss.detach()),
        value_loss=float(value_loss.detach()),
        desire_loss=float(desire_loss.detach()),
        free_term_penalty=float(free_term_penalty.detach()),
        accuracy=float(accuracy),
        explained_ratio=explained_ratio,
        positions=int(action.shape[0]),
    )


def desire_bce(output: PolicyOutput, batch: dict[str, Tensor]) -> Tensor:
    """$L_{desire} = \\sum_{i,k}\\mathrm{BCE}(d^{(k)}_i(a_t), y^{(k)}_i)$ (DESIGN.md §4)。

    教師ラベル `labels` は「局面 t における駒 i の欲求」で手には依存しない。
    一方 $d_i(a)$ は手ごとに出るので、**実際に指された手 $a_t$ の (移動先, 成り) を
    全駒に当てて**教師と突き合わせる (DESIGN.md の式が $d^{(k)}_i(a_t)$ と
    指し手を固定しているのと同じ形)。

    6軸は足し合わせ、駒は有効トークンで平均する。
    """
    desire = output.desire  # (B, N, 162, 6)
    batch_size, tokens, move_kinds, _ = desire.shape

    # action index = token * 162 + to * 2 + promote → 手成分 (0-161) を取り出す
    move_kind = batch["action"] % move_kinds
    picked = desire[
        torch.arange(batch_size, device=desire.device)[:, None],
        torch.arange(tokens, device=desire.device)[None, :],
        move_kind[:, None].expand(batch_size, tokens),
    ]  # (B, N, 6)

    loss = nn.functional.binary_cross_entropy(picked, batch["labels"], reduction="none")
    weights = batch["mask"].to(loss.dtype)
    return (loss.sum(dim=-1) * weights).sum() / weights.sum().clamp(min=1.0)


def free_term_l2(output: PolicyOutput, legal: Tensor) -> Tensor:
    """$L_g = \\frac{1}{|\\mathcal{A}|}\\sum_{i,a} g_i(a)^2$ (DESIGN.md §4)。

    合法手だけで平均する。非合法手の $g$ を罰しても意味がないうえ、
    局面ごとの合法手数の違いが罰則の強さに化けるのを防げる。
    """
    selected = legal.flatten(start_dim=1).view_as(output.free_term)
    if not bool(selected.any()):
        return torch.zeros((), device=output.free_term.device)
    return output.free_term[selected].square().mean()


def explanation_ratio(output: PolicyOutput, legal: Tensor) -> float:
    """欲求説明率 $\\mathbb{E}\\langle w_i,d_i\\rangle^2 / \\mathbb{E}\\,s_{i,a}^2$。

    DESIGN.md §4a の評価指標。大きいほど「手の選択が欲求で説明できている」、
    小さいほど自由項 $g$ (大局観) に頼っている。$\\lambda_g$ を振ると
    この値と一致率のトレードオフ曲線が描ける (論文の主図)。

    **100%を超えることがある**。$s = \\langle w,d\\rangle + g$ なので、$g$ が
    $\\langle w,d\\rangle$ と逆相関すると分母 $\\mathbb{E}s^2$ のほうが小さくなる。
    比であって割合ではないので、1で頭打ちにはならない (バグではない)。
    """
    explained = (output.desire * output.personality.unsqueeze(2)).sum(dim=-1)
    scores = explained + output.free_term

    selected = legal.flatten(start_dim=1).view_as(explained)
    if not bool(selected.any()):
        return 0.0

    numerator = explained[selected].square().mean()
    denominator = scores[selected].square().mean().clamp(min=1e-12)
    return float(numerator / denominator)


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: Iterable[dict[str, Tensor]], config: Config, device: torch.device
) -> Metrics:
    """検証セットの損失と一致率 (Gate1 の判定に使う数字)。"""
    model.eval()
    total = Metrics()
    for batch in loader:
        batch = move_batch(batch, device)
        output = model.forward_batch(batch)
        _, metrics = compute_loss(output, batch, config)
        total.update(metrics)
    return total


def build_scheduler(
    optimizer: torch.optim.Optimizer, *, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """線形ウォームアップ + コサイン減衰。

    Transformer はウォームアップなしの定数学習率だと序盤が不安定になりやすく、
    CNNベースラインとの比較 (Gate1) が「学習率の相性」で決まってしまう。
    両モデルに同じスケジュールを掛けることで、比較の条件を揃える。
    """
    import math

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device,
    *,
    log_every: int = 50,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Metrics:
    model.train()
    total = Metrics()

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        output = model.forward_batch(batch)
        loss, metrics = compute_loss(output, batch, config)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total.update(metrics)
        if step % log_every == 0:
            print(f"    step {step:5d}  {metrics.format()}")

    return total


def build_loaders(
    shard_dir: Path,
    *,
    batch_size: int,
    max_positions: int | None,
    workers: int,
    val_ratio: float,
) -> tuple[DataLoader, DataLoader]:
    shards = find_shards(shard_dir)
    if not shards:
        raise FileNotFoundError(
            f"シャードが見つかりません: {shard_dir}\n"
            "先に scripts/make_labels.py で棋譜を変換してください。"
        )

    train_shards, val_shards = split_shards(shards, val_ratio)
    train_set = ShardDataset(train_shards, max_positions=max_positions)
    val_set = ShardDataset(val_shards, max_positions=max_positions)
    print(f"train {len(train_set)} 局面 / val {len(val_set)} 局面")

    common = {"batch_size": batch_size, "collate_fn": collate, "num_workers": workers}
    return (
        DataLoader(train_set, shuffle=True, drop_last=True, **common),
        DataLoader(val_set, shuffle=False, **common),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("kokoro", "cnn"), default="kokoro")
    parser.add_argument(
        "--head", choices=HEAD_KINDS, default="plain",
        help="plain=素のsrc-dst (Phase 1) / desire=欲求分解 (Phase 2)",
    )
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument(
        "--max-positions", type=int, default=None, help="train/val それぞれの局面数上限"
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (既定は自動)")
    parser.add_argument("--cnn-channels", type=int, default=192, help="--model cnn のチャンネル数")
    parser.add_argument("--cnn-blocks", type=int, default=10, help="--model cnn の残差ブロック数")
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.05, help="全ステップ中のウォームアップ割合"
    )
    args = parser.parse_args()

    config = load_config()
    set_global_seed(config.seed)
    device = resolve_device(args.device)

    train_loader, val_loader = build_loaders(
        args.shard_dir,
        batch_size=args.batch_size,
        max_positions=args.max_positions,
        workers=args.workers,
        val_ratio=args.val_ratio,
    )

    model = build_model(
        args.model,
        config,
        channels=args.cnn_channels,
        blocks=args.cnn_blocks,
        head=args.head,
    ).to(device)
    parameters = sum(tensor.numel() for tensor in model.parameters())
    print(f"model={args.model} head={args.head} params={parameters / 1e6:.2f}M device={device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=int(total_steps * args.warmup_ratio),
        total_steps=total_steps,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        print(f"[epoch {epoch}/{args.epochs}]")
        train_metrics = train_epoch(
            model, train_loader, optimizer, config, device, scheduler=scheduler
        )
        val_metrics = evaluate(model, val_loader, config, device)
        elapsed = time.perf_counter() - started

        print(f"  train {train_metrics.format()}")
        print(f"  val   {val_metrics.format()}  ({elapsed:.1f}秒)")

        history.append(
            {"epoch": epoch, "train": asdict(train_metrics), "val": asdict(val_metrics)}
        )
        torch.save(
            {"model": model.state_dict(), "kind": args.model, "epoch": epoch},
            args.out_dir / f"{args.model}_latest.pt",
        )
        (args.out_dir / f"{args.model}_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    best = max(entry["val"]["accuracy"] for entry in history)
    print(f"最良の検証一致率: {best * 100:.2f}%  ({args.out_dir / f'{args.model}_history.json'})")


if __name__ == "__main__":
    main()
