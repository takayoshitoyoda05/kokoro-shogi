"""駒の一生を教師にする (2026-09-20): 運命の即時計算・f_i・転生保存則。"""

from __future__ import annotations

import numpy as np
import torch

from kokoro_shogi.config import Config, FeatureFlags, LossConfig, ModelConfig
from kokoro_shogi.core.squares import NUM_SQUARES
from kokoro_shogi.core.tokenizer import HAND_POSITION
from kokoro_shogi.data.sequence import compute_fate
from kokoro_shogi.train.distill import fate_target, rebirth_loss


def _toy_game():
    """2 駒 × 4 行。駒 0 は行 2 で取られる、駒 1 は行 3 で成る、最終手は駒 1 で勝ち。"""
    hand = HAND_POSITION[1]
    position = np.array([[10, 20], [11, 20], [hand, 21], [hand, 22]])
    owner = np.array([[0, 0], [0, 0], [1, 0], [1, 0]])
    mask = np.ones((4, 2), dtype=bool)
    promoted = np.array([[0, 0], [0, 0], [0, 0], [0, 1]])
    move_token = np.array([0, 1, 0, 1])
    result = np.array([1.0, -1.0, 1.0, 1.0], dtype=np.float32)
    return position, owner, mask, promoted, move_token, result


def test_compute_fate_counts_plies_to_first_event() -> None:
    cap, prom, mate, left = compute_fate(*_toy_game())
    # 駒 0: 行 0 から 2 手後、行 1 から 1 手後に取られる。持ち駒になったら運命なし
    assert cap[:, 0].tolist() == [2, 1, -1, -1]
    assert prom[:, 0].tolist() == [-1, -1, -1, -1]
    # 駒 1: 行 3 で成る。成った後は -1
    assert prom[:, 1].tolist() == [3, 2, 1, -1]
    assert cap[:, 1].tolist() == [-1, -1, -1, -1]
    # 最終手 (行 3, move_token=1) を指した側が勝ち → 駒 1 が詰ませた駒
    assert mate[:, 1].all() and not mate[:, 0].any()
    assert left.tolist() == [3, 2, 1, 0]


def test_compute_fate_no_mate_when_last_mover_lost() -> None:
    position, owner, mask, promoted, move_token, result = _toy_game()
    result = result.copy()
    result[-1] = -1.0
    _cap, _prom, mate, _left = compute_fate(position, owner, mask, promoted, move_token, result)
    assert not mate.any()


def _config(**loss) -> Config:
    return Config(model=ModelConfig(), loss=LossConfig(**loss), features=FeatureFlags())


def test_fate_target_capture_beats_promote_and_masks_hand() -> None:
    cap, prom, mate, left = compute_fate(*_toy_game())
    position = _toy_game()[0]
    batch = {
        "fate_capture": torch.as_tensor(cap),
        "fate_promote": torch.as_tensor(prom),
        "fate_mate": torch.as_tensor(mate),
        "plies_left": torch.as_tensor(left),
        "mask": torch.ones(4, 2, dtype=torch.bool),
        "position": torch.as_tensor(position),
    }
    config = _config(
        fate_gamma=0.5, fate_promote_bonus=0.5,
        fate_survive_bonus=0.3, fate_mate_bonus=1.0,
    )
    target, valid = fate_target(batch, config)
    # 駒 0 行 0: 2 手後に取られる → -0.5^2
    assert torch.isclose(target[0, 0], torch.tensor(-0.25))
    # 駒 1 行 0: 3 手後に成る (捕獲なし) → 0.5 · 0.5^3。詰ませた駒でも成りが優先
    assert torch.isclose(target[0, 1], torch.tensor(0.5 * 0.125))
    # 駒 1 行 3: 成り済み・捕獲なし → 詰ませた駒として m·γ^0 = 1.0
    assert torch.isclose(target[3, 1], torch.tensor(1.0))
    # 持ち駒 (行 2, 3 の駒 0) は無効
    assert valid[:, 0].tolist() == [True, True, False, False]
    assert valid[:, 1].all()


def test_rebirth_loss_only_on_captures_and_trains_kappa() -> None:
    species = torch.tensor([[0, 3]])
    kappa = torch.ones(8, requires_grad=True)
    prev_value = torch.tensor([[0.8, -0.2]])
    value = torch.tensor([[0.3, 0.1]], requires_grad=True)
    prev_position = torch.tensor([[10, 20]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    # 駒 0 が今取られた (盤上 → 持ち駒)
    position = torch.tensor([[NUM_SQUARES, 21]])
    loss = rebirth_loss(value, prev_value, species, position, prev_position, mask, mask, kappa)
    assert torch.isclose(loss, torch.tensor((0.3 - 0.8) ** 2))
    loss.backward()
    assert kappa.grad is not None and kappa.grad[0] != 0 and kappa.grad[3] == 0
    # 捕獲が無ければ 0
    none = rebirth_loss(value, prev_value, species, prev_position, prev_position, mask, mask, kappa)
    assert none.item() == 0.0


def test_rating_weight_is_linear_and_rejects_unknown() -> None:
    """レート線形重み (2026-09-21、arXiv:2603.29761 の linear r=20)。"""
    from pathlib import Path

    import cshogi

    from kokoro_shogi.data.floodgate import (
        WEIGHT_FLOOR,
        WEIGHT_MIN,
        WEIGHT_TOP,
        GameRecord,
        rating_weight,
    )

    def record(low: float, high: float) -> GameRecord:
        return GameRecord(
            path=Path("x.csa"), start_sfen=cshogi.STARTING_SFEN, moves=(),
            win=cshogi.BLACK_WIN, endgame="%TORYO", names=("a", "b"), ratings=(low, high),
        )

    assert rating_weight(record(WEIGHT_TOP + 500, 4500)) == 1.0
    assert rating_weight(record(WEIGHT_FLOOR - 500, 4000)) == WEIGHT_MIN
    middle = rating_weight(record((WEIGHT_FLOOR + WEIGHT_TOP) / 2, 4000))
    assert abs(middle - 0.5) < 1e-6
    # 両者の低い方で決まる
    assert rating_weight(record(4500, 2500)) == rating_weight(record(2500, 4500))
    # レート不明 (floodgate は 0.0) は重み付けできないので 0 = 呼び出し側が捨てる
    assert rating_weight(record(0.0, 4200)) == 0.0
    # 強さの比はおよそ 20:1
    assert abs(1.0 / WEIGHT_MIN - 20.0) < 1e-6


def test_weighted_loss_downweights_low_rated_positions() -> None:
    """weight 列があると policy/value 損失が重み付き平均になる。全て 1.0 なら従来と同一。"""
    import torch

    from kokoro_shogi.model.policy import PolicyOutput
    from kokoro_shogi.train.distill import compute_loss_tensors

    torch.manual_seed(0)
    logits = torch.randn(4, 12)
    batch = {
        "action": torch.tensor([0, 1, 2, 3]),
        "result": torch.zeros(4),
        "legal": torch.ones(4, 2, 3, 2, dtype=torch.bool),
    }
    output = PolicyOutput(logits=logits, value=torch.zeros(4), hidden=torch.zeros(4, 2, 3))
    config = _config()

    plain, _, _ = compute_loss_tensors(output, batch, config)
    same, _, _ = compute_loss_tensors(output, {**batch, "weight": torch.ones(4)}, config)
    assert torch.isclose(plain, same), "全て 1.0 の重みは従来と同一でなければならない"

    # 最初の 2 局面だけ重い → その 2 つの損失に寄る
    weight = torch.tensor([1.0, 1.0, 0.0, 0.0])
    weighted, _, _ = compute_loss_tensors(output, {**batch, "weight": weight}, config)
    head = torch.nn.functional.cross_entropy(logits[:2], batch["action"][:2])
    assert torch.isclose(weighted, head, atol=1e-5)


def test_malformed_csa_is_rejected_before_cshogi(tmp_path) -> None:
    """改行落ちで指し手がコメントに連結した CSA を、cshogi に渡す前に弾く (2026-09-21)。

    floodgate の 2012 年に実在し、そのまま `cshogi.Parser` に渡すと SIGSEGV で
    プロセスごと落ちる。Python では捕まえられないので事前に構造で判定する。
    """
    from kokoro_shogi.data.floodgate import looks_wellformed, parse_csa

    header = "V2\nN+a\nN-b\nP1-KY-KE-GI-KI-OU-KI-GI-KE-KY\n+\n"
    good = tmp_path / "good.csa"
    good.write_text(header + "'rating:a:b\n+2726FU\nT1\n-8384FU\nT2\n%TORYO\n")
    assert looks_wellformed(good)

    # 1 手目がコメント行の末尾に連結し、T 行だけが残っている
    bad = tmp_path / "bad.csa"
    bad.write_text(header + "'rating:a:b+2726FU\nT1\n\n-8384FU\nT2\n%TORYO\n")
    assert not looks_wellformed(bad)
    assert parse_csa(bad) is None


def test_year_filter_selects_by_path(tmp_path) -> None:
    """--years の年抽出 (2026-09-21)。floodgate は 2015 年頃まで CSA にレートを
    書いておらず、--rating-weight では全部落ちるので年で絞る必要がある。"""
    import re

    year_pattern = re.compile(r"(?:^|\D)(20\d\d)")
    wanted = {str(y) for y in range(2015, 2027)}

    def in_wanted(path) -> bool:
        for piece in (*path.parts[:-1], path.name):
            found = year_pattern.search(piece)
            if found:
                return found.group(1) in wanted
        return False

    from pathlib import Path

    assert in_wanted(Path("data/floodgate/csa/2024/2024/wdoor+x+20240101120000.csa"))
    assert in_wanted(Path("data/floodgate/csa/2015/a.csa"))
    assert not in_wanted(Path("data/floodgate/csa/2012/2012/wdoor+x+20120101120000.csa"))
    assert not in_wanted(Path("data/floodgate/csa/2008/a.csa"))
    # 年のディレクトリが無ければファイル名の日付で判定する
    assert in_wanted(Path("flat/wdoor+floodgate+20250115203004.csa"))
