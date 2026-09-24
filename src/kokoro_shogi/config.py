"""configs/*.yaml を型付きオブジェクトとして読み込む。

`configs/base.yaml` (ハイパーパラメータ) と `configs/features.yaml` (機能フラグ) が
コード側に入ってくる唯一の入口。DESIGN.md 設計原則2「全機能は configs の機能フラグで
個別にON/OFFできる」を成立させるため、フラグ参照は必ず FeatureFlags を経由させる。

未知のキーはエラーにする。yaml のtypoが「フラグが効かない」という無言の不具合に
化けるのを防ぐため。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"


@dataclass(frozen=True)
class ModelConfig:
    """DESIGN.md §3 のモデル形状。"""

    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_theta: int = 16
    d_mood: int = 32


@dataclass(frozen=True)
class LossConfig:
    """DESIGN.md §4 統一損失の係数。"""

    lambda_g: float = 0.01
    tau: float = 1.0
    c1: float = 1.0
    c2: float = 0.5
    c3: float = 0.01
    #: エンジン教師 (scripts/ops/join_teacher.py の *.teacher.npz) がある局面だけに効く 3 つ。
    #: 教師の無い局面・無いシャードでは従来どおり (勝敗 z と one-hot CE)
    c_soft: float = 1.0
    teacher_temp: float = 200.0
    teacher_value_weight: float = 1.0
    #: 駒の一生を教師にする (2026-09-20、docs/decisions/2026-09-20-fate-bipartite-proposal.md)。
    #: どちらも 0 で無効。段 1 の対照は 0 / 運命は lambda_fate / 運命+転生は両方
    lambda_fate: float = 0.0  # 運命損失 λ_f: Σ_i (V_i - f_i)^2
    fate_gamma: float = 0.966  # γ (20 手で 0.5)
    fate_promote_bonus: float = 0.5  # b: 成る
    fate_survive_bonus: float = 0.3  # c: 終局まで生き残る
    fate_mate_bonus: float = 1.0  # m: 勝った側の最終手を指す
    lambda_rebirth: float = 0.0  # 転生保存則 λ_r: 捕獲の前後で V_i(t+1) ≈ κ_s V_i(t)


@dataclass(frozen=True)
class FeatureFlags:
    """configs/features.yaml。全て false が「素の強い将棋AI」構成。"""

    mood: bool = False
    relations: bool = False
    council: bool = False
    individual: bool = False
    loyalty: bool = False
    league: bool = False


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    loss: LossConfig
    features: FeatureFlags
    seed: int = 42


def _build[T](cls: type[T], values: dict[str, Any], source: str) -> T:
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"{source}: 未知のキーがあります: {sorted(unknown)} (既知: {sorted(known)})"
        )
    return cls(**values)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: トップレベルはマッピングである必要があります。")
    return loaded


def load_config(config_dir: Path | str | None = None) -> Config:
    """base.yaml と features.yaml を読み込んで Config を返す。"""
    directory = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR

    base = _load_yaml(directory / "base.yaml")
    features = _load_yaml(directory / "features.yaml")

    base_source = str(directory / "base.yaml")
    return Config(
        model=_build(ModelConfig, base.get("model", {}), f"{base_source} model"),
        loss=_build(LossConfig, base.get("loss", {}), f"{base_source} loss"),
        features=_build(FeatureFlags, features, str(directory / "features.yaml")),
        seed=int(base.get("seed", 42)),
    )


def set_global_seed(seed: int) -> None:
    """再現性のため乱数シードを固定する。torch は入っていれば設定する。"""
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    # torch は dependency-groups の train 側にあり、前処理環境には無いことがある。
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
