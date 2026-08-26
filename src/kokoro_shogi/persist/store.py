"""駒の個体データを永続化するSQLiteストア [B] (DESIGN.md §4d)。

対局後ループ (勾配ループの外側) の記憶装置。保存するのは3つ:

- **θ_ind**: 個体性格 (16次元 float32)。COMA近似の功績で微小更新され、
  血統 (ES交叉+変異) で次世代へ渡る
- **career**: 対局数・生存・成り・MVP・貢献度合計 (INTERFACE.md §5 の career
  メッセージの供給元)
- **lineage**: 親子関係 (piece_id 同士)

piece_id は "P77_gen0_0003" (初期位置+世代+連番) で世代を運ぶ。同じ初期位置の
駒は世代を跨いで同じ「家系」になる。DBは1ファイル (既定 data/pieces.sqlite3)。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from kokoro_shogi.config import REPO_ROOT

DEFAULT_DB = REPO_ROOT / "data" / "pieces.sqlite3"
D_INDIVIDUAL = 16

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pieces (
    piece_id     TEXT PRIMARY KEY,
    species      TEXT NOT NULL,
    generation   INTEGER NOT NULL DEFAULT 0,
    theta        BLOB NOT NULL,
    games        INTEGER NOT NULL DEFAULT 0,
    survivals    INTEGER NOT NULL DEFAULT 0,
    promotions   INTEGER NOT NULL DEFAULT 0,
    mvp_count    INTEGER NOT NULL DEFAULT 0,
    contribution REAL NOT NULL DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS lineage (
    child  TEXT NOT NULL,
    parent TEXT NOT NULL,
    PRIMARY KEY (child, parent)
);
"""


def _generation_of(piece_id: str) -> int:
    """"P77_gen3_0012" → 3。形式外は0。"""
    try:
        return int(piece_id.split("_gen")[1].split("_")[0])
    except (IndexError, ValueError):
        return 0


class PieceStore:
    """θ_ind / career / lineage の読み書き。

    `load_theta` は未知の piece_id を零ベクトルで自動登録する
    (θ_ind の合流点は零初期化なので、新入りは「無個性」から始まる)。
    """

    def __init__(self, path: Path | str = DEFAULT_DB, *, d_individual: int = D_INDIVIDUAL) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.d_individual = d_individual
        self._db = sqlite3.connect(self.path)
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> PieceStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self._db.commit()
        self.close()

    # --- θ_ind ------------------------------------------------------------

    def load_theta(self, piece_ids: list[str], species: dict[str, str] | None = None) -> np.ndarray:
        """`(N, d_individual)` float32。未知IDは零ベクトルで登録して返す。"""
        out = np.zeros((len(piece_ids), self.d_individual), dtype=np.float32)
        for row, piece_id in enumerate(piece_ids):
            found = self._db.execute(
                "SELECT theta FROM pieces WHERE piece_id = ?", (piece_id,)
            ).fetchone()
            if found is None:
                kind = (species or {}).get(piece_id, piece_id[:1])
                self._db.execute(
                    "INSERT INTO pieces (piece_id, species, generation, theta) VALUES (?,?,?,?)",
                    (piece_id, kind, _generation_of(piece_id), out[row].tobytes()),
                )
            else:
                out[row] = np.frombuffer(found[0], dtype=np.float32)
        self._db.commit()
        return out

    def save_theta(self, thetas: dict[str, np.ndarray]) -> None:
        for piece_id, theta in thetas.items():
            self._db.execute(
                "UPDATE pieces SET theta = ? WHERE piece_id = ?",
                (np.asarray(theta, dtype=np.float32).tobytes(), piece_id),
            )
        self._db.commit()

    # --- career -----------------------------------------------------------

    def record_game(
        self,
        survived: dict[str, bool],
        promoted: dict[str, bool],
        contribution: dict[str, float],
        mvp_id: str | None,
    ) -> None:
        """1局ぶんのキャリアを積む。全キーは登録済み piece_id であること。"""
        for piece_id, alive in survived.items():
            self._db.execute(
                "UPDATE pieces SET games = games + 1, survivals = survivals + ?,"
                " promotions = promotions + ?, contribution = contribution + ?"
                " WHERE piece_id = ?",
                (
                    int(alive),
                    int(promoted.get(piece_id, False)),
                    float(contribution.get(piece_id, 0.0)),
                    piece_id,
                ),
            )
        if mvp_id is not None:
            self._db.execute(
                "UPDATE pieces SET mvp_count = mvp_count + 1 WHERE piece_id = ?", (mvp_id,)
            )
        self._db.commit()

    def careers(self) -> list[dict]:
        """career メッセージ (INTERFACE.md §5) の素材。games>0 の駒のみ。"""
        rows = self._db.execute(
            "SELECT piece_id, species, games, survivals, promotions, mvp_count, contribution"
            " FROM pieces WHERE games > 0 ORDER BY piece_id"
        ).fetchall()
        return [
            {
                "piece_id": piece_id,
                "species": species,
                "games": games,
                "survival_rate": survivals / games,
                "promotions": promotions,
                "mvp_count": mvp,
                "contribution": contribution,
            }
            for piece_id, species, games, survivals, promotions, mvp, contribution in rows
        ]

    # --- 血統 (ES交叉+変異) [B/F] ------------------------------------------

    def breed(
        self,
        parent_p: str,
        parent_q: str,
        child_id: str,
        *,
        sigma: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """$\\theta_c = \\beta\\theta_p + (1-\\beta)\\theta_q + \\varepsilon$ (DESIGN.md §4d)。

        子は pieces に登録され、lineage に両親が刻まれる。
        """
        rng = rng or np.random.default_rng()
        theta_p, theta_q = self.load_theta([parent_p, parent_q])
        beta = float(rng.uniform())
        child = beta * theta_p + (1.0 - beta) * theta_q
        child = child + rng.normal(0.0, sigma, size=child.shape).astype(np.float32)

        parent_row = self._db.execute(
            "SELECT species FROM pieces WHERE piece_id = ?", (parent_p,)
        ).fetchone()
        self._db.execute(
            "INSERT OR REPLACE INTO pieces (piece_id, species, generation, theta)"
            " VALUES (?,?,?,?)",
            (child_id, parent_row[0], _generation_of(child_id), child.tobytes()),
        )
        for parent in (parent_p, parent_q):
            self._db.execute(
                "INSERT OR IGNORE INTO lineage (child, parent) VALUES (?,?)", (child_id, parent)
            )
        self._db.commit()
        return child

    def lineage_of(self, piece_id: str) -> list[str]:
        return [
            row[0]
            for row in self._db.execute(
                "SELECT parent FROM lineage WHERE child = ? ORDER BY parent", (piece_id,)
            )
        ]


__all__ = ["D_INDIVIDUAL", "DEFAULT_DB", "PieceStore"]
