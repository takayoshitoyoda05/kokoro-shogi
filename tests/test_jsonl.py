"""JSONL出力 (logging/jsonl.py) が docs/INTERFACE.md Schema 1.0 に一致することを検証する。

このテストが守っているのは「AI側が仕様外のJSONをUnityへ送らない」こと。
INTERFACE.md を変更したら、まずここを直してから実装を直す。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kokoro_shogi.config import REPO_ROOT
from kokoro_shogi.logging.jsonl import (
    JsonlWriter,
    LastMove,
    Mood,
    PieceInfo,
    Relation,
    StateUpdate,
    iter_jsonl,
    to_json_line,
    validate_line,
    validate_obj,
)

SAMPLE_DIR = REPO_ROOT / "sample_data"

#: docs/INTERFACE.md §3 に載っている state_update の実例そのもの
INTERFACE_STATE_UPDATE = {
    "schema": "1.0",
    "type": "state_update",
    "ply": 42,
    "sfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
    "last_move": {
        "from": "77",
        "to": "76",
        "piece_id": "P77_gen0_0003",
        "capture": False,
        "promote": False,
        "drop": False,
    },
    "eval": 0.12,
    "pieces": [
        {
            "piece_id": "P77_gen0_0003",
            "species": "FU",
            "owner": 0,
            "square": "76",
            "mood": {"fear": 0.7, "aggression": 0.2, "valence": -0.3},
            "desire": {
                "survive": 0.8,
                "attack": 0.1,
                "promote": 0.3,
                "defend": 0.5,
                "advance": 0.6,
                "redeploy": 0.0,
            },
            "alpha": 0.05,
            "relations": [{"to": "K59_gen0_0001", "r": 0.9}],
        }
    ],
    "council": [
        {
            "round": 1,
            "proposals": [{"piece_id": "B88_gen0_0010", "move": "2d2b+", "bid": 2.3}],
        }
    ],
    "narration": "3ラウンド目、角が2二への成り込みを強く主張した。",
}

#: §4-§5 の実例
INTERFACE_OTHERS = [
    {
        "schema": "1.0",
        "type": "legal_moves",
        "moves": [
            {"from": "77", "to": "76", "promote": False},
            {"from": "00", "to": "55", "drop_species": "FU"},
        ],
    },
    {
        "schema": "1.0",
        "type": "move_request",
        "move": {"from": "77", "to": "76", "promote": False},
    },
    {"schema": "1.0", "type": "game_control", "command": "start"},
    {
        "schema": "1.0",
        "type": "career",
        "pieces": [
            {
                "piece_id": "P77_gen0_0003",
                "species": "FU",
                "games": 120,
                "survival_rate": 0.42,
                "promotions": 18,
                "mvp_count": 3,
            }
        ],
        "last_game_mvp": {"piece_id": "R28_gen0_0011", "contribution": 2.3},
    },
]


# --- 仕様との一致 -----------------------------------------------------------


def test_interface_example_round_trips_unchanged() -> None:
    """INTERFACE.md §3 の例が、1バイトも変わらずに読み書きできる。"""
    message = validate_obj(INTERFACE_STATE_UPDATE)
    assert json.loads(to_json_line(message)) == INTERFACE_STATE_UPDATE


@pytest.mark.parametrize("example", INTERFACE_OTHERS, ids=lambda item: str(item["type"]))
def test_interface_examples_of_other_types(example: dict) -> None:
    """§4-§5 の legal_moves / move_request / game_control / career も同形で往復する。"""
    message = validate_obj(example)
    assert json.loads(to_json_line(message)) == example


def test_message_types_match_unity_router() -> None:
    """扱う type が MessageRouter.cs の switch と一致している。"""
    from kokoro_shogi.logging.jsonl import MESSAGE_MODELS

    assert set(MESSAGE_MODELS) == {
        "state_update",
        "legal_moves",
        "move_request",
        "game_control",
        "career",
    }


# --- 不正データの検出 -------------------------------------------------------


def test_missing_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="type"):
        validate_obj({"schema": "1.0", "ply": 0, "sfen": "x"})


def test_unknown_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="未知のメッセージ種別"):
        validate_obj({"schema": "1.0", "type": "mood_update"})


def test_unknown_field_is_rejected() -> None:
    """INTERFACE.md §7「本文書にないフィールドの無断追加」の禁止をコードで守る。"""
    payload = dict(INTERFACE_STATE_UPDATE, extra_field=1)
    with pytest.raises(ValidationError):
        validate_obj(payload)


def test_wrong_schema_version_is_rejected() -> None:
    payload = dict(INTERFACE_STATE_UPDATE, schema="2.0")
    with pytest.raises(ValidationError):
        validate_obj(payload)


def test_missing_required_field_is_rejected() -> None:
    payload = {key: value for key, value in INTERFACE_STATE_UPDATE.items() if key != "sfen"}
    with pytest.raises(ValidationError):
        validate_obj(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fear", 1.5),      # [0, 1] を超える
        ("fear", -0.1),
        ("valence", 1.5),   # [-1, +1] を超える
        ("valence", -1.5),
    ],
)
def test_mood_range_is_enforced(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        Mood(**{field: value})


def test_eval_range_is_enforced() -> None:
    with pytest.raises(ValidationError):
        StateUpdate(ply=1, sfen="x", eval=1.4)


@pytest.mark.parametrize(
    "square",
    ["0", "76a", "06", "90", "HAND", ""],
)
def test_invalid_square_is_rejected(square: str) -> None:
    with pytest.raises(ValidationError):
        PieceInfo(piece_id="P77_gen0_0003", species="FU", owner=0, square=square)


def test_hand_square_is_allowed() -> None:
    piece = PieceInfo(piece_id="P77_gen0_0003", species="FU", owner=0, square="hand")
    assert piece.square == "hand"


@pytest.mark.parametrize(
    "piece_id",
    ["P77_gen0_003", "X77_gen0_0003", "P77_0003", "P0_gen0_0003", "p77_gen0_0003"],
)
def test_invalid_piece_id_is_rejected(piece_id: str) -> None:
    with pytest.raises(ValidationError):
        PieceInfo(piece_id=piece_id, species="FU", owner=0, square="76")


def test_unknown_species_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PieceInfo(piece_id="P77_gen0_0003", species="FUU", owner=0, square="76")


def test_drop_uses_from_00() -> None:
    """打ちの移動元は "00" (INTERFACE.md §3)。"""
    move = LastMove(**{"from": "00"}, to="55", piece_id="P77_gen0_0003", drop=True)
    assert json.loads(to_json_line(move))["from"] == "00"


def test_to_square_00_is_rejected() -> None:
    """移動先は実在のマスでなければならない。"""
    with pytest.raises(ValidationError):
        LastMove(**{"from": "77"}, to="00", piece_id="P77_gen0_0003")


def test_weak_relation_is_rejected() -> None:
    """relations は r >= 0.3 のものだけ送る (INTERFACE.md §3)。"""
    with pytest.raises(ValidationError):
        PieceInfo(
            piece_id="P77_gen0_0003",
            species="FU",
            owner=0,
            square="76",
            relations=[Relation(to="K59_gen0_0001", r=0.1)],
        )


def test_too_many_relations_are_rejected() -> None:
    relations = [Relation(to=f"P{index + 1}7_gen0_0001", r=0.9) for index in range(6)]
    with pytest.raises(ValidationError):
        PieceInfo(
            piece_id="P77_gen0_0003", species="FU", owner=0, square="76", relations=relations
        )


def test_duplicate_piece_id_is_rejected() -> None:
    piece = {"piece_id": "P77_gen0_0003", "species": "FU", "owner": 0, "square": "76"}
    with pytest.raises(ValidationError):
        StateUpdate(ply=1, sfen="x", pieces=[PieceInfo(**piece), PieceInfo(**piece)])


# --- ファイル入出力 ---------------------------------------------------------


def test_writer_round_trip(tmp_path: Path) -> None:
    messages = [
        StateUpdate(ply=0, sfen="a"),
        StateUpdate(ply=1, sfen="b", last_move=LastMove(**{"from": "77"}, to="76",
                                                        piece_id="P77_gen0_0003")),
    ]
    path = tmp_path / "game.jsonl"
    with JsonlWriter(path) as writer:
        writer.write_all(messages)

    assert path.read_text(encoding="utf-8").count("\n") == 2
    assert [message.ply for message in iter_jsonl(path)] == [0, 1]  # type: ignore[union-attr]


def test_iter_jsonl_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        to_json_line(StateUpdate(ply=0, sfen="a")) + "\n" + '{"type": "state_update"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="broken.jsonl:2"):
        list(iter_jsonl(path))


# --- 納品物そのものの検証 ---------------------------------------------------


def _sample_files() -> list[Path]:
    return sorted(SAMPLE_DIR.glob("game*.jsonl"))


@pytest.mark.skipif(not _sample_files(), reason="sample_data/ が未生成 (gen_sample_jsonl.py)")
def test_sample_files_are_valid() -> None:
    """納品する sample_data/*.jsonl が全行 Schema 1.0 を通る。"""
    for path in _sample_files():
        messages = list(iter_jsonl(path))
        assert messages, f"{path.name} が空です。"

        first = messages[0]
        assert isinstance(first, StateUpdate)
        assert first.ply == 0, "1行目は初期局面 (ply=0)"
        assert first.last_move is None, "初期局面に last_move は無い"

        for index, message in enumerate(messages):
            assert isinstance(message, StateUpdate)
            assert message.ply == index, f"{path.name}: ply が連番でない"
            if index > 0:
                assert message.last_move is not None, f"{path.name}: {index}手目に last_move が無い"


@pytest.mark.skipif(not _sample_files(), reason="sample_data/ が未生成 (gen_sample_jsonl.py)")
def test_sample_career_is_valid() -> None:
    career_path = SAMPLE_DIR / "career.json"
    if not career_path.exists():
        pytest.skip("career.json が未生成")
    message = validate_line(career_path.read_text(encoding="utf-8"))
    assert message.type == "career"
