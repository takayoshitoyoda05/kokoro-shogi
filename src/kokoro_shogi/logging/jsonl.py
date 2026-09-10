"""Unity向けメッセージ (docs/INTERFACE.md Schema 1.0) のコード側の正本。

AI側が Unity へ出す JSON は、リプレイ (JSONL) でもライブ (WebSocket) でも
**必ずこのモジュールのモデルを通す**。1通のWebSocketメッセージ = JSONLの1行と
完全に同形なので、両者でモデルを共有できる (INTERFACE.md §1)。

INTERFACE.md §7 は「本文書にないフィールドの無断追加 / リネーム / 値域の変更」を
禁じている。それをコードで担保するため、全モデルを extra="forbid" にし、
値域も pydantic の制約として書き下している。仕様に反する state_update は
Unityに届く前にここで例外になる。

使い方::

    with JsonlWriter(path) as writer:
        writer.write(state_update)

    for message in iter_jsonl(path):   # 読み戻して検証
        ...
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer
from pydantic.functional_serializers import SerializerFunctionWrapHandler

from kokoro_shogi.core.pieces import Species

SCHEMA_VERSION = "1.0"

# --- 値域つきの型 (INTERFACE.md §3 のコメントをそのまま制約にする) -----------

Unit = Annotated[float, Field(ge=0.0, le=1.0)]       # [0, 1]
Signed = Annotated[float, Field(ge=-1.0, le=1.0)]    # [-1, +1]

#: 盤上のマス "76" または持ち駒 "hand"
SquareStr = Annotated[str, Field(pattern=r"^(?:[1-9][1-9]|hand)$")]
#: 移動元。打ちの場合は "00"
FromStr = Annotated[str, Field(pattern=r"^(?:[1-9][1-9]|00)$")]
#: 移動先。打ちでも実在のマス
ToStr = Annotated[str, Field(pattern=r"^[1-9][1-9]$")]
#: 恒久ID <初期位置SFEN文字><筋段>_gen<世代>_<連番4桁>
PieceIdStr = Annotated[str, Field(pattern=r"^[PLNSGBRK][1-9][1-9]_gen\d+_\d{4}$")]

#: relations は r >= 0.3 のものだけを最大5件 (INTERFACE.md §3)
RELATION_MIN_R = 0.3
MAX_RELATIONS_PER_PIECE = 5


class _Strict(BaseModel):
    """INTERFACE.md にないフィールドを弾く共通設定。"""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class _Message(_Strict):
    """全メッセージ共通のヘッダ (INTERFACE.md §1)。"""

    schema_: Literal["1.0"] = Field(default=SCHEMA_VERSION, alias="schema")


# --- §3 state_update --------------------------------------------------------


class LastMove(_Strict):
    from_: FromStr = Field(alias="from")
    to: ToStr
    piece_id: PieceIdStr
    capture: bool = False
    promote: bool = False
    drop: bool = False


class Mood(_Strict):
    fear: Unit = 0.0
    aggression: Unit = 0.0
    valence: Signed = 0.0


class Desire(_Strict):
    survive: Unit = 0.0
    attack: Unit = 0.0
    promote: Unit = 0.0
    defend: Unit = 0.0
    advance: Unit = 0.0
    redeploy: Unit = 0.0


class Relation(_Strict):
    to: PieceIdStr
    r: Unit


class PieceInfo(_Strict):
    piece_id: PieceIdStr
    species: Species
    owner: Literal[0, 1]
    square: SquareStr
    mood: Mood = Field(default_factory=Mood)
    desire: Desire = Field(default_factory=Desire)
    alpha: Unit = 0.0
    relations: list[Relation] = Field(default_factory=list)

    @field_validator("relations")
    @classmethod
    def _check_relations(cls, value: list[Relation]) -> list[Relation]:
        if len(value) > MAX_RELATIONS_PER_PIECE:
            raise ValueError(f"relations は最大 {MAX_RELATIONS_PER_PIECE} 件です: {len(value)} 件")
        weak = [relation.to for relation in value if relation.r < RELATION_MIN_R]
        if weak:
            raise ValueError(f"relations は r >= {RELATION_MIN_R} のみ送ります: {weak}")
        return value


class Proposal(_Strict):
    piece_id: PieceIdStr
    move: str
    bid: float


class CouncilRound(_Strict):
    round: int = Field(ge=1)
    proposals: list[Proposal] = Field(default_factory=list)


class StateUpdate(_Message):
    """1手ごとの盤面 + 内面データ (INTERFACE.md §3)。"""

    type: Literal["state_update"] = "state_update"
    ply: int = Field(ge=0)
    sfen: str = Field(min_length=1)
    #: 直前の1手。ply=0 (初期局面) では None
    last_move: LastMove | None = None
    #: 形勢 V。手番側ではなく **常に先手** 有利が + (INTERFACE.md §3)
    eval: Signed = 0.0
    pieces: list[PieceInfo] = Field(default_factory=list, max_length=40)
    #: 会議機能OFF時は空配列
    council: list[CouncilRound] = Field(default_factory=list)
    #: 実況文。なければ空文字
    narration: str = ""

    @field_validator("pieces")
    @classmethod
    def _check_unique_piece_id(cls, value: list[PieceInfo]) -> list[PieceInfo]:
        seen: set[str] = set()
        for piece in value:
            if piece.piece_id in seen:
                raise ValueError(f"piece_id が重複しています: {piece.piece_id}")
            seen.add(piece.piece_id)
        return value


# --- §4 人間対局 ------------------------------------------------------------


class LegalMove(_Strict):
    from_: FromStr = Field(alias="from")
    to: ToStr
    promote: bool = False
    #: 打つ手のときだけ入る駒種
    drop_species: Species | None = None

    @field_validator("drop_species", mode="before")
    @classmethod
    def _empty_string_is_none(cls, value: object) -> object:
        """Unity の JsonUtility は null の string を "" で送ってくるので、無しとして扱う。"""
        return None if value == "" else value

    @model_serializer(mode="wrap")
    def _omit_inapplicable_fields(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """その手に当てはまらないフィールドはキーごと落とす (INTERFACE.md §4 の例と同形)。

        - 通常の手には drop_species が無い
        - 打つ手は成れないので promote が無い

        Unity の JsonUtility はキーが欠けていてもフィールドを既定値のまま残すので
        受信側の実装は変わらないが、仕様書の例とバイト単位で一致させておけば
        U1がデシリアライズを合わせるときに迷わない。
        """
        data = handler(self)
        if data.get("drop_species") is None:
            data.pop("drop_species", None)
        else:
            data.pop("promote", None)
        return data


class LegalMovesMessage(_Message):
    type: Literal["legal_moves"] = "legal_moves"
    moves: list[LegalMove] = Field(default_factory=list)


class MoveRequest(_Message):
    type: Literal["move_request"] = "move_request"
    move: LegalMove


class GameControl(_Message):
    type: Literal["game_control"] = "game_control"
    command: Literal["start", "resign", "reset"]


# --- §5 career --------------------------------------------------------------


class CareerPiece(_Strict):
    piece_id: PieceIdStr
    species: Species
    games: int = Field(ge=0)
    survival_rate: Unit = 0.0
    promotions: int = Field(ge=0)
    mvp_count: int = Field(ge=0)


class CareerMvp(_Strict):
    piece_id: PieceIdStr
    contribution: float


class CareerMessage(_Message):
    type: Literal["career"] = "career"
    pieces: list[CareerPiece] = Field(default_factory=list)
    last_game_mvp: CareerMvp | None = None


#: type の値 → モデル。MessageRouter.cs の switch と対応する
MESSAGE_MODELS: dict[str, type[_Message]] = {
    "state_update": StateUpdate,
    "legal_moves": LegalMovesMessage,
    "move_request": MoveRequest,
    "game_control": GameControl,
    "career": CareerMessage,
}

Message = StateUpdate | LegalMovesMessage | MoveRequest | GameControl | CareerMessage


# --- 入出力 -----------------------------------------------------------------


def to_json_line(message: _Message) -> str:
    """メッセージを1行のJSON文字列にする (改行を含まない)。"""
    return message.model_dump_json(by_alias=True)


def validate_obj(obj: Any) -> Message:
    """dict を type に応じたモデルへ検証付きで変換する。"""
    if not isinstance(obj, dict):
        raise ValueError(f"メッセージはJSONオブジェクトである必要があります: {type(obj).__name__}")

    message_type = obj.get("type")
    if message_type is None:
        raise ValueError("必須フィールド type がありません。")

    model = MESSAGE_MODELS.get(message_type)
    if model is None:
        raise ValueError(
            f"未知のメッセージ種別です: {message_type!r} (既知: {sorted(MESSAGE_MODELS)})"
        )
    return model.model_validate(obj)  # type: ignore[return-value]


def validate_line(line: str) -> Message:
    """JSONL 1行を検証してモデルへ変換する。"""
    return validate_obj(json.loads(line))


def iter_jsonl(path: Path | str) -> Iterator[Message]:
    """JSONLファイルを1行ずつ検証しながら読む。空行は飛ばす。"""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield validate_line(line)
            except Exception as exc:
                raise ValueError(f"{file_path}:{number}: {exc}") from exc


class JsonlWriter:
    """1行1メッセージで JSONL を書き出す。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._file = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def write(self, message: _Message) -> None:
        if self._file is None:
            raise RuntimeError("JsonlWriter は with 文の中で使ってください。")
        self._file.write(to_json_line(message))
        self._file.write("\n")

    def write_all(self, messages: Iterable[_Message]) -> None:
        for message in messages:
            self.write(message)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
