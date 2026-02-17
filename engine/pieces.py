"""Piece definitions and Banqi capture relationships."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class Side(str, Enum):
    """Player side."""

    RED = "red"
    BLACK = "black"

    def opponent(self) -> "Side":
        return Side.BLACK if self is Side.RED else Side.RED


class Rank(str, Enum):
    """Banqi/Xiangqi rank names."""

    GENERAL = "general"
    ADVISOR = "advisor"
    ELEPHANT = "elephant"
    HORSE = "horse"
    CHARIOT = "chariot"
    CANNON = "cannon"
    SOLDIER = "soldier"


RANK_STRENGTH: Dict[Rank, int] = {
    Rank.GENERAL: 7,
    Rank.ADVISOR: 6,
    Rank.ELEPHANT: 5,
    Rank.CHARIOT: 4,
    Rank.HORSE: 3,
    Rank.CANNON: 2,
    Rank.SOLDIER: 1,
}

RANK_VALUES: Dict[Rank, int] = {
    Rank.GENERAL: 90,
    Rank.ADVISOR: 20,
    Rank.ELEPHANT: 20,
    Rank.HORSE: 40,
    Rank.CHARIOT: 50,
    Rank.CANNON: 45,
    Rank.SOLDIER: 10,
}

SIDE_PREFIX: Dict[Side, str] = {
    Side.RED: "R",
    Side.BLACK: "B",
}

RANK_SYMBOL: Dict[Rank, str] = {
    Rank.GENERAL: "G",
    Rank.ADVISOR: "A",
    Rank.ELEPHANT: "E",
    Rank.HORSE: "H",
    Rank.CHARIOT: "R",
    Rank.CANNON: "C",
    Rank.SOLDIER: "S",
}

PIECE_COUNTS: Dict[Rank, int] = {
    Rank.GENERAL: 1,
    Rank.ADVISOR: 2,
    Rank.ELEPHANT: 2,
    Rank.HORSE: 2,
    Rank.CHARIOT: 2,
    Rank.CANNON: 2,
    Rank.SOLDIER: 5,
}

RANK_ORDER: List[Rank] = [
    Rank.GENERAL,
    Rank.ADVISOR,
    Rank.ELEPHANT,
    Rank.CHARIOT,
    Rank.HORSE,
    Rank.CANNON,
    Rank.SOLDIER,
]


@dataclass(frozen=True)
class Piece:
    """A revealed Banqi piece owned by one side."""

    side: Side
    rank: Rank

    @property
    def value(self) -> int:
        return RANK_VALUES[self.rank]

    @property
    def symbol(self) -> str:
        return f"{SIDE_PREFIX[self.side]}{RANK_SYMBOL[self.rank]}"


@dataclass(frozen=True)
class HiddenPiece:
    """Marker for face-down piece on the board."""

    symbol: str = "##"


HIDDEN = HiddenPiece()


def build_full_piece_pool() -> List[Piece]:
    """Build the full 32-piece Banqi pool."""
    pool: List[Piece] = []
    for side in (Side.RED, Side.BLACK):
        for rank, count in PIECE_COUNTS.items():
            for _ in range(count):
                pool.append(Piece(side=side, rank=rank))
    return pool
