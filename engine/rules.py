"""Rules helpers for Taiwanese Banqi."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from engine.pieces import Rank, RANK_STRENGTH

BOARD_ROWS = 4
BOARD_COLS = 8

Position = Tuple[int, int]


def in_bounds(pos: Position) -> bool:
    """Return whether a position is inside the Banqi board."""
    row, col = pos
    return 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS


def orthogonal_neighbors(pos: Position) -> Iterable[Position]:
    """Yield orthogonally adjacent positions in bounds."""
    row, col = pos
    candidates = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
    for candidate in candidates:
        if in_bounds(candidate):
            yield candidate


def is_adjacent_orthogonal(a: Position, b: Position) -> bool:
    """Return whether b is one orthogonal step away from a."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def can_standard_capture(attacker: Rank, defender: Rank) -> bool:
    """Return whether standard (non-cannon-jump) capture is legal."""
    if attacker is Rank.CANNON:
        return False
    if attacker is Rank.GENERAL and defender is Rank.SOLDIER:
        return False
    if attacker is Rank.SOLDIER and defender is Rank.GENERAL:
        return True
    return RANK_STRENGTH[attacker] >= RANK_STRENGTH[defender]


def positions_between(start: Position, end: Position) -> List[Position]:
    """Return squares strictly between two aligned positions."""
    sr, sc = start
    er, ec = end
    between: List[Position] = []
    if sr == er:
        step = 1 if ec > sc else -1
        for col in range(sc + step, ec, step):
            between.append((sr, col))
    elif sc == ec:
        step = 1 if er > sr else -1
        for row in range(sr + step, er, step):
            between.append((row, sc))
    return between


def is_straight_line(start: Position, end: Position) -> bool:
    """Return whether positions are aligned orthogonally."""
    return start[0] == end[0] or start[1] == end[1]


def count_non_empty_in_path(board_cells: Sequence[Sequence[object | None]], path: Sequence[Position]) -> int:
    """Count occupied cells in a path."""
    count = 0
    for row, col in path:
        if board_cells[row][col] is not None:
            count += 1
    return count
