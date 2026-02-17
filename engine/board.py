"""Banqi board state, legal move generation, and state encoding."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from engine.pieces import HIDDEN, HiddenPiece, Piece, Rank, RANK_ORDER, Side, build_full_piece_pool
from engine.rules import (
    BOARD_COLS,
    BOARD_ROWS,
    Position,
    can_standard_capture,
    in_bounds,
    orthogonal_neighbors,
)

ACTION_FLIP_COUNT = BOARD_ROWS * BOARD_COLS
ACTION_MOVE_COUNT = (BOARD_ROWS * BOARD_COLS) * (BOARD_ROWS * BOARD_COLS)
ACTION_SPACE_SIZE = ACTION_FLIP_COUNT + ACTION_MOVE_COUNT


@dataclass(frozen=True)
class Move:
    """A Banqi action."""

    kind: str
    to_pos: Position
    from_pos: Optional[Position] = None


@dataclass(frozen=True)
class MoveResult:
    """Result metadata for an applied move."""

    captured_piece: Optional[Piece]
    flipped_piece: Optional[Piece]
    winner: Optional[Side]
    is_draw: bool


Cell = Optional[object]

_RANK_CHANNEL_INDEX = {rank: idx for idx, rank in enumerate(RANK_ORDER)}


class Board:
    """Banqi board with hidden pieces and deterministic RNG support."""

    rows: int = BOARD_ROWS
    cols: int = BOARD_COLS
    action_space_size: int = ACTION_SPACE_SIZE

    def __init__(
        self,
        seed: Optional[int] = None,
        draw_no_progress_limit: int = 50,
        forbid_repetition: bool = True,
        repetition_policy: str = "long_chase_only",
        long_chase_limit: int = 6,
    ) -> None:
        self.seed = seed
        self.draw_no_progress_limit = draw_no_progress_limit
        self.forbid_repetition = forbid_repetition
        self.repetition_policy = repetition_policy
        self.long_chase_limit = long_chase_limit
        self.current_turn = Side.RED
        self.no_progress_plies = 0
        self.ply_count = 0

        self._rng = random.Random(seed)
        self._piece_bag: List[Piece] = build_full_piece_pool()
        self._rng.shuffle(self._piece_bag)

        self.grid: List[List[Cell]] = [[HIDDEN for _ in range(self.cols)] for _ in range(self.rows)]
        self._state_visits: Dict[Tuple[object, ...], int] = {}
        self._last_non_capture_chase_targets: Dict[Side, Set[Position]] = {
            Side.RED: set(),
            Side.BLACK: set(),
        }
        self._long_chase_streak: Dict[Side, int] = {
            Side.RED: 0,
            Side.BLACK: 0,
        }
        self._record_current_state()

    def clone(self) -> "Board":
        """Deep copy board state, including RNG state."""
        cloned = Board.__new__(Board)
        cloned.seed = self.seed
        cloned.draw_no_progress_limit = self.draw_no_progress_limit
        cloned.forbid_repetition = self.forbid_repetition
        cloned.repetition_policy = self.repetition_policy
        cloned.long_chase_limit = self.long_chase_limit
        cloned.current_turn = self.current_turn
        cloned.no_progress_plies = self.no_progress_plies
        cloned.ply_count = self.ply_count
        cloned._piece_bag = list(self._piece_bag)
        cloned.grid = [list(row) for row in self.grid]
        cloned._state_visits = dict(self._state_visits)
        cloned._last_non_capture_chase_targets = {
            Side.RED: set(self._last_non_capture_chase_targets[Side.RED]),
            Side.BLACK: set(self._last_non_capture_chase_targets[Side.BLACK]),
        }
        cloned._long_chase_streak = {
            Side.RED: self._long_chase_streak[Side.RED],
            Side.BLACK: self._long_chase_streak[Side.BLACK],
        }
        cloned._rng = random.Random()
        cloned._rng.setstate(self._rng.getstate())
        return cloned

    def iter_positions(self) -> Iterable[Position]:
        """Yield all board positions."""
        for row in range(self.rows):
            for col in range(self.cols):
                yield (row, col)

    def get_cell(self, pos: Position) -> Cell:
        """Return cell content at a position."""
        row, col = pos
        return self.grid[row][col]

    def get_legal_moves(self) -> List[Move]:
        """Generate all legal moves for the current player."""
        pseudo_moves = self._get_pseudo_legal_moves()

        if not self.forbid_repetition:
            return pseudo_moves

        legal_moves: List[Move] = []
        for move in pseudo_moves:
            if not self._would_repeat_position(move):
                legal_moves.append(move)
        return legal_moves

    def _get_pseudo_legal_moves(self) -> List[Move]:
        """Generate legal-by-piece-rule moves without repetition-policy filtering."""
        pseudo_moves: List[Move] = []

        for pos in self.iter_positions():
            cell = self.get_cell(pos)
            if isinstance(cell, HiddenPiece):
                pseudo_moves.append(Move(kind="flip", to_pos=pos))

        for pos in self.iter_positions():
            cell = self.get_cell(pos)
            if not isinstance(cell, Piece):
                continue
            if cell.side is not self.current_turn:
                continue
            pseudo_moves.extend(self._piece_moves(pos, cell))
        return pseudo_moves

    def _piece_moves(self, from_pos: Position, piece: Piece) -> List[Move]:
        piece_moves: List[Move] = []
        for to_pos in orthogonal_neighbors(from_pos):
            target = self.get_cell(to_pos)
            if target is None:
                piece_moves.append(Move(kind="move", from_pos=from_pos, to_pos=to_pos))
                continue
            if isinstance(target, Piece) and target.side is not piece.side:
                if can_standard_capture(piece.rank, target.rank):
                    piece_moves.append(Move(kind="move", from_pos=from_pos, to_pos=to_pos))

        if piece.rank is Rank.CANNON:
            piece_moves.extend(self._cannon_capture_moves(from_pos, piece.side))
        return piece_moves

    def _cannon_capture_moves(self, from_pos: Position, side: Side) -> List[Move]:
        moves: List[Move] = []
        row, col = from_pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            screen_seen = False
            while in_bounds((r, c)):
                cell = self.grid[r][c]
                if cell is None:
                    r += dr
                    c += dc
                    continue
                if not screen_seen:
                    screen_seen = True
                    r += dr
                    c += dc
                    continue
                if isinstance(cell, Piece) and cell.side is not side:
                    moves.append(Move(kind="move", from_pos=from_pos, to_pos=(r, c)))
                break
        return moves

    def apply_move(self, move: Move) -> MoveResult:
        """Apply a legal move and switch turn."""
        legal_moves = self.get_legal_moves()
        if move not in legal_moves:
            raise ValueError(f"Illegal move: {move}")

        mover = self.current_turn
        captured_piece, flipped_piece = self._apply_move_no_validation(move)
        if move.kind == "move" and captured_piece is None:
            current_targets = self._compute_chase_targets(mover)
            previous_targets = self._last_non_capture_chase_targets[mover]
            continued_chase = bool(previous_targets.intersection(current_targets)) and bool(current_targets)
            if continued_chase:
                self._long_chase_streak[mover] += 1
            elif current_targets:
                self._long_chase_streak[mover] = 1
            else:
                self._long_chase_streak[mover] = 0
            self._last_non_capture_chase_targets[mover] = current_targets
        else:
            self._last_non_capture_chase_targets[mover] = set()
            self._long_chase_streak[mover] = 0

        self.ply_count += 1
        self.current_turn = self.current_turn.opponent()
        self._record_current_state()

        is_terminal, winner, is_draw = self.game_over()
        return MoveResult(
            captured_piece=captured_piece,
            flipped_piece=flipped_piece,
            winner=winner if is_terminal else None,
            is_draw=is_draw if is_terminal else False,
        )

    def _apply_move_no_validation(self, move: Move) -> Tuple[Optional[Piece], Optional[Piece]]:
        """Apply move effects without checking legality/history."""
        captured_piece: Optional[Piece] = None
        flipped_piece: Optional[Piece] = None

        if move.kind == "flip":
            row, col = move.to_pos
            if not self._piece_bag:
                raise RuntimeError("No pieces left in bag for flip.")
            flipped_piece = self._piece_bag.pop()
            self.grid[row][col] = flipped_piece
            self.no_progress_plies = 0
        elif move.kind == "move" and move.from_pos is not None:
            src_row, src_col = move.from_pos
            dst_row, dst_col = move.to_pos
            moving_piece = self.grid[src_row][src_col]
            if not isinstance(moving_piece, Piece):
                raise ValueError("Source square does not contain a movable piece.")
            target = self.grid[dst_row][dst_col]
            if isinstance(target, Piece):
                captured_piece = target
                self.no_progress_plies = 0
            else:
                self.no_progress_plies += 1
            self.grid[src_row][src_col] = None
            self.grid[dst_row][dst_col] = moving_piece
        else:
            raise ValueError(f"Unsupported move format: {move}")
        return captured_piece, flipped_piece

    def _state_key(self) -> Tuple[object, ...]:
        """State hash key for repetition detection."""
        cells: List[object] = []
        for row, col in self.iter_positions():
            cell = self.grid[row][col]
            if cell is None:
                cells.append(".")
            elif isinstance(cell, HiddenPiece):
                cells.append("#")
            elif isinstance(cell, Piece):
                cells.append((cell.side.value, cell.rank.value))
            else:
                cells.append("?")

        bag_state = tuple((piece.side.value, piece.rank.value) for piece in self._piece_bag)
        return (self.current_turn.value, tuple(cells), bag_state)

    def _record_current_state(self) -> None:
        key = self._state_key()
        self._state_visits[key] = self._state_visits.get(key, 0) + 1

    def _would_repeat_position(self, move: Move) -> bool:
        """Return True if move violates configured repetition policy."""
        trial = self.clone()
        mover = trial.current_turn
        captured_piece, _ = trial._apply_move_no_validation(move)
        trial.ply_count += 1
        trial.current_turn = trial.current_turn.opponent()
        next_key = trial._state_key()
        repeated = next_key in self._state_visits
        if not repeated:
            return False

        policy = self.repetition_policy
        if policy == "superko":
            return True
        if policy == "off":
            return False
        if policy == "long_chase_only":
            if move.kind != "move" or captured_piece is not None:
                return False
            current_targets = trial._compute_chase_targets(mover)
            if not current_targets:
                return False
            previous_targets = self._last_non_capture_chase_targets[mover]
            continued_chase = bool(previous_targets.intersection(current_targets))
            if not continued_chase:
                return False

            # Allow finite pursuit windows, forbid over-limit long-chase loops.
            next_streak = self._long_chase_streak[mover] + 1
            return next_streak > self.long_chase_limit
        # Unknown policy fallback: safe strict behavior.
        return True

    def _compute_chase_targets(self, attacker_side: Side) -> Set[Position]:
        """
        Return enemy piece positions currently capturable by attacker_side.

        This approximates "chasing pressure" for long-chase prohibition.
        """
        trial = self.clone()
        trial.current_turn = attacker_side
        targets: Set[Position] = set()
        for move in trial._get_pseudo_legal_moves():
            if move.kind != "move" or move.from_pos is None:
                continue
            target = trial.get_cell(move.to_pos)
            if isinstance(target, Piece) and target.side is not attacker_side:
                targets.add(move.to_pos)
        return targets

    def piece_count_remaining(self, side: Side) -> int:
        """Count pieces of a side still alive (revealed on board + hidden in bag)."""
        on_board = 0
        for pos in self.iter_positions():
            cell = self.get_cell(pos)
            if isinstance(cell, Piece) and cell.side is side:
                on_board += 1
        in_bag = sum(1 for p in self._piece_bag if p.side is side)
        return on_board + in_bag

    def game_over(self) -> Tuple[bool, Optional[Side], bool]:
        """Return (is_terminal, winner, is_draw)."""
        if self.no_progress_plies >= self.draw_no_progress_limit:
            return True, None, True

        red_remaining = self.piece_count_remaining(Side.RED)
        black_remaining = self.piece_count_remaining(Side.BLACK)
        if red_remaining == 0 and black_remaining == 0:
            return True, None, True
        if red_remaining == 0:
            return True, Side.BLACK, False
        if black_remaining == 0:
            return True, Side.RED, False

        if not self.get_legal_moves():
            return True, self.current_turn.opponent(), False
        return False, None, False

    def encode_state(self) -> np.ndarray:
        """Encode current state for neural network input."""
        channels = 17
        encoded = np.zeros((channels, self.rows, self.cols), dtype=np.float32)

        # Side to move plane.
        encoded[0, :, :] = 1.0 if self.current_turn is Side.RED else 0.0

        for row, col in self.iter_positions():
            cell = self.grid[row][col]
            if isinstance(cell, HiddenPiece):
                encoded[1, row, col] = 1.0
            elif isinstance(cell, Piece):
                rank_idx = _RANK_CHANNEL_INDEX[cell.rank]
                base_channel = 2 if cell.side is Side.RED else 9
                encoded[base_channel + rank_idx, row, col] = 1.0

        encoded[16, :, :] = min(1.0, self.no_progress_plies / max(1, self.draw_no_progress_limit))
        return encoded

    @staticmethod
    def pos_to_index(pos: Position) -> int:
        """Convert a board position to flattened index."""
        return pos[0] * BOARD_COLS + pos[1]

    @staticmethod
    def index_to_pos(index: int) -> Position:
        """Convert flattened index to board position."""
        return (index // BOARD_COLS, index % BOARD_COLS)

    def move_to_action(self, move: Move) -> int:
        """Map move to fixed action index."""
        if move.kind == "flip":
            return self.pos_to_index(move.to_pos)
        if move.kind == "move" and move.from_pos is not None:
            from_idx = self.pos_to_index(move.from_pos)
            to_idx = self.pos_to_index(move.to_pos)
            return ACTION_FLIP_COUNT + from_idx * ACTION_FLIP_COUNT + to_idx
        raise ValueError(f"Cannot encode move: {move}")

    def action_to_move(self, action: int) -> Optional[Move]:
        """Map action index back to a move template."""
        if action < 0 or action >= ACTION_SPACE_SIZE:
            return None
        if action < ACTION_FLIP_COUNT:
            return Move(kind="flip", to_pos=self.index_to_pos(action))
        relative = action - ACTION_FLIP_COUNT
        from_idx = relative // ACTION_FLIP_COUNT
        to_idx = relative % ACTION_FLIP_COUNT
        if from_idx == to_idx:
            return None
        return Move(kind="move", from_pos=self.index_to_pos(from_idx), to_pos=self.index_to_pos(to_idx))

    def legal_action_mask(self) -> np.ndarray:
        """Return boolean mask over fixed action space for legal moves."""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        for move in self.get_legal_moves():
            action = self.move_to_action(move)
            mask[action] = True
        return mask

    def render_ascii(self) -> str:
        """Return a simple human-readable board representation."""
        lines: List[str] = []
        header = "    " + " ".join(f"{c:>2d}" for c in range(self.cols))
        lines.append(header)
        for row in range(self.rows):
            row_cells: List[str] = []
            for col in range(self.cols):
                cell = self.grid[row][col]
                if cell is None:
                    row_cells.append("..")
                elif isinstance(cell, HiddenPiece):
                    row_cells.append("##")
                elif isinstance(cell, Piece):
                    row_cells.append(cell.symbol)
                else:
                    row_cells.append("??")
            lines.append(f"{row:>2d}  " + " ".join(row_cells))
        return "\n".join(lines)
