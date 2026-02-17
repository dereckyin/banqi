"""Minimax AI with alpha-beta pruning for Banqi."""

from __future__ import annotations

import logging
import math
import random
from typing import Dict, List, Optional, Tuple

from ai.base_ai import BaseAI
from engine.board import Board, Move
from engine.pieces import Piece, Rank, Side

LOGGER = logging.getLogger(__name__)


class MinimaxAI(BaseAI):
    """Rule-based Banqi AI."""

    def __init__(
        self,
        depth: int = 3,
        seed: Optional[int] = None,
        prefer_capture_over_flip: bool = True,
        flip_penalty: float = 3.0,
        max_branching: int = 24,
        use_transposition: bool = True,
        flip_stale_threshold: int = 8,
        exploratory_flip_bonus: float = 2.0,
        debug_top_k: int = 3,
    ) -> None:
        self.depth = depth
        self._rng = random.Random(seed)
        self.prefer_capture_over_flip = prefer_capture_over_flip
        self.flip_penalty = flip_penalty
        self.max_branching = max_branching
        self.use_transposition = use_transposition
        self.flip_stale_threshold = flip_stale_threshold
        self.exploratory_flip_bonus = exploratory_flip_bonus
        self.debug_top_k = max(1, debug_top_k)
        self._ttable: Dict[Tuple[bytes, int, str], float] = {}

    def choose_move(self, board: Board) -> Move:
        """Choose move via depth-limited alpha-beta search."""
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            raise RuntimeError("No legal moves available.")

        if self.prefer_capture_over_flip:
            capture_moves = [move for move in legal_moves if self._is_capture_move(board, move)]
            if capture_moves:
                legal_moves = capture_moves

        self._ttable.clear()
        perspective = board.current_turn
        ordered = self._order_moves(board, legal_moves)
        if self.max_branching > 0:
            ordered = ordered[: self.max_branching]
        best_tuple = (-math.inf, -math.inf)
        best_moves: List[Move] = []
        diagnostics: List[Tuple[Move, float, float, bool]] = []

        for move in ordered:
            immediate_bonus = self._immediate_tactical_bonus(board, move) + self._immediate_safety_bonus(board, move)
            child = board.clone()
            # Root legality is already filtered. Disable repetition filtering in deep search for speed.
            child.forbid_repetition = False
            child.apply_move(move)
            value = self._alphabeta(
                child,
                self.depth - 1,
                -math.inf,
                math.inf,
                perspective,
            )
            diagnostics.append((move, value, immediate_bonus, self._is_capture_move(board, move)))
            scored = (value, immediate_bonus)
            if scored > best_tuple:
                best_tuple = scored
                best_moves = [move]
            elif scored == best_tuple:
                best_moves.append(move)

        chosen = self._rng.choice(best_moves)
        self._log_diagnostics(diagnostics, chosen)
        LOGGER.debug("Minimax selected %s with score %.3f tie_bonus=%.3f", chosen, best_tuple[0], best_tuple[1])
        return chosen

    def _log_diagnostics(self, diagnostics: List[Tuple[Move, float, float, bool]], chosen: Move) -> None:
        """Emit top-k candidate breakdown when DEBUG is enabled."""
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return
        ranked = sorted(diagnostics, key=lambda item: (item[1], item[2]), reverse=True)
        top = ranked[: self.debug_top_k]
        for idx, (move, value, bonus, is_capture) in enumerate(top, start=1):
            LOGGER.debug(
                "Candidate #%d move=%s eval=%.3f bonus=%.3f capture=%s chosen=%s",
                idx,
                move,
                value,
                bonus,
                is_capture,
                move == chosen,
            )

    def _alphabeta(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        perspective: Side,
    ) -> float:
        if self.use_transposition:
            key = self._transposition_key(board, depth, perspective)
            if key in self._ttable:
                return self._ttable[key]

        terminal, winner, is_draw = board.game_over()
        if terminal:
            if is_draw:
                return 0.0
            return 10_000.0 if winner is perspective else -10_000.0
        if depth <= 0:
            score = self._evaluate(board, perspective)
            if self.use_transposition:
                self._ttable[key] = score
            return score

        legal_moves = board.get_legal_moves()
        ordered = self._order_moves(board, legal_moves)
        if self.max_branching > 0:
            ordered = ordered[: self.max_branching]
        maximizing = board.current_turn is perspective

        if maximizing:
            best = -math.inf
            for move in ordered:
                child = board.clone()
                child.forbid_repetition = False
                child.apply_move(move)
                score = self._alphabeta(child, depth - 1, alpha, beta, perspective)
                best = max(best, score)
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            if self.use_transposition:
                self._ttable[key] = best
            return best

        best = math.inf
        for move in ordered:
            child = board.clone()
            child.forbid_repetition = False
            child.apply_move(move)
            score = self._alphabeta(child, depth - 1, alpha, beta, perspective)
            best = min(best, score)
            beta = min(beta, score)
            if alpha >= beta:
                break
        if self.use_transposition:
            self._ttable[key] = best
        return best

    def _evaluate(self, board: Board, perspective: Side) -> float:
        """Heuristic evaluation from one side's perspective."""
        material = 0.0
        reveal_tempo = 0.0
        for pos in board.iter_positions():
            cell = board.get_cell(pos)
            if isinstance(cell, Piece):
                sign = 1.0 if cell.side is perspective else -1.0
                material += sign * cell.value
                reveal_tempo += sign * 0.2

        own_remaining = board.piece_count_remaining(perspective)
        opp_remaining = board.piece_count_remaining(perspective.opponent())
        remaining_balance = (own_remaining - opp_remaining) * 2.5

        mobility = self._mobility_for(board, perspective) - self._mobility_for(board, perspective.opponent())
        tactical = self._capture_opportunities(board, perspective) - self._capture_opportunities(
            board, perspective.opponent()
        )
        safety = self._safety_balance(board, perspective)
        support = self._support_balance(board, perspective)

        flip_pressure = self._flip_pressure(board, perspective) - self._flip_pressure(board, perspective.opponent())
        return (
            material
            + reveal_tempo
            + remaining_balance
            + (1.2 * mobility)
            + (2.0 * tactical)
            + (2.2 * safety)
            + (1.4 * support)
            + flip_pressure
        )

    def _mobility_for(self, board: Board, side: Side) -> int:
        state = board.clone()
        state.forbid_repetition = False
        state.current_turn = side
        return len(state.get_legal_moves())

    def _capture_opportunities(self, board: Board, side: Side) -> float:
        state = board.clone()
        state.forbid_repetition = False
        state.current_turn = side
        captures = 0.0
        for move in state.get_legal_moves():
            if move.kind != "move" or move.from_pos is None:
                continue
            target = state.get_cell(move.to_pos)
            if isinstance(target, Piece) and target.side is not side:
                # Strongly value tactical pressure on high-value targets.
                captures += 1.0 + (target.value / 100.0)
                if target.rank is Rank.GENERAL:
                    captures += 1.5
        return captures

    def _immediate_tactical_bonus(self, board: Board, move: Move) -> float:
        """Break ties in favor of forcing tactical moves."""
        if move.kind != "move":
            return self._flip_move_bonus(board)
        target = board.get_cell(move.to_pos)
        if isinstance(target, Piece):
            bonus = float(target.value)
            if target.rank is Rank.GENERAL:
                bonus += 200.0
            return bonus
        return 0.0

    def _immediate_safety_bonus(self, board: Board, move: Move) -> float:
        """Penalize obvious hanging moves and reward covered moves under pressure."""
        if move.kind != "move" or move.from_pos is None:
            return 0.0
        trial = board.clone()
        trial.forbid_repetition = False
        mover = trial.current_turn
        trial.apply_move(move)
        moved_piece = trial.get_cell(move.to_pos)
        if not isinstance(moved_piece, Piece):
            return 0.0

        enemy_attacks = self._attack_map(trial, mover.opponent())
        ally_attacks = self._attack_map(trial, mover)
        attacked = move.to_pos in enemy_attacks
        defended = move.to_pos in ally_attacks
        if attacked and not defended:
            return -1.5 * moved_piece.value
        if attacked and defended:
            return -0.3 * moved_piece.value
        if defended:
            return 0.1 * moved_piece.value
        return 0.0

    def _is_capture_move(self, board: Board, move: Move) -> bool:
        if move.kind != "move" or move.from_pos is None:
            return False
        target = board.get_cell(move.to_pos)
        return isinstance(target, Piece)

    def _flip_pressure(self, board: Board, side: Side) -> float:
        """Bias against excessive flipping, but allow flips in stale positions."""
        state = board.clone()
        state.forbid_repetition = False
        state.current_turn = side
        moves = state.get_legal_moves()
        flips = sum(1 for move in moves if move.kind == "flip")
        captures = sum(1 for move in moves if self._is_capture_move(state, move))
        if captures > 0:
            return -0.6 * self.flip_penalty

        stale_ratio = min(1.0, state.no_progress_plies / max(1, self.flip_stale_threshold))
        hidden_bonus = 0.03 * flips
        return hidden_bonus + (self.exploratory_flip_bonus * stale_ratio) - 0.2

    def _flip_move_bonus(self, board: Board) -> float:
        """
        Dynamic tie-break for flip moves.

        - If capture exists, strongly discourage flip.
        - If no capture and game is stale, encourage flipping.
        """
        state = board.clone()
        state.forbid_repetition = False
        moves = state.get_legal_moves()
        captures = sum(1 for move in moves if self._is_capture_move(state, move))
        if captures > 0:
            return -2.0 * self.flip_penalty

        stale_ratio = min(1.0, state.no_progress_plies / max(1, self.flip_stale_threshold))
        flip_candidates = sum(1 for move in moves if move.kind == "flip")
        quiet_moves = sum(
            1
            for move in moves
            if move.kind == "move" and not self._is_capture_move(state, move)
        )
        base = -0.5 * self.flip_penalty
        if quiet_moves <= 1:
            base += 1.0
        if flip_candidates > 0:
            base += 0.02 * flip_candidates
        base += stale_ratio * self.exploratory_flip_bonus
        return base

    def _attack_map(self, board: Board, side: Side) -> Dict[Tuple[int, int], List[Piece]]:
        """Map target square -> attacking pieces for one side."""
        state = board.clone()
        state.forbid_repetition = False
        state.current_turn = side
        attacks: Dict[Tuple[int, int], List[Piece]] = {}
        for move in state.get_legal_moves():
            if move.kind != "move" or move.from_pos is None:
                continue
            target = state.get_cell(move.to_pos)
            if isinstance(target, Piece) and target.side is not side:
                attacker = state.get_cell(move.from_pos)
                if isinstance(attacker, Piece):
                    attacks.setdefault(move.to_pos, []).append(attacker)
        return attacks

    def _safety_balance(self, board: Board, perspective: Side) -> float:
        """Material-weighted safety score: avoid hanging pieces, value protected pressure."""
        own_attacks = self._attack_map(board, perspective)
        opp_attacks = self._attack_map(board, perspective.opponent())
        score = 0.0
        for pos in board.iter_positions():
            piece = board.get_cell(pos)
            if not isinstance(piece, Piece):
                continue
            if piece.side is perspective:
                attacked = pos in opp_attacks
                defended = pos in own_attacks
                if attacked and not defended:
                    score -= 1.1 * piece.value
                elif attacked and defended:
                    score -= 0.35 * piece.value
            else:
                attacked = pos in own_attacks
                defended = pos in opp_attacks
                if attacked and not defended:
                    score += 0.9 * piece.value
                elif attacked and defended:
                    score += 0.2 * piece.value
        return score

    def _support_balance(self, board: Board, perspective: Side) -> float:
        """
        Reward formation: when chased, nearby allies reduce risk.
        This encourages moving toward protection under pressure.
        """
        score = 0.0
        for pos in board.iter_positions():
            piece = board.get_cell(pos)
            if not isinstance(piece, Piece):
                continue
            sign = 1.0 if piece.side is perspective else -1.0
            allies = self._adjacent_allies(board, pos, piece.side)
            enemies = self._adjacent_enemies(board, pos, piece.side)
            # If enemy is adjacent (likely chase), prefer having ally support nearby.
            if enemies > 0:
                score += sign * (0.5 * allies - 0.3 * enemies)
        return score

    def _adjacent_allies(self, board: Board, pos: Tuple[int, int], side: Side) -> int:
        row, col = pos
        neighbors = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
        count = 0
        for r, c in neighbors:
            if not (0 <= r < board.rows and 0 <= c < board.cols):
                continue
            cell = board.get_cell((r, c))
            if isinstance(cell, Piece) and cell.side is side:
                count += 1
        return count

    def _adjacent_enemies(self, board: Board, pos: Tuple[int, int], side: Side) -> int:
        row, col = pos
        neighbors = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
        count = 0
        for r, c in neighbors:
            if not (0 <= r < board.rows and 0 <= c < board.cols):
                continue
            cell = board.get_cell((r, c))
            if isinstance(cell, Piece) and cell.side is not side:
                count += 1
        return count

    def _transposition_key(self, board: Board, depth: int, perspective: Side) -> Tuple[bytes, int, str]:
        """Hashable key for cached minimax values."""
        state_bytes = board.encode_state().tobytes()
        return (state_bytes, depth, perspective.value)

    def _order_moves(self, board: Board, moves: List[Move]) -> List[Move]:
        """Sort moves to improve pruning."""

        def score(move: Move) -> Tuple[int, int]:
            if move.kind == "flip":
                return (0, 0)
            if move.from_pos is None:
                return (1, 0)
            target = board.get_cell(move.to_pos)
            if isinstance(target, Piece):
                return (3, target.value)
            return (2, 0)

        return sorted(moves, key=score, reverse=True)
