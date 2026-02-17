"""Placeholder for future MCTS implementation."""

from __future__ import annotations

from ai.base_ai import BaseAI
from engine.board import Board, Move


class MCTSAI(BaseAI):
    """Reserved class for future Banqi MCTS agent."""

    def choose_move(self, board: Board) -> Move:
        raise NotImplementedError("MCTS AI is a placeholder in this milestone.")
