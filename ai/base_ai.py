"""Base AI interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.board import Board, Move


class BaseAI(ABC):
    """Abstract AI strategy contract."""

    @abstractmethod
    def choose_move(self, board: Board) -> Move:
        """Choose a legal move for the given board state."""
        raise NotImplementedError
