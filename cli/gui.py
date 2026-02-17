"""Tkinter desktop GUI for Banqi."""

from __future__ import annotations

import argparse
import logging
import tkinter as tk
from tkinter import messagebox
from typing import Optional, Tuple

from ai.dqn_ai import DQNAI, EpsilonSchedule
from ai.minimax_ai import MinimaxAI
from engine.board import Board, Move
from engine.pieces import HiddenPiece, Piece, Rank, Side

Position = Tuple[int, int]

BLACK_PIECE_TEXT = {
    Rank.GENERAL: "將",
    Rank.ADVISOR: "士",
    Rank.ELEPHANT: "象",
    Rank.CHARIOT: "車",
    Rank.HORSE: "馬",
    Rank.CANNON: "包",
    Rank.SOLDIER: "卒",
}

RED_PIECE_TEXT = {
    Rank.GENERAL: "帥",
    Rank.ADVISOR: "士",
    Rank.ELEPHANT: "相",
    Rank.CHARIOT: "俥",
    Rank.HORSE: "傌",
    Rank.CANNON: "砲",
    Rank.SOLDIER: "兵",
}


class BanqiGUI(tk.Tk):
    """Simple Banqi desktop interface."""

    def __init__(
        self,
        depth: int = 3,
        seed: int | None = None,
        human_side: Side = Side.RED,
        ai_type: str = "minimax",
        ai_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        self.title("Banqi AI")
        self.resizable(False, False)

        self.board = Board(seed=seed)
        self.ai_type = ai_type
        self.ai_checkpoint = ai_checkpoint
        self.ai = self._build_ai(depth=depth, seed=seed, ai_type=ai_type, ai_checkpoint=ai_checkpoint)
        self.human_side = human_side
        self.selected_from: Optional[Position] = None

        self.status_var = tk.StringVar(value="Welcome to Banqi.")
        self.turn_var = tk.StringVar(value="")

        self._build_layout()
        self._refresh_view()

    def _build_ai(self, depth: int, seed: int | None, ai_type: str, ai_checkpoint: str | None):
        if ai_type == "minimax":
            return MinimaxAI(depth=depth, seed=seed)
        if ai_type == "dqn":
            if not ai_checkpoint:
                raise ValueError("DQN mode requires --ai-checkpoint")
            agent = DQNAI(
                epsilon_schedule=EpsilonSchedule(start=0.0, end=0.0, decay=1.0),
                seed=seed,
            )
            agent.load(ai_checkpoint)
            agent.epsilon = 0.0
            return agent
        raise ValueError(f"Unsupported ai_type: {ai_type}")

    def _build_layout(self) -> None:
        outer = tk.Frame(self, padx=10, pady=10)
        outer.pack()

        tk.Label(outer, text="Banqi (Taiwanese Dark Chess)", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, columnspan=8, sticky="w", pady=(0, 6)
        )
        tk.Label(outer, textvariable=self.turn_var, anchor="w").grid(row=1, column=0, columnspan=8, sticky="w")
        tk.Label(outer, textvariable=self.status_var, anchor="w").grid(row=2, column=0, columnspan=8, sticky="w", pady=(0, 8))

        self.buttons: list[list[tk.Button]] = []
        board_frame = tk.Frame(outer, bd=1, relief=tk.SOLID)
        board_frame.grid(row=3, column=0, columnspan=8)

        for row in range(self.board.rows):
            button_row: list[tk.Button] = []
            for col in range(self.board.cols):
                btn = tk.Button(
                    board_frame,
                    text="##",
                    width=5,
                    height=2,
                    command=lambda r=row, c=col: self._on_cell_click((r, c)),
                )
                btn.grid(row=row, column=col, padx=1, pady=1)
                button_row.append(btn)
            self.buttons.append(button_row)

        tk.Button(outer, text="New Game", command=self._new_game).grid(row=4, column=0, pady=(8, 0), sticky="w")
        tk.Button(outer, text="Quit", command=self.destroy).grid(row=4, column=1, pady=(8, 0), sticky="w")

    def _new_game(self) -> None:
        seed = self.board.seed
        self.board = Board(seed=seed)
        self.selected_from = None
        self.status_var.set("New game started.")
        self._refresh_view()
        if self.board.current_turn is not self.human_side:
            self.after(200, self._ai_turn)

    def _cell_text(self, pos: Position) -> str:
        cell = self.board.get_cell(pos)
        if cell is None:
            return ".."
        if isinstance(cell, HiddenPiece):
            return "##"
        if isinstance(cell, Piece):
            return RED_PIECE_TEXT[cell.rank] if cell.side is Side.RED else BLACK_PIECE_TEXT[cell.rank]
        return "??"

    def _cell_fg(self, pos: Position) -> str:
        cell = self.board.get_cell(pos)
        if isinstance(cell, Piece):
            return "#c00000" if cell.side is Side.RED else "#003a8c"
        return "#202020"

    def _refresh_view(self) -> None:
        for row in range(self.board.rows):
            for col in range(self.board.cols):
                btn = self.buttons[row][col]
                pos = (row, col)
                btn.configure(text=self._cell_text(pos), fg=self._cell_fg(pos), bg="SystemButtonFace")

        if self.selected_from is not None:
            r, c = self.selected_from
            self.buttons[r][c].configure(bg="#ffd966")

        self.turn_var.set(
            f"Turn: {self.board.current_turn.value} | No-progress: {self.board.no_progress_plies}/{self.board.draw_no_progress_limit}"
        )

    def _piece_label(self, piece: Piece) -> str:
        return RED_PIECE_TEXT[piece.rank] if piece.side is Side.RED else BLACK_PIECE_TEXT[piece.rank]

    def _try_apply_move(self, move: Move) -> bool:
        legal_moves = self.board.get_legal_moves()
        if move not in legal_moves:
            self.status_var.set("Illegal move.")
            return False

        result = self.board.apply_move(move)
        if result.flipped_piece is not None:
            self.status_var.set(f"翻到：{self._piece_label(result.flipped_piece)}")
        elif result.captured_piece is not None:
            self.status_var.set(f"吃子：{self._piece_label(result.captured_piece)}")
        else:
            self.status_var.set("已移動。")

        self.selected_from = None
        self._refresh_view()
        self._check_game_end()
        return True

    def _check_game_end(self) -> bool:
        terminal, winner, is_draw = self.board.game_over()
        if not terminal:
            return False
        if is_draw:
            messagebox.showinfo("Game Over", "Draw.")
        else:
            messagebox.showinfo("Game Over", f"Winner: {winner.value if winner else 'none'}")
        return True

    def _on_cell_click(self, pos: Position) -> None:
        if self.board.current_turn is not self.human_side:
            self.status_var.set("Wait for AI turn.")
            return

        if self._check_game_end():
            return

        cell = self.board.get_cell(pos)

        if isinstance(cell, HiddenPiece):
            if self._try_apply_move(Move(kind="flip", to_pos=pos)):
                if not self._check_game_end() and self.board.current_turn is not self.human_side:
                    self.after(200, self._ai_turn)
            return

        if self.selected_from is None:
            if isinstance(cell, Piece) and cell.side is self.human_side:
                self.selected_from = pos
                self.status_var.set("Select destination.")
                self._refresh_view()
            else:
                self.status_var.set("Select one of your revealed pieces or flip a hidden square.")
            return

        if pos == self.selected_from:
            self.selected_from = None
            self.status_var.set("Selection cleared.")
            self._refresh_view()
            return

        move = Move(kind="move", from_pos=self.selected_from, to_pos=pos)
        if self._try_apply_move(move):
            if not self._check_game_end() and self.board.current_turn is not self.human_side:
                self.after(200, self._ai_turn)
        else:
            self._refresh_view()

    def _ai_turn(self) -> None:
        if self._check_game_end():
            return
        if self.board.current_turn is self.human_side:
            return

        move = self.ai.choose_move(self.board)
        result = self.board.apply_move(move)
        if result.flipped_piece is not None:
            self.status_var.set(f"AI 翻到：{self._piece_label(result.flipped_piece)}")
        elif result.captured_piece is not None:
            self.status_var.set(f"AI 吃子：{self._piece_label(result.captured_piece)}")
        else:
            self.status_var.set("AI 已移動。")
        self._refresh_view()
        self._check_game_end()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Banqi desktop GUI")
    parser.add_argument("--depth", type=int, default=3, help="Minimax depth")
    parser.add_argument("--ai-type", type=str, choices=["minimax", "dqn"], default="minimax", help="Opponent AI type")
    parser.add_argument("--ai-checkpoint", type=str, default=None, help="Path to DQN checkpoint when --ai-type dqn")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic game seed")
    parser.add_argument("--human-side", choices=["red", "black"], default="red", help="Human side")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    human_side = Side.RED if args.human_side == "red" else Side.BLACK
    app = BanqiGUI(
        depth=args.depth,
        seed=args.seed,
        human_side=human_side,
        ai_type=args.ai_type,
        ai_checkpoint=args.ai_checkpoint,
    )
    if app.board.current_turn is not app.human_side:
        app.after(200, app._ai_turn)
    app.mainloop()


if __name__ == "__main__":
    main()
