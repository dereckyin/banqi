"""GUI spectator mode for AI-vs-AI Banqi matches."""

from __future__ import annotations

import argparse
import logging
import tkinter as tk
from typing import Optional, Tuple

from ai.dqn_ai import DQNAI, EpsilonSchedule
from ai.minimax_ai import MinimaxAI
from engine.board import Board
from engine.pieces import HiddenPiece, Piece, Rank, Side

Position = Tuple[int, int]
LOGGER = logging.getLogger(__name__)

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


def piece_label(piece: Piece) -> str:
    return RED_PIECE_TEXT[piece.rank] if piece.side is Side.RED else BLACK_PIECE_TEXT[piece.rank]


def build_agent(ai_kind: str, depth: int, checkpoint: Optional[str], seed: Optional[int]):
    if ai_kind == "minimax":
        return MinimaxAI(depth=depth, seed=seed)
    if ai_kind == "dqn":
        agent = DQNAI(
            epsilon_schedule=EpsilonSchedule(start=0.0, end=0.0, decay=1.0),
            seed=seed,
        )
        if checkpoint:
            agent.load(checkpoint)
        agent.epsilon = 0.0
        return agent
    raise ValueError(f"Unsupported AI type: {ai_kind}")


class SelfPlaySpectatorGUI(tk.Tk):
    """Desktop spectator GUI for AI-vs-AI Banqi matches."""

    def __init__(
        self,
        red_ai_kind: str,
        black_ai_kind: str,
        red_depth: int,
        black_depth: int,
        red_checkpoint: Optional[str],
        black_checkpoint: Optional[str],
        delay_ms: int,
        games: int,
        seed: Optional[int],
    ) -> None:
        super().__init__()
        self.title("Banqi AI 觀戰模式")
        self.resizable(False, False)

        self.delay_ms = max(50, delay_ms)
        self.total_games = max(1, games)
        self.base_seed = seed
        self.current_game = 1
        self.running = True

        self.red_wins = 0
        self.black_wins = 0
        self.draws = 0

        self.red_ai_kind = red_ai_kind
        self.black_ai_kind = black_ai_kind
        self.red_depth = red_depth
        self.black_depth = black_depth
        self.red_checkpoint = red_checkpoint
        self.black_checkpoint = black_checkpoint

        self.board = Board(seed=self._game_seed())
        self.red_ai = build_agent(red_ai_kind, red_depth, red_checkpoint, seed=self._game_seed())
        self.black_ai = build_agent(black_ai_kind, black_depth, black_checkpoint, seed=None if self._game_seed() is None else self._game_seed() + 97)

        self.title_var = tk.StringVar()
        self.turn_var = tk.StringVar()
        self.status_var = tk.StringVar(value="準備開始觀戰。")
        self.score_var = tk.StringVar()

        self._build_layout()
        self._refresh_view()
        self.after(self.delay_ms, self._tick)

    def _game_seed(self) -> Optional[int]:
        if self.base_seed is None:
            return None
        return self.base_seed + self.current_game - 1

    def _build_layout(self) -> None:
        outer = tk.Frame(self, padx=10, pady=10)
        outer.pack()

        tk.Label(outer, textvariable=self.title_var, font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, columnspan=8, sticky="w", pady=(0, 6)
        )
        tk.Label(outer, textvariable=self.turn_var, anchor="w").grid(row=1, column=0, columnspan=8, sticky="w")
        tk.Label(outer, textvariable=self.status_var, anchor="w").grid(row=2, column=0, columnspan=8, sticky="w")
        tk.Label(outer, textvariable=self.score_var, anchor="w").grid(row=3, column=0, columnspan=8, sticky="w", pady=(0, 8))

        self.buttons: list[list[tk.Button]] = []
        board_frame = tk.Frame(outer, bd=1, relief=tk.SOLID)
        board_frame.grid(row=4, column=0, columnspan=8)
        for row in range(self.board.rows):
            button_row: list[tk.Button] = []
            for col in range(self.board.cols):
                btn = tk.Button(board_frame, text="##", width=5, height=2, state=tk.DISABLED)
                btn.grid(row=row, column=col, padx=1, pady=1)
                button_row.append(btn)
            self.buttons.append(button_row)

        self.toggle_btn = tk.Button(outer, text="暫停", command=self._toggle_running)
        self.toggle_btn.grid(row=5, column=0, pady=(8, 0), sticky="w")
        tk.Button(outer, text="下一局", command=self._next_game).grid(row=5, column=1, pady=(8, 0), sticky="w")
        tk.Button(outer, text="離開", command=self.destroy).grid(row=5, column=2, pady=(8, 0), sticky="w")

    def _cell_text(self, pos: Position) -> str:
        cell = self.board.get_cell(pos)
        if cell is None:
            return ".."
        if isinstance(cell, HiddenPiece):
            return "##"
        if isinstance(cell, Piece):
            return piece_label(cell)
        return "??"

    def _cell_fg(self, pos: Position) -> str:
        cell = self.board.get_cell(pos)
        if isinstance(cell, Piece):
            return "#c00000" if cell.side is Side.RED else "#003a8c"
        return "#202020"

    def _refresh_view(self) -> None:
        self.title_var.set(f"Banqi AI 觀戰 | 第 {self.current_game}/{self.total_games} 局")
        self.turn_var.set(f"目前回合：{self.board.current_turn.value} | 無進展步數：{self.board.no_progress_plies}")
        self.score_var.set(f"戰績：紅勝 {self.red_wins} / 黑勝 {self.black_wins} / 和局 {self.draws}")

        for row in range(self.board.rows):
            for col in range(self.board.cols):
                pos = (row, col)
                self.buttons[row][col].configure(
                    text=self._cell_text(pos),
                    fg=self._cell_fg(pos),
                    bg="SystemButtonFace",
                    disabledforeground=self._cell_fg(pos),
                )

    def _toggle_running(self) -> None:
        self.running = not self.running
        self.toggle_btn.configure(text="繼續" if not self.running else "暫停")
        self.status_var.set("已暫停。" if not self.running else "已繼續。")
        if self.running:
            self.after(self.delay_ms, self._tick)

    def _next_game(self) -> None:
        self.current_game += 1
        if self.current_game > self.total_games:
            self.current_game = 1
            self.red_wins = 0
            self.black_wins = 0
            self.draws = 0
        self._reset_game()
        self.status_var.set("已切換到下一局。")
        self._refresh_view()
        if self.running:
            self.after(self.delay_ms, self._tick)

    def _reset_game(self) -> None:
        seed = self._game_seed()
        self.board = Board(seed=seed)
        self.red_ai = build_agent(self.red_ai_kind, self.red_depth, self.red_checkpoint, seed=seed)
        black_seed = None if seed is None else seed + 97
        self.black_ai = build_agent(self.black_ai_kind, self.black_depth, self.black_checkpoint, seed=black_seed)

    def _tick(self) -> None:
        if not self.running:
            return

        terminal, winner, is_draw = self.board.game_over()
        if terminal:
            if is_draw:
                self.draws += 1
                self.status_var.set("本局結束：和局。")
            elif winner is Side.RED:
                self.red_wins += 1
                self.status_var.set("本局結束：紅方勝。")
            elif winner is Side.BLACK:
                self.black_wins += 1
                self.status_var.set("本局結束：黑方勝。")
            self._refresh_view()

            if self.current_game >= self.total_games:
                self.running = False
                self.toggle_btn.configure(text="繼續")
                self.status_var.set("觀戰完成。可按「下一局」重播。")
                return

            self.current_game += 1
            self._reset_game()
            self._refresh_view()
            self.after(self.delay_ms, self._tick)
            return

        actor = self.red_ai if self.board.current_turn is Side.RED else self.black_ai
        side_name = "紅方" if self.board.current_turn is Side.RED else "黑方"
        move = actor.choose_move(self.board)
        result = self.board.apply_move(move)
        if result.flipped_piece is not None:
            self.status_var.set(f"{side_name} 翻到：{piece_label(result.flipped_piece)}")
        elif result.captured_piece is not None:
            self.status_var.set(f"{side_name} 吃子：{piece_label(result.captured_piece)}")
        else:
            self.status_var.set(f"{side_name} 移動。")
        LOGGER.debug("Watch move: %s %s", side_name, move)

        self._refresh_view()
        self.after(self.delay_ms, self._tick)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Banqi AI 對奕觀戰模式")
    parser.add_argument("--red-ai", choices=["minimax", "dqn"], default="minimax", help="紅方 AI 類型")
    parser.add_argument("--black-ai", choices=["minimax", "dqn"], default="minimax", help="黑方 AI 類型")
    parser.add_argument("--red-depth", type=int, default=2, help="紅方 minimax 深度")
    parser.add_argument("--black-depth", type=int, default=2, help="黑方 minimax 深度")
    parser.add_argument("--red-checkpoint", type=str, default=None, help="紅方 DQN checkpoint 路徑")
    parser.add_argument("--black-checkpoint", type=str, default=None, help="黑方 DQN checkpoint 路徑")
    parser.add_argument("--delay-ms", type=int, default=350, help="每步播放延遲（毫秒）")
    parser.add_argument("--games", type=int, default=5, help="連續觀戰局數")
    parser.add_argument("--seed", type=int, default=None, help="基準隨機種子")
    parser.add_argument("--log-level", type=str, default="INFO", help="日誌等級")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    app = SelfPlaySpectatorGUI(
        red_ai_kind=args.red_ai,
        black_ai_kind=args.black_ai,
        red_depth=args.red_depth,
        black_depth=args.black_depth,
        red_checkpoint=args.red_checkpoint,
        black_checkpoint=args.black_checkpoint,
        delay_ms=args.delay_ms,
        games=args.games,
        seed=args.seed,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
