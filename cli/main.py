"""CLI entrypoint for playing Banqi against AI."""

from __future__ import annotations

import argparse
import logging
from typing import Optional

from ai.minimax_ai import MinimaxAI
from engine.board import Board, Move
from engine.pieces import Side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Taiwanese Banqi in terminal.")
    parser.add_argument("--depth", type=int, default=3, help="Minimax depth")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic game seed")
    parser.add_argument(
        "--human-side",
        type=str,
        default="red",
        choices=["red", "black"],
        help="Which side the human controls",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level")
    return parser.parse_args()


def parse_user_move(command: str) -> Optional[Move]:
    parts = command.strip().split()
    if not parts:
        return None

    op = parts[0].lower()
    if op == "flip" and len(parts) == 3:
        row, col = int(parts[1]), int(parts[2])
        return Move(kind="flip", to_pos=(row, col))
    if op == "move" and len(parts) == 5:
        r1, c1, r2, c2 = map(int, parts[1:])
        return Move(kind="move", from_pos=(r1, c1), to_pos=(r2, c2))
    return None


def run_cli() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger("banqi.cli")

    board = Board(seed=args.seed)
    ai = MinimaxAI(depth=args.depth, seed=args.seed)
    human_side = Side.RED if args.human_side == "red" else Side.BLACK

    logger.info("Starting Banqi game. Human=%s AI=%s", human_side.value, human_side.opponent().value)
    print("Commands: flip <row> <col> | move <r1> <c1> <r2> <c2> | help | quit")

    while True:
        terminal, winner, is_draw = board.game_over()
        print()
        print(board.render_ascii())
        print(f"Turn: {board.current_turn.value} | No-progress: {board.no_progress_plies}/{board.draw_no_progress_limit}")

        if terminal:
            if is_draw:
                print("Game ended in draw.")
            else:
                print(f"Winner: {winner.value if winner else 'none'}")
            break

        if board.current_turn is human_side:
            user_input = input("Your move> ").strip()
            if user_input.lower() in {"quit", "exit"}:
                print("Exiting game.")
                break
            if user_input.lower() == "help":
                print("Commands: flip <row> <col> | move <r1> <c1> <r2> <c2> | quit")
                continue

            try:
                move = parse_user_move(user_input)
                if move is None:
                    print("Invalid command format.")
                    continue
                if move not in board.get_legal_moves():
                    print("Illegal move for current state.")
                    continue
                result = board.apply_move(move)
                if result.flipped_piece is not None:
                    print(f"You flipped: {result.flipped_piece.symbol}")
                elif result.captured_piece is not None:
                    print(f"You captured: {result.captured_piece.symbol}")
            except ValueError:
                print("Invalid numeric input.")
                continue
        else:
            ai_move = ai.choose_move(board)
            result = board.apply_move(ai_move)
            print(f"AI move: {ai_move}")
            if result.flipped_piece is not None:
                print(f"AI flipped: {result.flipped_piece.symbol}")
            elif result.captured_piece is not None:
                print(f"AI captured: {result.captured_piece.symbol}")


if __name__ == "__main__":
    run_cli()
