import 'package:flutter_test/flutter_test.dart';

import 'package:flutter_banqi_app/core/ai/minimax_ai.dart';
import 'package:flutter_banqi_app/core/engine/banqi_move.dart';
import 'package:flutter_banqi_app/core/engine/board_state.dart';
import 'package:flutter_banqi_app/core/engine/types.dart';

void main() {
  test('minimax prefers adjacent capture over flip', () {
    final board = BoardState.initial(seed: 1)..forbidRepetition = false;
    for (var r = 0; r < BoardState.rows; r++) {
      for (var c = 0; c < BoardState.cols; c++) {
        board.grid[r][c] = const CellState.empty();
      }
    }
    board.pieceBag
      ..clear()
      ..add(const Piece(side: Side.black, rank: Rank.soldier));
    board.currentTurn = Side.red;
    board.grid[1][1] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.chariot),
    );
    board.grid[1][2] = const CellState.revealed(
      Piece(side: Side.black, rank: Rank.soldier),
    );
    board.grid[0][0] = const CellState.hidden();

    final ai = MinimaxAI(depth: 2, seed: 42);
    final move = ai.chooseMove(board);
    expect(move, const BanqiMove.move(Position(1, 1), Position(1, 2)));
  });
}
