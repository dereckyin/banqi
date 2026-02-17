import 'package:flutter_test/flutter_test.dart';

import 'package:flutter_banqi_app/core/engine/banqi_move.dart';
import 'package:flutter_banqi_app/core/engine/board_state.dart';
import 'package:flutter_banqi_app/core/engine/types.dart';

void main() {
  test('initial board has 32 legal flip moves', () {
    final board = BoardState.initial(seed: 42);
    final legal = board.legalMoves();
    expect(legal.length, 32);
    expect(legal.where((m) => m.kind == MoveKind.flip).length, 32);
  });

  test('horse cannot capture chariot, chariot can capture horse', () {
    final board = BoardState.initial(seed: 1)..forbidRepetition = false;
    for (var r = 0; r < BoardState.rows; r++) {
      for (var c = 0; c < BoardState.cols; c++) {
        board.grid[r][c] = const CellState.empty();
      }
    }
    board.pieceBag.clear();
    board.currentTurn = Side.red;
    board.grid[1][1] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.horse),
    );
    board.grid[1][2] = const CellState.revealed(
      Piece(side: Side.black, rank: Rank.chariot),
    );
    final redMoves = board.legalMoves();
    expect(
      redMoves.contains(const BanqiMove.move(Position(1, 1), Position(1, 2))),
      isFalse,
    );

    board.currentTurn = Side.black;
    final blackMoves = board.legalMoves();
    expect(
      blackMoves.contains(const BanqiMove.move(Position(1, 2), Position(1, 1))),
      isTrue,
    );
  });

  test('long chase limit blocks continued chasing even without repetition', () {
    final board = BoardState.initial(seed: 7)..forbidRepetition = true;
    for (var r = 0; r < BoardState.rows; r++) {
      for (var c = 0; c < BoardState.cols; c++) {
        board.grid[r][c] = const CellState.empty();
      }
    }
    board.pieceBag.clear();
    board.currentTurn = Side.red;
    board.grid[0][1] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.chariot),
    );
    board.grid[0][3] = const CellState.revealed(
      Piece(side: Side.black, rank: Rank.soldier),
    );

    board.lastNonCaptureChaseTargets[Side.red] = {const Position(0, 3)};
    board.longChaseStreak[Side.red] = board.longChaseLimit;

    final legalMoves = board.legalMoves();
    expect(
      legalMoves.contains(const BanqiMove.move(Position(0, 1), Position(0, 2))),
      isFalse,
    );
  });
}
