import 'package:flutter_test/flutter_test.dart';

import 'package:flutter_banqi_app/core/ai/minimax_ai.dart';
import 'package:flutter_banqi_app/core/engine/banqi_move.dart';
import 'package:flutter_banqi_app/core/engine/board_state.dart';
import 'package:flutter_banqi_app/core/engine/types.dart';

void main() {
  double avgNearestEnemyDistance(BoardState board, Side side) {
    final myPieces = <Position>[];
    final enemyPieces = <Position>[];
    for (final pos in board.positions) {
      final p = board.cell(pos).piece;
      if (p == null) {
        continue;
      }
      if (p.side == side) {
        myPieces.add(pos);
      } else {
        enemyPieces.add(pos);
      }
    }
    if (myPieces.isEmpty || enemyPieces.isEmpty) {
      return 0.0;
    }
    var sum = 0.0;
    for (final mine in myPieces) {
      var nearest = 999;
      for (final enemy in enemyPieces) {
        final d = (mine.row - enemy.row).abs() + (mine.col - enemy.col).abs();
        if (d < nearest) {
          nearest = d;
        }
      }
      sum += nearest.toDouble();
    }
    return sum / myPieces.length;
  }

  void clearBoard(BoardState board) {
    for (var r = 0; r < BoardState.rows; r++) {
      for (var c = 0; c < BoardState.cols; c++) {
        board.grid[r][c] = const CellState.empty();
        board.hiddenLayout[r][c] = null;
      }
    }
    board.pieceBag.clear();
    board.currentTurn = Side.red;
    board.noProgressPlies = 0;
    board.plyCount = 0;
    board.stateVisits.clear();
    board.lastNonCaptureChaseTargets[Side.red] = <Position>{};
    board.lastNonCaptureChaseTargets[Side.black] = <Position>{};
    board.longChaseStreak[Side.red] = 0;
    board.longChaseStreak[Side.black] = 0;
  }

  test('minimax prefers adjacent capture over flip', () {
    final board = BoardState.initial(seed: 1)..forbidRepetition = false;
    clearBoard(board);
    board.currentTurn = Side.red;
    board.grid[1][1] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.chariot),
    );
    board.grid[1][2] = const CellState.revealed(
      Piece(side: Side.black, rank: Rank.soldier),
    );
    board.grid[0][0] = const CellState.hidden();
    board.hiddenLayout[0][0] = const Piece(side: Side.black, rank: Rank.soldier);
    board.pieceBag.add(const Piece(side: Side.black, rank: Rank.soldier));

    final ai = MinimaxAI(depth: 2, seed: 42);
    final move = ai.chooseMove(board);
    expect(move, const BanqiMove.move(Position(1, 1), Position(1, 2)));
  });

  test('hard escape rule forces threatened general to a safe square', () {
    final board = BoardState.initial(seed: 7)..forbidRepetition = false;
    clearBoard(board);
    board.currentTurn = Side.red;

    board.grid[1][1] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.general),
    );
    board.grid[1][2] = const CellState.revealed(
      Piece(side: Side.black, rank: Rank.soldier),
    );
    board.grid[0][0] = const CellState.revealed(
      Piece(side: Side.black, rank: Rank.soldier),
    );

    final ai = MinimaxAI(depth: 2, seed: 42);
    final move = ai.chooseMove(board);
    expect(move, const BanqiMove.move(Position(1, 1), Position(2, 1)));
  });

  test('regression: avoid trapped high-value capture when safer capture exists', () {
    final board = BoardState.initial(seed: 8)..forbidRepetition = false;
    clearBoard(board);
    board.currentTurn = Side.red;

    // Trap candidate: red general captures black advisor, then black soldier
    // can immediately recapture the general.
    board.grid[1][1] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.general),
    );
    board.grid[1][2] = const CellState.revealed(
      Piece(side: Side.black, rank: Rank.advisor),
    );
    board.grid[1][3] = const CellState.revealed(
      Piece(side: Side.black, rank: Rank.soldier),
    );

    // Safer alternative capture.
    board.grid[2][1] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.horse),
    );
    board.grid[2][2] = const CellState.revealed(
      Piece(side: Side.black, rank: Rank.soldier),
    );

    final ai = MinimaxAI(depth: 2, seed: 42);
    final move = ai.chooseMove(board);
    expect(move, isNot(const BanqiMove.move(Position(1, 1), Position(1, 2))));
    expect(move, const BanqiMove.move(Position(2, 1), Position(2, 2)));
  });

  test('preferCaptureOverFlip does not force trap capture when alternatives exist', () {
    final board = BoardState.initial(seed: 9)..forbidRepetition = false;
    clearBoard(board);
    board.currentTurn = Side.red;

    board.grid[1][1] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.general),
    );
    board.grid[1][2] = const CellState.revealed(
      Piece(side: Side.black, rank: Rank.advisor),
    );
    board.grid[1][3] = const CellState.revealed(
      Piece(side: Side.black, rank: Rank.soldier),
    );
    board.grid[2][1] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.horse),
    );
    board.grid[0][0] = const CellState.hidden();
    board.hiddenLayout[0][0] = const Piece(side: Side.red, rank: Rank.soldier);
    board.pieceBag.add(const Piece(side: Side.red, rank: Rank.soldier));

    final ai = MinimaxAI(depth: 2, seed: 42, preferCaptureOverFlip: true);
    final move = ai.chooseMove(board);
    expect(move, isNot(const BanqiMove.move(Position(1, 1), Position(1, 2))));
  });

  test('when capture priority is off, AI still chooses movement over endless flips', () {
    final board = BoardState.initial(seed: 10)..forbidRepetition = false;
    clearBoard(board);
    board.currentTurn = Side.red;

    board.grid[1][1] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.horse),
    );
    board.grid[1][2] = const CellState.revealed(
      Piece(side: Side.black, rank: Rank.soldier),
    );
    board.grid[0][0] = const CellState.hidden();
    board.grid[0][1] = const CellState.hidden();
    board.hiddenLayout[0][0] = const Piece(side: Side.black, rank: Rank.general);
    board.hiddenLayout[0][1] = const Piece(side: Side.red, rank: Rank.soldier);
    board.pieceBag
      ..add(const Piece(side: Side.black, rank: Rank.general))
      ..add(const Piece(side: Side.red, rank: Rank.soldier));

    final ai = MinimaxAI(depth: 2, seed: 42, preferCaptureOverFlip: false);
    final move = ai.chooseMove(board);
    expect(move.kind, MoveKind.move);
  });

  test('endgame hunt mode prefers closing distance when no capture exists', () {
    final board = BoardState.initial(seed: 11)..forbidRepetition = false;
    clearBoard(board);
    board.currentTurn = Side.red;

    board.grid[0][0] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.horse),
    );
    board.grid[3][0] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.soldier),
    );
    board.grid[1][4] = const CellState.revealed(
      Piece(side: Side.black, rank: Rank.chariot),
    );

    final before = avgNearestEnemyDistance(board, Side.red);
    final ai = MinimaxAI(depth: 2, seed: 42, preferCaptureOverFlip: false);
    final move = ai.chooseMove(board);
    expect(move.kind, MoveKind.move);
    board.applyMove(move);
    final after = avgNearestEnemyDistance(board, Side.red);
    expect(after, lessThan(before));
  });

  test('supported core piece does not force flee when ally can counter-capture', () {
    final board = BoardState.initial(seed: 12)..forbidRepetition = false;
    clearBoard(board);
    board.currentTurn = Side.red;

    board.grid[1][1] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.general),
    );
    // Adjacent guard so core piece is not considered isolated.
    board.grid[2][1] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.soldier),
    );
    // Attacker threatening the general.
    board.grid[0][1] = const CellState.revealed(
      Piece(side: Side.black, rank: Rank.soldier),
    );
    // Ally counter-capture option.
    board.grid[0][2] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.chariot),
    );

    final ai = MinimaxAI(depth: 2, seed: 42, preferCaptureOverFlip: true);
    final move = ai.chooseMove(board);
    // Ally captures attacker instead of forcing the core piece to keep fleeing.
    expect(move, const BanqiMove.move(Position(0, 2), Position(0, 1)));
  });

  test('threatened core prefers safe counter-capture over plain retreat', () {
    final board = BoardState.initial(seed: 13)..forbidRepetition = false;
    clearBoard(board);
    board.currentTurn = Side.red;

    board.grid[1][1] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.advisor),
    );
    // Attacker threatening advisor.
    board.grid[1][2] = const CellState.revealed(
      Piece(side: Side.black, rank: Rank.soldier),
    );
    // Adjacent ally support to avoid hard forced-flee path.
    board.grid[2][1] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.soldier),
    );
    // Keep capture destination safe after counter-capture.
    board.grid[0][2] = const CellState.revealed(
      Piece(side: Side.red, rank: Rank.chariot),
    );

    final ai = MinimaxAI(depth: 2, seed: 42, preferCaptureOverFlip: false);
    final move = ai.chooseMove(board);
    expect(move, const BanqiMove.move(Position(1, 1), Position(1, 2)));
  });
}
