import 'dart:math';

import 'banqi_move.dart';
import 'types.dart';

class BoardState {
  BoardState._({
    required this.seed,
    required this.drawNoProgressLimit,
    required this.forbidRepetition,
    required this.repetitionPolicy,
    required this.longChaseLimit,
    required this.currentTurn,
    required this.noProgressPlies,
    required this.plyCount,
    required this.grid,
    required this.hiddenLayout,
    required this.pieceBag,
    required this.stateVisits,
    required this.lastNonCaptureChaseTargets,
    required this.longChaseStreak,
  });

  factory BoardState.initial({
    int rows = 4,
    int cols = 8,
    int? seed,
    int drawNoProgressLimit = 7,
    bool forbidRepetition = true,
    String repetitionPolicy = 'long_chase_only',
    int longChaseLimit = 7,
  }) {
    final rng = Random(seed);
    final grid = List.generate(
      rows,
      (_) => List.generate(cols, (_) => const CellState.hidden()),
    );
    final pieceBag = buildFullPiecePool(rng);
    final hiddenLayout = List.generate(
      rows,
      (_) => List<Piece?>.filled(cols, null),
    );
    var idx = 0;
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        final p = pieceBag[idx++];
        hiddenLayout[r][c] = Piece(side: p.side, rank: p.rank);
      }
    }
    final board = BoardState._(
      seed: seed,
      drawNoProgressLimit: drawNoProgressLimit,
      forbidRepetition: forbidRepetition,
      repetitionPolicy: repetitionPolicy,
      longChaseLimit: longChaseLimit,
      currentTurn: Side.red,
      noProgressPlies: 0,
      plyCount: 0,
      grid: grid,
      hiddenLayout: hiddenLayout,
      pieceBag: pieceBag.map((p) => Piece(side: p.side, rank: p.rank)).toList(),
      stateVisits: <String, int>{},
      lastNonCaptureChaseTargets: {
        Side.red: <Position>{},
        Side.black: <Position>{},
      },
      longChaseStreak: {Side.red: 0, Side.black: 0},
    );
    board._recordCurrentState();
    return board;
  }

  static const int rows = 4;
  static const int cols = 8;

  final int? seed;
  final int drawNoProgressLimit;
  bool forbidRepetition;
  final String repetitionPolicy;
  final int longChaseLimit;

  Side currentTurn;
  int noProgressPlies;
  int plyCount;

  final List<List<CellState>> grid;
  final List<List<Piece?>> hiddenLayout;
  final List<Piece> pieceBag;
  final Map<String, int> stateVisits;
  final Map<Side, Set<Position>> lastNonCaptureChaseTargets;
  final Map<Side, int> longChaseStreak;

  BoardState clone() {
    return BoardState._(
      seed: seed,
      drawNoProgressLimit: drawNoProgressLimit,
      forbidRepetition: forbidRepetition,
      repetitionPolicy: repetitionPolicy,
      longChaseLimit: longChaseLimit,
      currentTurn: currentTurn,
      noProgressPlies: noProgressPlies,
      plyCount: plyCount,
      grid: grid.map((row) => row.map((cell) => cell.copy()).toList()).toList(),
      hiddenLayout: hiddenLayout
          .map(
            (row) => row
                .map((p) => p == null ? null : Piece(side: p.side, rank: p.rank))
                .toList(),
          )
          .toList(),
      pieceBag: pieceBag.map((p) => Piece(side: p.side, rank: p.rank)).toList(),
      stateVisits: Map<String, int>.from(stateVisits),
      lastNonCaptureChaseTargets: {
        Side.red: Set<Position>.from(
          lastNonCaptureChaseTargets[Side.red] ?? <Position>{},
        ),
        Side.black: Set<Position>.from(
          lastNonCaptureChaseTargets[Side.black] ?? <Position>{},
        ),
      },
      longChaseStreak: {
        Side.red: longChaseStreak[Side.red] ?? 0,
        Side.black: longChaseStreak[Side.black] ?? 0,
      },
    );
  }

  Iterable<Position> get positions sync* {
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        yield Position(r, c);
      }
    }
  }

  CellState cell(Position pos) => grid[pos.row][pos.col];

  List<BanqiMove> legalMoves() {
    final pseudo = _pseudoLegalMoves();
    if (!forbidRepetition) {
      return pseudo;
    }
    return pseudo.where((m) => !_wouldRepeatPosition(m)).toList();
  }

  String? illegalReason(BanqiMove move) {
    final legal = legalMoves();
    if (legal.contains(move)) {
      return null;
    }

    if (move.kind == MoveKind.move &&
        legal.isNotEmpty &&
        legal.every((m) => m.kind == MoveKind.flip)) {
      return '此時僅可翻牌';
    }

    final pseudo = _pseudoLegalMoves();
    if (!pseudo.contains(move)) {
      if (move.kind == MoveKind.flip && !cell(move.to).isHidden) {
        return '該位置不可翻牌';
      }
      if (move.kind == MoveKind.move && move.from != null) {
        final source = cell(move.from!).piece;
        if (source == null) {
          return '起點沒有可移動的棋子';
        }
        if (source.side != currentTurn) {
          return '請選擇目前輪到的一方棋子';
        }
      }
      return '不符合棋子規則';
    }

    if (forbidRepetition && _wouldRepeatPosition(move)) {
      if (repetitionPolicy == 'long_chase_only') {
        return '違反長追限制';
      }
      return '違反重複局面限制';
    }

    return '不符合棋子規則';
  }

  List<BanqiMove> _pseudoLegalMoves() {
    final moves = <BanqiMove>[];
    for (final pos in positions) {
      if (cell(pos).isHidden) {
        moves.add(BanqiMove.flip(pos));
      }
    }
    for (final pos in positions) {
      final piece = cell(pos).piece;
      if (piece == null || piece.side != currentTurn) {
        continue;
      }
      moves.addAll(_pieceMoves(pos, piece));
    }
    return moves;
  }

  List<BanqiMove> _pieceMoves(Position from, Piece piece) {
    final moves = <BanqiMove>[];
    for (final to in _neighbors(from)) {
      final target = cell(to);
      if (target.isEmpty) {
        moves.add(BanqiMove.move(from, to));
      } else if (target.piece != null && target.piece!.side != piece.side) {
        if (_canStandardCapture(piece.rank, target.piece!.rank)) {
          moves.add(BanqiMove.move(from, to));
        }
      }
    }

    if (piece.rank == Rank.cannon) {
      moves.addAll(_cannonCaptureMoves(from, piece.side));
    }
    return moves;
  }

  List<BanqiMove> _cannonCaptureMoves(Position from, Side side) {
    final moves = <BanqiMove>[];
    const directions = <List<int>>[
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
    ];
    for (final d in directions) {
      var r = from.row + d[0];
      var c = from.col + d[1];
      var screenSeen = false;
      while (_inBounds(r, c)) {
        final state = grid[r][c];
        if (state.isEmpty) {
          r += d[0];
          c += d[1];
          continue;
        }
        if (!screenSeen) {
          screenSeen = true;
          r += d[0];
          c += d[1];
          continue;
        }
        if (state.piece != null && state.piece!.side != side) {
          moves.add(BanqiMove.move(from, Position(r, c)));
        }
        break;
      }
    }
    return moves;
  }

  MoveResult applyMove(BanqiMove move) {
    if (!legalMoves().contains(move)) {
      throw StateError('Illegal move: $move');
    }
    final mover = currentTurn;
    final raw = _applyMoveNoValidation(move);

    if (move.kind == MoveKind.move && raw.capturedPiece == null) {
      final currentTargets = _captureTargetsFrom(move.to, mover);
      final previousTargets = lastNonCaptureChaseTargets[mover] ?? <Position>{};
      if (currentTargets.isNotEmpty) {
        final continuingChase =
            previousTargets.isEmpty ||
            currentTargets.any((p) => previousTargets.contains(p));
        longChaseStreak[mover] = continuingChase
            ? (longChaseStreak[mover] ?? 0) + 1
            : 1;
      } else {
        longChaseStreak[mover] = 0;
      }
      lastNonCaptureChaseTargets[mover] = currentTargets;
    } else {
      lastNonCaptureChaseTargets[mover] = <Position>{};
      longChaseStreak[mover] = 0;
    }

    plyCount += 1;
    currentTurn = currentTurn.opponent;
    _recordCurrentState();

    final terminal = gameOver();
    return MoveResult(
      capturedPiece: raw.capturedPiece,
      flippedPiece: raw.flippedPiece,
      winner: terminal.$1 ? terminal.$2 : null,
      isDraw: terminal.$1 ? terminal.$3 : false,
    );
  }

  ({Piece? capturedPiece, Piece? flippedPiece}) _applyMoveNoValidation(
    BanqiMove move,
  ) {
    Piece? captured;
    Piece? flipped;

    if (move.kind == MoveKind.flip) {
      final fixed = hiddenLayout[move.to.row][move.to.col];
      if (fixed == null) {
        throw StateError('No fixed piece available for this hidden square.');
      }
      flipped = Piece(side: fixed.side, rank: fixed.rank);
      hiddenLayout[move.to.row][move.to.col] = null;
      final bagIdx = pieceBag.indexWhere(
        (p) => p.side == flipped!.side && p.rank == flipped.rank,
      );
      if (bagIdx >= 0) {
        pieceBag.removeAt(bagIdx);
      }
      grid[move.to.row][move.to.col] = CellState.revealed(flipped);
      noProgressPlies = 0;
      return (capturedPiece: captured, flippedPiece: flipped);
    }

    final from = move.from!;
    final source = grid[from.row][from.col].piece;
    if (source == null) {
      throw StateError('Source square has no movable piece.');
    }
    final target = grid[move.to.row][move.to.col].piece;
    if (target != null) {
      captured = target;
      noProgressPlies = 0;
    } else {
      noProgressPlies += 1;
    }
    grid[from.row][from.col] = const CellState.empty();
    grid[move.to.row][move.to.col] = CellState.revealed(source);
    return (capturedPiece: captured, flippedPiece: flipped);
  }

  (bool, Side?, bool) gameOver() {
    if (noProgressPlies >= drawNoProgressLimit) {
      return (true, null, true);
    }
    final redRemaining = pieceCountRemaining(Side.red);
    final blackRemaining = pieceCountRemaining(Side.black);
    if (redRemaining == 0 && blackRemaining == 0) {
      return (true, null, true);
    }
    if (redRemaining == 0) {
      return (true, Side.black, false);
    }
    if (blackRemaining == 0) {
      return (true, Side.red, false);
    }
    if (legalMoves().isEmpty) {
      return (true, currentTurn.opponent, false);
    }
    return (false, null, false);
  }

  int pieceCountRemaining(Side side) {
    var onBoard = 0;
    for (final pos in positions) {
      final p = cell(pos).piece;
      if (p != null && p.side == side) {
        onBoard += 1;
      }
    }
    final inBag = pieceBag.where((p) => p.side == side).length;
    return onBoard + inBag;
  }

  List<double> encodeState() {
    const channels = 17;
    final data = List<double>.filled(channels * rows * cols, 0.0);

    void setAt(int ch, int row, int col, double value) {
      final index = ch * rows * cols + row * cols + col;
      data[index] = value;
    }

    final sideToMove = currentTurn == Side.red ? 1.0 : 0.0;
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        setAt(0, r, c, sideToMove);
      }
    }

    final rankOrder = Rank.values;
    for (final pos in positions) {
      final state = cell(pos);
      if (state.isHidden) {
        setAt(1, pos.row, pos.col, 1.0);
      } else if (state.piece != null) {
        final rankIdx = rankOrder.indexOf(state.piece!.rank);
        final base = state.piece!.side == Side.red ? 2 : 9;
        setAt(base + rankIdx, pos.row, pos.col, 1.0);
      }
    }

    final progressRatio = min(
      1.0,
      noProgressPlies / max(1, drawNoProgressLimit),
    );
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        setAt(16, r, c, progressRatio);
      }
    }
    return data;
  }

  bool _wouldRepeatPosition(BanqiMove move) {
    final trial = clone()..forbidRepetition = false;
    final mover = trial.currentTurn;
    final raw = trial._applyMoveNoValidation(move);
    trial.plyCount += 1;
    trial.currentTurn = trial.currentTurn.opponent;
    final nextKey = trial._stateKey();
    final repeated = stateVisits.containsKey(nextKey);
    if (repetitionPolicy == 'off') {
      return false;
    }

    if (repetitionPolicy == 'long_chase_only') {
      if (move.kind == MoveKind.move && raw.capturedPiece == null) {
        final currentTargets = trial._captureTargetsFrom(move.to, mover);
        final previousTargets = lastNonCaptureChaseTargets[mover] ?? <Position>{};
        if (currentTargets.isNotEmpty) {
          final continuingChase =
              previousTargets.isEmpty ||
              currentTargets.any((p) => previousTargets.contains(p));
          final nextStreak = continuingChase
              ? (longChaseStreak[mover] ?? 0) + 1
              : 1;
          // Block as soon as streak reaches configured limit.
          if (continuingChase && nextStreak >= longChaseLimit) {
            return true;
          }
        }
      }
      // In long-chase-only policy, generic repeated positions are allowed.
      return false;
    }

    if (!repeated) {
      return false;
    }
    if (repetitionPolicy == 'superko') {
      return true;
    }
    return true;
  }

  Set<Position> _captureTargetsFrom(Position from, Side attackerSide) {
    final piece = cell(from).piece;
    if (piece == null || piece.side != attackerSide) {
      return <Position>{};
    }
    final targets = <Position>{};
    for (final move in _pieceMoves(from, piece)) {
      if (move.kind != MoveKind.move || move.from == null) {
        continue;
      }
      final target = cell(move.to).piece;
      if (target != null && target.side != attackerSide) {
        targets.add(move.to);
      }
    }
    return targets;
  }

  String _stateKey() {
    final buffer = StringBuffer()
      ..write(currentTurn.name)
      ..write('|');
    for (final pos in positions) {
      final state = cell(pos);
      if (state.isHidden) {
        final hp = hiddenLayout[pos.row][pos.col];
        if (hp == null) {
          buffer.write('#');
        } else {
          buffer.write('${hp.side.name[0]}${hp.rank.name[0]}');
        }
      } else if (state.piece == null) {
        buffer.write('.');
      } else {
        buffer.write(
          '${state.piece!.side.name[0]}${state.piece!.rank.name[0]}',
        );
      }
      buffer.write(',');
    }
    buffer.write('|');
    for (final p in pieceBag) {
      buffer.write('${p.side.name[0]}${p.rank.name[0]},');
    }
    return buffer.toString();
  }

  void _recordCurrentState() {
    final key = _stateKey();
    stateVisits[key] = (stateVisits[key] ?? 0) + 1;
  }

  bool _canStandardCapture(Rank attacker, Rank defender) {
    if (attacker == Rank.cannon) {
      return false;
    }
    if (attacker == Rank.general && defender == Rank.soldier) {
      return false;
    }
    if (attacker == Rank.soldier && defender == Rank.general) {
      return true;
    }
    return (rankStrength[attacker] ?? 0) >= (rankStrength[defender] ?? 0);
  }

  bool _inBounds(int row, int col) =>
      row >= 0 && row < rows && col >= 0 && col < cols;

  Iterable<Position> _neighbors(Position pos) sync* {
    const dirs = <List<int>>[
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
    ];
    for (final d in dirs) {
      final nr = pos.row + d[0];
      final nc = pos.col + d[1];
      if (_inBounds(nr, nc)) {
        yield Position(nr, nc);
      }
    }
  }
}
