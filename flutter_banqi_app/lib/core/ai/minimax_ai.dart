import 'dart:math';

import '../engine/banqi_move.dart';
import '../engine/board_state.dart';
import '../engine/types.dart';
import 'base_ai.dart';

class EvaluationWeights {
  const EvaluationWeights({
    this.mobility = 1.3,
    this.tactical = 2.0,
    this.safety = 1.5,
    this.flip = 1.0,
    this.localSafety = 1.0,
    this.localFlipBias = 1.0,
  });

  final double mobility;
  final double tactical;
  final double safety;
  final double flip;
  final double localSafety;
  final double localFlipBias;

  factory EvaluationWeights.fromJson(Map<String, dynamic> json) {
    double read(String key, double fallback) {
      final value = json[key];
      if (value is num) {
        return value.toDouble();
      }
      return fallback;
    }

    return EvaluationWeights(
      mobility: read('mobility', 1.3),
      tactical: read('tactical', 2.0),
      safety: read('safety', 1.5),
      flip: read('flip', 1.0),
      localSafety: read('localSafety', 1.0),
      localFlipBias: read('localFlipBias', 1.0),
    ).clamped();
  }

  Map<String, dynamic> toJson() {
    return {
      'mobility': mobility,
      'tactical': tactical,
      'safety': safety,
      'flip': flip,
      'localSafety': localSafety,
      'localFlipBias': localFlipBias,
    };
  }

  EvaluationWeights copyWith({
    double? mobility,
    double? tactical,
    double? safety,
    double? flip,
    double? localSafety,
    double? localFlipBias,
  }) {
    return EvaluationWeights(
      mobility: mobility ?? this.mobility,
      tactical: tactical ?? this.tactical,
      safety: safety ?? this.safety,
      flip: flip ?? this.flip,
      localSafety: localSafety ?? this.localSafety,
      localFlipBias: localFlipBias ?? this.localFlipBias,
    );
  }

  EvaluationWeights clamped() {
    double clamp(double value, double minValue, double maxValue) {
      return value.clamp(minValue, maxValue).toDouble();
    }

    return EvaluationWeights(
      mobility: clamp(mobility, 0.6, 2.4),
      tactical: clamp(tactical, 0.8, 3.5),
      safety: clamp(safety, 0.7, 3.2),
      flip: clamp(flip, 0.4, 2.4),
      localSafety: clamp(localSafety, 0.6, 2.2),
      localFlipBias: clamp(localFlipBias, 0.5, 2.0),
    );
  }
}

class MinimaxAI implements BaseAI {
  MinimaxAI({
    this.depth = 2,
    int? seed,
    this.preferCaptureOverFlip = true,
    this.flipPenalty = 2.5,
    this.maxBranching = 14,
    this.quiescenceDepth = 1,
    this.seeDepth = 6,
    this.lightweightEvaluation = true,
    EvaluationWeights? evaluationWeights,
  }) : _rng = Random(seed),
       _weights = (evaluationWeights ?? const EvaluationWeights()).clamped();

  final int depth;
  final bool preferCaptureOverFlip;
  final double flipPenalty;
  final int maxBranching;
  final int quiescenceDepth;
  final int seeDepth;
  final bool lightweightEvaluation;
  final Random _rng;
  EvaluationWeights _weights;

  EvaluationWeights get weights => _weights;

  void updateWeights(EvaluationWeights next) {
    _weights = next.clamped();
  }

  @override
  BanqiMove chooseMove(BoardState board) {
    final allLegal = board.legalMoves();
    var legal = allLegal;
    if (legal.isEmpty) {
      throw StateError('No legal moves available.');
    }
    final perspective = board.currentTurn;
    final allCaptures = allLegal.where((m) => _isCaptureMove(board, m)).toList();
    final allNonCaptures = allLegal.where((m) => !_isCaptureMove(board, m)).toList();
    final allPieceMoves = allLegal.where((m) => m.kind == MoveKind.move).toList();

    if (preferCaptureOverFlip) {
      if (allCaptures.isNotEmpty) {
        // Prefer captures, but do not force obvious recapture traps.
        final safeCaptures = allCaptures
            .where((m) => !_isImmediateRecaptureTrap(board, m, perspective))
            .toList();
        if (safeCaptures.isNotEmpty) {
          legal = safeCaptures;
        } else if (allNonCaptures.isNotEmpty) {
          legal = allNonCaptures;
        } else {
          legal = allCaptures;
        }
      }
    } else if (allPieceMoves.isNotEmpty) {
      // When capture-priority is off, prevent endless flipping once movement is available.
      legal = allPieceMoves;
    }

    final hasCapture = allLegal.any((m) => _isCaptureMove(board, m));
    final hiddenPieces = _hiddenPieceCount(board);
    var baseDepth = hasCapture ? max(depth, 3) : depth;
    if (hiddenPieces <= 8) {
      baseDepth = max(baseDepth, 3);
    }
    if (hiddenPieces <= 4) {
      baseDepth = max(baseDepth, 4);
    }
    final extraDepth = legal.length <= 8 ? 1 : 0;
    final searchDepth = baseDepth + extraDepth;
    final ordered = _orderMoves(board, legal).take(maxBranching).toList();
    final safePreferred = ordered
        .where((m) => !_isImmediateRecaptureTrap(board, m, perspective))
        .toList();
    var candidates = safePreferred.isNotEmpty ? safePreferred : ordered;
    final mustHardEscape = _mustForceHardEscape(board, perspective);
    if (mustHardEscape) {
      final hardEscapeMoves = _orderMoves(board, allLegal)
          .where((m) => _isHardEscapeForThreatenedCorePiece(board, m, perspective))
          .take(maxBranching)
          .toList();
      if (hardEscapeMoves.isNotEmpty) {
        final forcedEscapes = candidates.where((m) => hardEscapeMoves.contains(m)).toList();
        candidates = forcedEscapes.isNotEmpty ? forcedEscapes : hardEscapeMoves;
      }
    }
    var bestScore = double.negativeInfinity;
    var bestBonus = double.negativeInfinity;
    final bestMoves = <BanqiMove>[];

    for (final move in candidates) {
      final child = board.clone()..forbidRepetition = false;
      child.applyMove(move);
      final value = _alphabeta(
        child,
        searchDepth - 1,
        double.negativeInfinity,
        double.infinity,
        perspective,
      );
      final risk = _recaptureRiskAdjustment(board, move, perspective);
      final see = _seeForCurrentTurnCapture(board, move);
      final preservePenalty = _corePieceMovePenalty(board, move, perspective);
      final escapeBias = _threatEscapeBias(board, move, perspective);
      final counterEscapeBias = _counterattackEscapeBias(board, move, perspective);
      final bonus = _immediateBonus(board, move);
      final chaseBias = _endgameChaseMoveBias(board, move, perspective);
      final huntBias = _surroundHuntBias(board, move, perspective);
      final moveScore =
          value +
          risk +
          (0.20 * see) +
          escapeBias +
          counterEscapeBias +
          (0.25 * bonus) +
          chaseBias +
          huntBias -
          preservePenalty;
      if (moveScore > bestScore || (moveScore == bestScore && bonus > bestBonus)) {
        bestScore = moveScore;
        bestBonus = bonus;
        bestMoves
          ..clear()
          ..add(move);
      } else if (moveScore == bestScore && bonus == bestBonus) {
        bestMoves.add(move);
      }
    }
    return bestMoves[_rng.nextInt(bestMoves.length)];
  }

  double _alphabeta(
    BoardState board,
    int remainDepth,
    double alpha,
    double beta,
    Side perspective,
  ) {
    final terminal = board.gameOver();
    if (terminal.$1) {
      if (terminal.$3) {
        return 0.0;
      }
      return terminal.$2 == perspective ? 10000.0 : -10000.0;
    }
    if (remainDepth <= 0) {
      final qBoost = _hasCaptureAvailable(board) ? 1 : 0;
      return _quiescence(board, alpha, beta, perspective, quiescenceDepth + qBoost);
    }

    final moves = _orderMoves(
      board,
      board.legalMoves(),
    ).take(maxBranching).toList();
    final maximizing = board.currentTurn == perspective;
    if (maximizing) {
      var best = double.negativeInfinity;
      for (final move in moves) {
        final child = board.clone()..forbidRepetition = false;
        child.applyMove(move);
        final score = _alphabeta(
          child,
          remainDepth - 1,
          alpha,
          beta,
          perspective,
        );
        best = max(best, score);
        alpha = max(alpha, score);
        if (alpha >= beta) {
          break;
        }
      }
      return best;
    }

    var best = double.infinity;
    for (final move in moves) {
      final child = board.clone()..forbidRepetition = false;
      child.applyMove(move);
      final score = _alphabeta(
        child,
        remainDepth - 1,
        alpha,
        beta,
        perspective,
      );
      best = min(best, score);
      beta = min(beta, score);
      if (alpha >= beta) {
        break;
      }
    }
    return best;
  }

  double _quiescence(
    BoardState board,
    double alpha,
    double beta,
    Side perspective,
    int remainQDepth,
  ) {
    final terminal = board.gameOver();
    if (terminal.$1) {
      if (terminal.$3) {
        return 0.0;
      }
      return terminal.$2 == perspective ? 10000.0 : -10000.0;
    }

    final standPat = _evaluate(board, perspective);
    if (remainQDepth <= 0) {
      return standPat;
    }

    final maximizing = board.currentTurn == perspective;
    if (maximizing) {
      if (standPat >= beta) {
        return standPat;
      }
      alpha = max(alpha, standPat);
    } else {
      if (standPat <= alpha) {
        return standPat;
      }
      beta = min(beta, standPat);
    }

    final captures = _quiescenceCaptures(board, perspective, maximizing).take(8);
    if (captures.isEmpty) {
      return standPat;
    }

    if (maximizing) {
      var best = standPat;
      for (final move in captures) {
        final child = board.clone()..forbidRepetition = false;
        child.applyMove(move);
        final score = _quiescence(
          child,
          alpha,
          beta,
          perspective,
          remainQDepth - 1,
        );
        best = max(best, score);
        alpha = max(alpha, score);
        if (alpha >= beta) {
          break;
        }
      }
      return best;
    }

    var best = standPat;
    for (final move in captures) {
      final child = board.clone()..forbidRepetition = false;
      child.applyMove(move);
      final score = _quiescence(
        child,
        alpha,
        beta,
        perspective,
        remainQDepth - 1,
      );
      best = min(best, score);
      beta = min(beta, score);
      if (alpha >= beta) {
        break;
      }
    }
    return best;
  }

  double _evaluate(BoardState board, Side perspective) {
    var material = 0.0;
    for (final pos in board.positions) {
      final piece = board.cell(pos).piece;
      if (piece == null) {
        continue;
      }
      material += piece.side == perspective ? piece.value : -piece.value;
    }

    if (lightweightEvaluation) {
      return material +
          (_weights.localSafety * _localSafety(board, perspective)) +
          (_weights.localFlipBias * _localFlipBias(board, perspective)) +
          _corePieceSafety(board, perspective) +
          _endgameProximity(board, perspective);
    }

    final mobility =
        _mobility(board, perspective) - _mobility(board, perspective.opponent);
    final tactical =
        _captureOptions(board, perspective) -
        _captureOptions(board, perspective.opponent);
    final safety = _safety(board, perspective);
    final flip =
        _flipPressure(board, perspective) -
        _flipPressure(board, perspective.opponent);
    return material +
        (_weights.mobility * mobility) +
        (_weights.tactical * tactical) +
        (_weights.safety * safety) +
        (_weights.flip * flip) +
        _corePieceSafety(board, perspective) +
        _endgameProximity(board, perspective);
  }

  double _localSafety(BoardState board, Side perspective) {
    var score = 0.0;
    for (final pos in board.positions) {
      final piece = board.cell(pos).piece;
      if (piece == null) {
        continue;
      }
      final allies = _adjacentCount(board, pos, piece.side);
      final enemies = _adjacentCount(board, pos, piece.side.opponent);
      final sign = piece.side == perspective ? 1.0 : -1.0;
      // Penalize exposed pieces and reward supported formation.
      score +=
          sign *
          ((0.12 * piece.value * allies) - (0.16 * piece.value * enemies));
    }
    return score;
  }

  int _adjacentCount(BoardState board, Position pos, Side side) {
    const dirs = <List<int>>[
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
    ];
    var count = 0;
    for (final d in dirs) {
      final nr = pos.row + d[0];
      final nc = pos.col + d[1];
      if (nr < 0 || nr >= BoardState.rows || nc < 0 || nc >= BoardState.cols) {
        continue;
      }
      final p = board.cell(Position(nr, nc)).piece;
      if (p != null && p.side == side) {
        count += 1;
      }
    }
    return count;
  }

  double _localFlipBias(BoardState board, Side perspective) {
    final sideFactor = board.currentTurn == perspective ? 1.0 : -1.0;
    final staleRatio =
        board.noProgressPlies / max(1, board.drawNoProgressLimit);
    return -0.2 * sideFactor + 0.4 * sideFactor * staleRatio;
  }

  int _mobility(BoardState board, Side side) {
    final state = board.clone()
      ..forbidRepetition = false
      ..currentTurn = side;
    return state.legalMoves().length;
  }

  double _captureOptions(BoardState board, Side side) {
    final state = board.clone()
      ..forbidRepetition = false
      ..currentTurn = side;
    var score = 0.0;
    for (final move in state.legalMoves()) {
      if (move.kind != MoveKind.move || move.from == null) {
        continue;
      }
      final target = state.cell(move.to).piece;
      if (target != null && target.side != side) {
        score += 1.0 + target.value / 100.0;
      }
    }
    return score;
  }

  double _safety(BoardState board, Side side) {
    final enemyAttacks = _attackMap(board, side.opponent);
    final ownAttacks = _attackMap(board, side);
    var score = 0.0;
    for (final pos in board.positions) {
      final piece = board.cell(pos).piece;
      if (piece == null) {
        continue;
      }
      if (piece.side == side) {
        final attacked = enemyAttacks.contains(pos);
        final defended = ownAttacks.contains(pos);
        if (attacked && !defended) {
          score -= piece.value * 0.9;
        } else if (attacked && defended) {
          score -= piece.value * 0.2;
        }
      } else {
        final attacked = ownAttacks.contains(pos);
        final defended = enemyAttacks.contains(pos);
        if (attacked && !defended) {
          score += piece.value * 0.8;
        }
      }
    }
    return score;
  }

  Set<Position> _attackMap(BoardState board, Side side) {
    final state = board.clone()
      ..forbidRepetition = false
      ..currentTurn = side;
    final attacked = <Position>{};
    for (final move in state.legalMoves()) {
      if (move.kind != MoveKind.move || move.from == null) {
        continue;
      }
      final target = state.cell(move.to).piece;
      if (target != null && target.side != side) {
        attacked.add(move.to);
      }
    }
    return attacked;
  }

  double _immediateBonus(BoardState board, BanqiMove move) {
    if (move.kind == MoveKind.flip) {
      final staleRatio =
          board.noProgressPlies / max(1, board.drawNoProgressLimit);
      return -flipPenalty + min(1.5, staleRatio);
    }
    final target = board.cell(move.to).piece;
    if (target == null) {
      return 0.0;
    }
    return target.value.toDouble();
  }

  double _recaptureRiskAdjustment(BoardState board, BanqiMove move, Side perspective) {
    if (move.kind != MoveKind.move || move.from == null) {
      return 0.0;
    }

    final child = board.clone()..forbidRepetition = false;
    child.applyMove(move);
    final landedPiece = child.cell(move.to).piece;
    if (landedPiece == null || landedPiece.side != perspective) {
      return 0.0;
    }

    final enemyThreats = _attackersTo(child, move.to, perspective.opponent);
    if (enemyThreats == 0) {
      return 0.6;
    }
    final allyDefenders = _attackersTo(child, move.to, perspective);
    final netThreat = max(1, enemyThreats - allyDefenders);
    final movedValue = landedPiece.value.toDouble();
    final capturedValue = board.cell(move.to).piece?.value.toDouble() ?? 0.0;
    final see = _seeForCurrentTurnCapture(board, move);
    final protectionTier = _protectionWeightByRank(landedPiece.rank);
    final immediateTrade = capturedValue - movedValue;
    final highValuePenalty = movedValue >= 45 ? 0.35 * movedValue * protectionTier : 0.0;
    final unfavorableTradePenalty = immediateTrade < 0
        ? (-immediateTrade * (0.55 + 0.25 * (protectionTier - 1.0)))
        : 0.0;
    final threatPenalty =
        ((0.14 * movedValue * netThreat) + (0.08 * movedValue * enemyThreats)) *
        protectionTier;
    final defenderRelief = 0.05 * movedValue * allyDefenders * (2.0 - min(1.6, protectionTier));
    final gainReward = 0.08 * capturedValue;
    final seeAdjustment = see >= 0
        ? 0.12 * see
        : -0.30 * see.abs() * protectionTier;
    return gainReward -
        threatPenalty -
        highValuePenalty -
        unfavorableTradePenalty +
        defenderRelief +
        seeAdjustment;
  }

  bool _isImmediateRecaptureTrap(BoardState board, BanqiMove move, Side perspective) {
    if (move.kind != MoveKind.move || move.from == null) {
      return false;
    }
    final moving = board.cell(move.from!).piece;
    final target = board.cell(move.to).piece;
    if (moving == null || moving.side != perspective || target == null) {
      return false;
    }
    // Focus this hard filter on valuable core pieces.
    if (moving.rank != Rank.general &&
        moving.rank != Rank.advisor &&
        moving.rank != Rank.cannon) {
      return false;
    }

    final child = board.clone()..forbidRepetition = false;
    child.applyMove(move);
    final enemyThreats = _attackersTo(child, move.to, perspective.opponent);
    if (enemyThreats <= 0) {
      return false;
    }
    final allyDefenders = _attackersTo(child, move.to, perspective);
    final netThreat = enemyThreats - allyDefenders;
    if (netThreat <= 0) {
      return false;
    }

    final movedValue = moving.value.toDouble();
    final capturedValue = target.value.toDouble();
    final see = _seeForCurrentTurnCapture(board, move);
    final tradeIsGood = capturedValue >= (0.9 * movedValue);
    // If net threatened and exchange is not favorable under SEE, treat as trap.
    return see < 0 && !tradeIsGood;
  }

  double _corePieceMovePenalty(BoardState board, BanqiMove move, Side perspective) {
    if (move.kind != MoveKind.move || move.from == null) {
      return 0.0;
    }
    final movingPiece = board.cell(move.from!).piece;
    if (movingPiece == null || movingPiece.side != perspective) {
      return 0.0;
    }
    if (movingPiece.rank != Rank.general && movingPiece.rank != Rank.advisor) {
      return 0.0;
    }
    final child = board.clone()..forbidRepetition = false;
    child.applyMove(move);
    final enemyThreats = _attackersTo(child, move.to, perspective.opponent);
    if (enemyThreats == 0) {
      return 0.0;
    }
    final allyDefenders = _attackersTo(child, move.to, perspective);
    final netThreat = max(1, enemyThreats - allyDefenders);
    final capturedValue = board.cell(move.to).piece?.value.toDouble() ?? 0.0;
    final basePenalty = movingPiece.rank == Rank.general ? 36.0 : 22.0;
    final exposurePenalty = basePenalty * netThreat;
    final gainRelief = 0.20 * capturedValue;
    return max(0.0, exposurePenalty - gainRelief);
  }

  double _threatEscapeBias(BoardState board, BanqiMove move, Side perspective) {
    if (move.kind != MoveKind.move || move.from == null) {
      return 0.0;
    }
    final moving = board.cell(move.from!).piece;
    if (moving == null || moving.side != perspective) {
      return 0.0;
    }

    final enemy = perspective.opponent;
    final beforeThreat = _attackersTo(board, move.from!, enemy) - _attackersTo(board, move.from!, perspective);
    if (beforeThreat <= 0) {
      return 0.0;
    }

    final child = board.clone()..forbidRepetition = false;
    child.applyMove(move);
    final afterThreat = _attackersTo(child, move.to, enemy) - _attackersTo(child, move.to, perspective);
    final value = moving.value.toDouble();
    var bias = (beforeThreat - afterThreat) * 0.35 * value;

    if (afterThreat <= 0) {
      // Explicit reward for successful escape from immediate threat.
      bias += 0.40 * value;
    } else if (afterThreat > beforeThreat) {
      // Penalize moves that worsen exposure when already threatened.
      bias -= 0.45 * value;
    }

    return bias;
  }

  double _counterattackEscapeBias(BoardState board, BanqiMove move, Side perspective) {
    if (move.kind != MoveKind.move || move.from == null) {
      return 0.0;
    }
    final moving = board.cell(move.from!).piece;
    final captured = board.cell(move.to).piece;
    if (moving == null || moving.side != perspective || captured == null) {
      return 0.0;
    }

    final enemy = perspective.opponent;
    final beforeThreat = _attackersTo(board, move.from!, enemy) - _attackersTo(board, move.from!, perspective);
    if (beforeThreat <= 0) {
      return 0.0;
    }

    final child = board.clone()..forbidRepetition = false;
    child.applyMove(move);
    final afterThreat = _attackersTo(child, move.to, enemy) - _attackersTo(child, move.to, perspective);
    if (afterThreat > 0) {
      return 0.0;
    }

    final see = _seeForCurrentTurnCapture(board, move);
    if (see < 0) {
      return 0.0;
    }

    // Prefer counter-capture over pure retreat when both resolve immediate danger.
    final capturedValue = captured.value.toDouble();
    final moverValue = moving.value.toDouble();
    return 0.22 * capturedValue + 0.08 * moverValue + 0.10 * see;
  }

  int _attackersTo(BoardState board, Position target, Side attacker) {
    final state = board.clone()
      ..forbidRepetition = false
      ..currentTurn = attacker;
    var count = 0;
    for (final move in state.legalMoves()) {
      if (move.kind != MoveKind.move || move.from == null || move.to != target) {
        continue;
      }
      final targetPiece = state.cell(move.to).piece;
      if (targetPiece != null && targetPiece.side != attacker) {
        count += 1;
      }
    }
    return count;
  }

  bool _hasCaptureAvailable(BoardState board) {
    for (final move in board.legalMoves()) {
      if (_isCaptureMove(board, move)) {
        return true;
      }
    }
    return false;
  }

  int _revealedPieceCount(BoardState board) {
    var count = 0;
    for (final pos in board.positions) {
      if (board.cell(pos).piece != null) {
        count += 1;
      }
    }
    return count;
  }

  int _hiddenPieceCount(BoardState board) {
    var count = 0;
    for (final pos in board.positions) {
      if (board.cell(pos).isHidden) {
        count += 1;
      }
    }
    return count;
  }

  double _endgameProximity(BoardState board, Side perspective) {
    final livePieces = _revealedPieceCount(board);
    if (livePieces > 8) {
      return 0.0;
    }
    final myPieces = <Position>[];
    final enemyPieces = <Position>[];
    for (final pos in board.positions) {
      final p = board.cell(pos).piece;
      if (p == null) {
        continue;
      }
      if (p.side == perspective) {
        myPieces.add(pos);
      } else {
        enemyPieces.add(pos);
      }
    }
    if (myPieces.isEmpty || enemyPieces.isEmpty) {
      return 0.0;
    }

    var distanceSum = 0.0;
    for (final mine in myPieces) {
      var nearest = 999;
      for (final enemy in enemyPieces) {
        final d = (mine.row - enemy.row).abs() + (mine.col - enemy.col).abs();
        if (d < nearest) {
          nearest = d;
        }
      }
      distanceSum += nearest.toDouble();
    }
    final avgDist = distanceSum / myPieces.length;
    final pressureScale = livePieces <= 4 ? 2.2 : 1.4;
    return -pressureScale * avgDist;
  }

  double _endgameChaseMoveBias(BoardState board, BanqiMove move, Side perspective) {
    final hiddenPieces = _hiddenPieceCount(board);
    final revealedPieces = _revealedPieceCount(board);
    final isLateGame = hiddenPieces <= 6 || revealedPieces <= 10;
    if (!isLateGame) {
      return 0.0;
    }
    if (move.kind != MoveKind.move || move.from == null) {
      // In late game prefer movement over unnecessary flips.
      return -0.8;
    }

    final beforeDist = _averageNearestEnemyDistance(board, perspective);
    final child = board.clone()..forbidRepetition = false;
    child.applyMove(move);
    final afterDist = _averageNearestEnemyDistance(child, perspective);
    final distDelta = beforeDist - afterDist; // positive means moving closer.

    final staleRatio = board.noProgressPlies / max(1, board.drawNoProgressLimit);
    final urgency = 1.0 + (2.0 * staleRatio);
    var bias = distDelta * 3.2 * urgency;

    // Strongly discourage drifting away when no capture exists.
    final hasCaptureNow = board.legalMoves().any((m) => _isCaptureMove(board, m));
    if (!hasCaptureNow && distDelta < 0) {
      bias += distDelta * 3.6;
    }
    return bias;
  }

  double _surroundHuntBias(BoardState board, BanqiMove move, Side perspective) {
    final hiddenPieces = _hiddenPieceCount(board);
    final revealedPieces = _revealedPieceCount(board);
    final huntPhase = hiddenPieces <= 6 || revealedPieces <= 10;
    if (!huntPhase) {
      return 0.0;
    }

    final enemyPieces = _positionsOfSide(board, perspective.opponent);
    if (enemyPieces.isEmpty) {
      return 0.0;
    }

    final target = _pickHuntTarget(board, enemyPieces, perspective);
    if (target == null) {
      return 0.0;
    }

    // During hunt phase, favor movement over flipping when movement exists.
    if (move.kind != MoveKind.move || move.from == null) {
      final hasMoveNow = board.legalMoves().any((m) => m.kind == MoveKind.move);
      return hasMoveNow ? -1.1 : 0.0;
    }

    final beforeClosest = _closestAllyDistanceToTarget(board, perspective, target);
    final beforeAdj = _adjacentAlliesToTarget(board, perspective, target);
    final beforeEscape = _targetEscapeSquares(board, target, perspective);
    final beforeCenterEscape = _targetCenterEscapeScore(board, target, perspective);

    final child = board.clone()..forbidRepetition = false;
    child.applyMove(move);
    final afterClosest = _closestAllyDistanceToTarget(child, perspective, target);
    final afterAdj = _adjacentAlliesToTarget(child, perspective, target);
    final afterEscape = _targetEscapeSquares(child, target, perspective);
    final afterCenterEscape = _targetCenterEscapeScore(child, target, perspective);

    final closeDelta = (beforeClosest - afterClosest).toDouble();
    final adjDelta = (afterAdj - beforeAdj).toDouble();
    final escapeDelta = (beforeEscape - afterEscape).toDouble();
    final cornerDelta = beforeCenterEscape - afterCenterEscape;
    final staleRatio = board.noProgressPlies / max(1, board.drawNoProgressLimit);
    final urgency = 1.0 + 1.5 * staleRatio;

    var bias = (2.1 * closeDelta + 2.6 * adjDelta + 1.8 * escapeDelta + 1.4 * cornerDelta) * urgency;

    // Extra reward for building a two-piece net around target.
    if (afterAdj >= 2 && afterAdj > beforeAdj) {
      bias += 1.8 * urgency;
    }

    return bias;
  }

  double _averageNearestEnemyDistance(BoardState board, Side side) {
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

  List<Position> _positionsOfSide(BoardState board, Side side) {
    final positions = <Position>[];
    for (final pos in board.positions) {
      final p = board.cell(pos).piece;
      if (p != null && p.side == side) {
        positions.add(pos);
      }
    }
    return positions;
  }

  Position? _pickHuntTarget(BoardState board, List<Position> enemies, Side hunter) {
    Position? best;
    var bestScore = -1e9;
    for (final pos in enemies) {
      final p = board.cell(pos).piece;
      if (p == null) {
        continue;
      }
      final dist = _closestAllyDistanceToTarget(board, hunter, pos);
      final score = (0.08 * p.value) - dist;
      if (score > bestScore) {
        bestScore = score;
        best = pos;
      }
    }
    return best;
  }

  int _closestAllyDistanceToTarget(BoardState board, Side side, Position target) {
    var nearest = 999;
    for (final pos in board.positions) {
      final p = board.cell(pos).piece;
      if (p == null || p.side != side) {
        continue;
      }
      final d = (pos.row - target.row).abs() + (pos.col - target.col).abs();
      if (d < nearest) {
        nearest = d;
      }
    }
    return nearest == 999 ? 0 : nearest;
  }

  int _adjacentAlliesToTarget(BoardState board, Side side, Position target) {
    const dirs = <List<int>>[
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
    ];
    var count = 0;
    for (final d in dirs) {
      final nr = target.row + d[0];
      final nc = target.col + d[1];
      if (nr < 0 || nr >= BoardState.rows || nc < 0 || nc >= BoardState.cols) {
        continue;
      }
      final p = board.cell(Position(nr, nc)).piece;
      if (p != null && p.side == side) {
        count += 1;
      }
    }
    return count;
  }

  int _targetEscapeSquares(BoardState board, Position target, Side hunter) {
    const dirs = <List<int>>[
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
    ];
    var count = 0;
    final hunterAttackMap = _attackMap(board, hunter);
    for (final d in dirs) {
      final nr = target.row + d[0];
      final nc = target.col + d[1];
      if (nr < 0 || nr >= BoardState.rows || nc < 0 || nc >= BoardState.cols) {
        continue;
      }
      final pos = Position(nr, nc);
      final cell = board.cell(pos);
      if (!cell.isEmpty) {
        continue;
      }
      if (hunterAttackMap.contains(pos)) {
        continue;
      }
      count += 1;
    }
    return count;
  }

  double _targetCenterEscapeScore(BoardState board, Position target, Side hunter) {
    const dirs = <List<int>>[
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
    ];
    final hunterAttackMap = _attackMap(board, hunter);
    final centerRow = (BoardState.rows - 1) / 2.0;
    final centerCol = (BoardState.cols - 1) / 2.0;
    var score = 0.0;
    for (final d in dirs) {
      final nr = target.row + d[0];
      final nc = target.col + d[1];
      if (nr < 0 || nr >= BoardState.rows || nc < 0 || nc >= BoardState.cols) {
        continue;
      }
      final pos = Position(nr, nc);
      final cell = board.cell(pos);
      if (!cell.isEmpty || hunterAttackMap.contains(pos)) {
        continue;
      }
      // Escapes toward center are more dangerous; reducing these means better corner trap.
      final centerDist = (nr - centerRow).abs() + (nc - centerCol).abs();
      score += (5.0 - centerDist).clamp(0.0, 5.0);
    }
    return score;
  }

  double _corePieceSafety(BoardState board, Side perspective) {
    var score = 0.0;
    for (final pos in board.positions) {
      final p = board.cell(pos).piece;
      if (p == null) {
        continue;
      }
      if (p.rank != Rank.general && p.rank != Rank.advisor) {
        continue;
      }
      final weight = p.rank == Rank.general ? 30.0 : 18.0;
      final enemyThreats = _attackersTo(board, pos, p.side.opponent);
      final allyDefenders = _attackersTo(board, pos, p.side);
      var local = 0.0;
      if (enemyThreats > 0) {
        local -= weight * (enemyThreats - 0.7 * allyDefenders);
      } else {
        local += 0.25 * weight;
      }
      score += p.side == perspective ? local : -local;
    }
    return score;
  }

  double _protectionWeightByRank(Rank rank) {
    switch (rank) {
      case Rank.general:
        return 1.9;
      case Rank.advisor:
        return 1.45;
      case Rank.cannon:
        return 1.25;
      default:
        return 1.0;
    }
  }

  bool _isHardEscapeForThreatenedCorePiece(
    BoardState board,
    BanqiMove move,
    Side perspective,
  ) {
    if (move.kind != MoveKind.move || move.from == null) {
      return false;
    }
    final moving = board.cell(move.from!).piece;
    if (moving == null || moving.side != perspective) {
      return false;
    }
    // Hard rule applies to General/Advisor families only.
    if (moving.rank != Rank.general && moving.rank != Rank.advisor) {
      return false;
    }

    final enemy = perspective.opponent;
    final beforeThreat =
        _attackersTo(board, move.from!, enemy) - _attackersTo(board, move.from!, perspective);
    if (beforeThreat <= 0) {
      return false;
    }

    final child = board.clone()..forbidRepetition = false;
    child.applyMove(move);
    final afterThreat =
        _attackersTo(child, move.to, enemy) - _attackersTo(child, move.to, perspective);
    return afterThreat <= 0;
  }

  bool _mustForceHardEscape(BoardState board, Side perspective) {
    // Only force "must-escape" when core piece is truly isolated and under clear net threat.
    for (final pos in board.positions) {
      final p = board.cell(pos).piece;
      if (p == null || p.side != perspective) {
        continue;
      }
      if (p.rank != Rank.general && p.rank != Rank.advisor) {
        continue;
      }
      final enemyThreats = _attackersTo(board, pos, perspective.opponent);
      if (enemyThreats <= 0) {
        continue;
      }
      final allyDefenders = _attackersTo(board, pos, perspective);
      final adjacentAllies = _adjacentCount(board, pos, perspective);
      final netThreat = enemyThreats - allyDefenders;
      // If nearby allies can cover or counterattack, do not force pure fleeing.
      final hasNearbySupport = adjacentAllies >= 1 || allyDefenders >= enemyThreats;
      if (netThreat > 0 && !hasNearbySupport) {
        return true;
      }
    }
    return false;
  }

  double _flipPressure(BoardState board, Side side) {
    final state = board.clone()
      ..forbidRepetition = false
      ..currentTurn = side;
    final moves = state.legalMoves();
    final hasCapture = moves.any((m) => _isCaptureMove(state, m));
    if (hasCapture) {
      return -flipPenalty;
    }
    return -0.1 * moves.where((m) => m.kind == MoveKind.flip).length;
  }

  bool _isCaptureMove(BoardState board, BanqiMove move) {
    if (move.kind != MoveKind.move || move.from == null) {
      return false;
    }
    return board.cell(move.to).piece != null;
  }

  List<BanqiMove> _orderMoves(BoardState board, List<BanqiMove> moves) {
    final sorted = [...moves];
    sorted.sort((a, b) => _scoreMove(board, b).compareTo(_scoreMove(board, a)));
    return sorted;
  }

  int _scoreMove(BoardState board, BanqiMove move) {
    if (move.kind == MoveKind.flip) {
      return 0;
    }
    final target = board.cell(move.to).piece;
    if (target != null) {
      final see = _seeForCurrentTurnCapture(board, move).round();
      return 300 + target.value + see;
    }
    return 100;
  }

  Iterable<BanqiMove> _quiescenceCaptures(
    BoardState board,
    Side perspective,
    bool maximizing,
  ) {
    final captures = board.legalMoves().where((m) => _isCaptureMove(board, m));
    final filtered = captures.where((move) {
      final see = _seeForCurrentTurnCapture(board, move);
      final targetValue = board.cell(move.to).piece?.value.toDouble() ?? 0.0;
      // Keep clearly sound exchanges; still allow some tactical shots on high-value targets.
      return see >= -6.0 || targetValue >= 40.0;
    }).toList();
    filtered.sort(
      (a, b) => _quiescenceCaptureScore(
        board,
        b,
        perspective,
        maximizing,
      ).compareTo(_quiescenceCaptureScore(board, a, perspective, maximizing)),
    );
    return filtered;
  }

  double _quiescenceCaptureScore(
    BoardState board,
    BanqiMove move,
    Side perspective,
    bool maximizing,
  ) {
    final mover = board.currentTurn;
    final see = _seeForCurrentTurnCapture(board, move);
    final targetValue = board.cell(move.to).piece?.value.toDouble() ?? 0.0;
    final signed = mover == perspective ? see : -see;
    final score = signed + (0.15 * targetValue);
    return maximizing ? score : -score;
  }

  double _seeForCurrentTurnCapture(BoardState board, BanqiMove move) {
    if (!_isCaptureMove(board, move) || move.from == null) {
      return 0.0;
    }
    final target = board.cell(move.to).piece;
    if (target == null) {
      return 0.0;
    }
    final child = board.clone()..forbidRepetition = false;
    child.applyMove(move);
    final replyGain = _seeSequence(
      child,
      move.to,
      board.currentTurn.opponent,
      max(0, seeDepth - 1),
    );
    return target.value.toDouble() - replyGain;
  }

  double _seeSequence(
    BoardState board,
    Position target,
    Side sideToMove,
    int remainDepth,
  ) {
    if (remainDepth <= 0) {
      return 0.0;
    }
    final capture = _leastValuableCaptureTo(board, target, sideToMove);
    if (capture == null) {
      return 0.0;
    }
    final captured = board.cell(target).piece;
    if (captured == null) {
      return 0.0;
    }

    final child = board.clone()..forbidRepetition = false;
    child.applyMove(capture);
    final continuation = _seeSequence(
      child,
      target,
      sideToMove.opponent,
      remainDepth - 1,
    );
    return max(0.0, captured.value.toDouble() - continuation);
  }

  BanqiMove? _leastValuableCaptureTo(BoardState board, Position target, Side attacker) {
    final probe = board.clone()
      ..forbidRepetition = false
      ..currentTurn = attacker;
    BanqiMove? bestMove;
    var bestAttackerValue = 1 << 30;
    var bestCapturedValue = -1;
    for (final move in probe.legalMoves()) {
      if (move.kind != MoveKind.move || move.from == null || move.to != target) {
        continue;
      }
      final attackerPiece = probe.cell(move.from!).piece;
      final victimPiece = probe.cell(target).piece;
      if (attackerPiece == null || victimPiece == null || victimPiece.side == attacker) {
        continue;
      }
      final attackerValue = attackerPiece.value;
      final victimValue = victimPiece.value;
      if (attackerValue < bestAttackerValue ||
          (attackerValue == bestAttackerValue && victimValue > bestCapturedValue)) {
        bestAttackerValue = attackerValue;
        bestCapturedValue = victimValue;
        bestMove = move;
      }
    }
    return bestMove;
  }
}
