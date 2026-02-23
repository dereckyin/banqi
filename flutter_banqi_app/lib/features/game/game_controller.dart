import 'package:flutter/foundation.dart';

import '../../core/ai/minimax_ai.dart';
import '../../core/engine/banqi_move.dart';
import '../../core/engine/board_state.dart';
import '../../core/engine/types.dart';
import '../stats/learning_stats_service.dart';

class _GameSnapshot {
  const _GameSnapshot({
    required this.board,
    required this.humanSide,
    required this.sideLocked,
    required this.lastMove,
    required this.moveAnimTick,
    required this.playerMoveKind,
  });

  final BoardState board;
  final Side humanSide;
  final bool sideLocked;
  final BanqiMove? lastMove;
  final int moveAnimTick;
  final MoveKind playerMoveKind;
}

class _PendingStrategyResult {
  const _PendingStrategyResult({
    required this.aiSide,
    required this.winner,
    required this.isDraw,
  });

  final Side aiSide;
  final Side? winner;
  final bool isDraw;
}

class GameController extends ChangeNotifier {
  GameController({
    this.seed,
    int depth = 2,
    Side humanSide = Side.red,
    int longChaseLimit = 7,
    int drawNoProgressLimit = 50,
    int aiMoveDelayMs = 500,
  }) : _humanSide = humanSide,
       _preferredHumanSide = humanSide,
       _depth = depth,
       _longChaseLimit = longChaseLimit,
       _drawNoProgressLimit = drawNoProgressLimit,
       _aiMoveDelayMs = aiMoveDelayMs {
    _newGame();
  }

  final int? seed;
  late BoardState board;
  late MinimaxAI _ai;
  Position? selected;
  BanqiMove? lastMove;
  String status = '準備開始';
  bool thinking = false;
  bool spectatorMode = false;
  bool _actionLock = false;
  bool _sideLocked = false;
  int _moveAnimTick = 0;
  bool _learningCommittedForCurrentGame = false;
  final LearningStatsService _learning = LearningStatsService.instance;
  _GameSnapshot? _playerTurnSnapshot;
  bool _undoReady = false;
  _PendingStrategyResult? _pendingStrategyResult;

  Side _humanSide;
  Side _preferredHumanSide;
  int _depth;
  int _longChaseLimit;
  int _drawNoProgressLimit;
  int _aiMoveDelayMs;

  Side get humanSide => _humanSide;
  int get depth => _depth;
  int get longChaseLimit => _longChaseLimit;
  int get drawNoProgressLimit => _drawNoProgressLimit;
  int get aiMoveDelayMs => _aiMoveDelayMs;
  int get moveAnimTick => _moveAnimTick;
  AdaptiveLearningState get learningState => _learning.state;
  EvaluationWeights get learningWeights => _learning.currentWeights;
  double get recentScoreRate {
    final outcomes = _learning.state.recentOutcomes;
    if (outcomes.isEmpty) {
      return 0.0;
    }
    final score = outcomes.fold<double>(
      0.0,
      (sum, v) => sum + (v == 1 ? 1.0 : (v == 0 ? 0.5 : 0.0)),
    );
    return score / outcomes.length;
  }
  String get redCapturedSummary => _capturedSummary(Side.red);
  String get blackCapturedSummary => _capturedSummary(Side.black);
  bool get canApplyStrategyAdjustment =>
      _pendingStrategyResult != null && !_learningCommittedForCurrentGame;
  bool get canUndo =>
      _undoReady &&
      _playerTurnSnapshot != null &&
      !_learningCommittedForCurrentGame &&
      !board.gameOver().$1;

  void updateSettings({
    required int depth,
    required Side humanSide,
    required int longChaseLimit,
    required int drawNoProgressLimit,
    int? aiMoveDelayMs,
  }) {
    _depth = depth;
    _preferredHumanSide = humanSide;
    _humanSide = humanSide;
    _longChaseLimit = longChaseLimit;
    _drawNoProgressLimit = drawNoProgressLimit;
    if (aiMoveDelayMs != null) {
      _aiMoveDelayMs = aiMoveDelayMs;
    }
    _newGame();
  }

  void restart() => _newGame();

  void stopSpectatorMode() {
    if (!spectatorMode) {
      return;
    }
    spectatorMode = false;
    thinking = false;
    status = '已停止 AI 對戰';
    notifyListeners();
  }

  Future<void> applyStrategyAdjustmentForLastGame() async {
    final pending = _pendingStrategyResult;
    if (pending == null || _learningCommittedForCurrentGame) {
      return;
    }
    await _learning.updateAfterGame(
      aiSide: pending.aiSide,
      winner: pending.winner,
      isDraw: pending.isDraw,
    );
    _ai.updateWeights(_learning.currentWeights);
    _learningCommittedForCurrentGame = true;
    _pendingStrategyResult = null;
    status = '已依上一局調整策略';
    notifyListeners();
  }

  void undoOneStep() {
    if (!canUndo || thinking || spectatorMode || _actionLock) {
      return;
    }
    final snapshot = _playerTurnSnapshot!;
    _playerTurnSnapshot = null;
    _undoReady = false;
    board = snapshot.board.clone();
    _humanSide = snapshot.humanSide;
    _sideLocked = snapshot.sideLocked;
    selected = null;
    lastMove = snapshot.lastMove;
    _moveAnimTick = snapshot.moveAnimTick;
    thinking = false;
    status = '已悔棋一步';
    notifyListeners();
  }

  Future<void> tap(Position pos) async {
    if (thinking || spectatorMode || _actionLock) {
      if (thinking || spectatorMode) {
        status = '請等待目前回合完成';
        notifyListeners();
      }
      return;
    }
    final terminal = board.gameOver();
    if (terminal.$1 || (_sideLocked && board.currentTurn != _humanSide)) {
      return;
    }

    final cell = board.cell(pos);
    if (cell.isHidden) {
      _actionLock = true;
      try {
        final ok = _tryMove(BanqiMove.flip(pos));
        if (ok) {
          await _runAiTurnIfNeeded();
        }
      } finally {
        _actionLock = false;
      }
      return;
    }

    if (selected == null) {
      if (cell.piece != null && cell.piece!.side == _humanSide) {
        selected = pos;
        status = '已選取棋子，請選目的地';
        notifyListeners();
      } else {
        status = '請選擇自己的明棋或翻牌';
        notifyListeners();
      }
      return;
    }

    if (selected == pos) {
      selected = null;
      status = '取消選取';
      notifyListeners();
      return;
    }

    if (cell.piece != null && cell.piece!.side == _humanSide) {
      selected = pos;
      status = '已切換選取棋子';
      notifyListeners();
      return;
    }

    final ok = _tryMove(BanqiMove.move(selected!, pos));
    _actionLock = true;
    try {
      if (ok) {
        selected = null;
        await _runAiTurnIfNeeded();
      } else {
        status = '非法走法';
        notifyListeners();
      }
    } finally {
      _actionLock = false;
    }
  }

  bool _tryMove(BanqiMove move) {
    try {
      _playerTurnSnapshot = _GameSnapshot(
        board: board.clone(),
        humanSide: _humanSide,
        sideLocked: _sideLocked,
        lastMove: lastMove,
        moveAnimTick: _moveAnimTick,
        playerMoveKind: move.kind,
      );
      _undoReady = false;
      final result = board.applyMove(move);
      selected = null;
      lastMove = move;
      _moveAnimTick += 1;
      if (result.flippedPiece != null) {
        if (!_sideLocked) {
          _humanSide = result.flippedPiece!.side;
          _sideLocked = true;
          // First flip decides player color only; keep turn alternation stable.
          // If board turn now equals player side (e.g. flipped black on first move),
          // switch to opponent so AI can respond immediately.
          if (board.currentTurn == _humanSide) {
            board.currentTurn = board.currentTurn.opponent;
          }
          status = _humanSide == Side.red
              ? '翻到：${result.flippedPiece!.glyph}，你是紅方先手'
              : '翻到：${result.flippedPiece!.glyph}，你是黑方';
        } else {
          status = '翻到：${result.flippedPiece!.glyph}';
        }
      } else if (result.capturedPiece != null) {
        status = '吃子：${result.capturedPiece!.glyph}';
      } else {
        status = '已移動';
      }
      notifyListeners();
      return true;
    } catch (_) {
      status = board.illegalReason(move) ?? '非法走法';
      notifyListeners();
      return false;
    }
  }

  Future<void> _runAiTurnIfNeeded() async {
    final terminal = board.gameOver();
    if (terminal.$1) {
      await _learnFromTerminal(terminal);
      notifyListeners();
      return;
    }
    if (board.currentTurn == _humanSide) {
      notifyListeners();
      return;
    }

    thinking = true;
    status = 'AI 思考中...';
    notifyListeners();

    // Keep a tiny pacing gap even when AI computes instantly.
    await Future<void>.delayed(Duration(milliseconds: _aiMoveDelayMs));
    final move = _ai.chooseMove(board);
    final result = board.applyMove(move);
    lastMove = move;
    _moveAnimTick += 1;
    if (result.flippedPiece != null) {
      status = 'AI 翻到：${result.flippedPiece!.glyph}';
    } else if (result.capturedPiece != null) {
      status = 'AI 吃子：${result.capturedPiece!.glyph}';
    } else {
      status = 'AI 已移動';
    }
    final playerMovedPiece = _playerTurnSnapshot?.playerMoveKind == MoveKind.move;
    _undoReady = (result.capturedPiece?.side == _humanSide) && playerMovedPiece;
    if (!_undoReady) {
      _playerTurnSnapshot = null;
    }
    await _learnFromTerminal(board.gameOver());
    thinking = false;
    notifyListeners();
  }

  void _newGame() {
    board = BoardState.initial(
      seed: seed,
      drawNoProgressLimit: _drawNoProgressLimit,
      longChaseLimit: _longChaseLimit,
      repetitionPolicy: 'long_chase_only',
      forbidRepetition: true,
    );
    _ai = MinimaxAI(
      depth: _depth,
      seed: seed,
      evaluationWeights: _learning.currentWeights,
    );
    _humanSide = _preferredHumanSide;
    selected = null;
    lastMove = null;
    _moveAnimTick = 0;
    _learningCommittedForCurrentGame = false;
    _playerTurnSnapshot = null;
    _undoReady = false;
    _pendingStrategyResult = null;
    thinking = false;
    spectatorMode = false;
    _actionLock = false;
    _sideLocked = false;
    status = '新對局開始，請先翻一顆棋決定你是紅方或黑方';
    notifyListeners();
  }

  Future<void> watchOneAiVsAiGame({int delayMs = 220}) async {
    _newGame();
    spectatorMode = true;
    _sideLocked = true;
    status = '觀戰模式：電腦對戰中...';
    notifyListeners();

    while (spectatorMode) {
      final terminal = board.gameOver();
      if (terminal.$1) {
        final perspective = _learning.state.totalGames.isEven ? Side.red : Side.black;
        await _learnFromTerminal(terminal, aiSide: perspective);
        status = terminal.$3
            ? '本局結果：和局（自動下一局）'
            : '本局結果：${terminal.$2 == Side.red ? '紅方勝' : '黑方勝'}（自動下一局）';
        thinking = false;
        notifyListeners();
        await Future<void>.delayed(const Duration(milliseconds: 320));
        if (!spectatorMode) {
          return;
        }
        board = BoardState.initial(
          seed: seed,
          drawNoProgressLimit: _drawNoProgressLimit,
          longChaseLimit: _longChaseLimit,
          repetitionPolicy: 'long_chase_only',
          forbidRepetition: true,
        );
        _ai = MinimaxAI(
          depth: _depth,
          seed: seed,
          evaluationWeights: _learning.currentWeights,
        );
        selected = null;
        lastMove = null;
        _moveAnimTick = 0;
        _learningCommittedForCurrentGame = false;
        _playerTurnSnapshot = null;
        _undoReady = false;
        status = '觀戰模式：電腦對戰中...';
        notifyListeners();
        continue;
      }

      thinking = true;
      notifyListeners();
      await Future<void>.delayed(Duration(milliseconds: delayMs));
      if (!spectatorMode) {
        thinking = false;
        notifyListeners();
        return;
      }
      final move = _ai.chooseMove(board);
      final result = board.applyMove(move);
      lastMove = move;
      _moveAnimTick += 1;
      if (result.flippedPiece != null) {
        status = 'AI 翻到：${result.flippedPiece!.glyph}';
      } else if (result.capturedPiece != null) {
        status = 'AI 吃子：${result.capturedPiece!.glyph}';
      } else {
        status = 'AI 已移動';
      }
      await _learnFromTerminal(board.gameOver(), aiSide: Side.red);
      thinking = false;
      notifyListeners();
    }
  }

  Future<void> _learnFromTerminal(
    (bool, Side?, bool) terminal, {
    Side? aiSide,
  }) async {
    if (!terminal.$1 || _learningCommittedForCurrentGame) {
      return;
    }
    _pendingStrategyResult = _PendingStrategyResult(
      aiSide: aiSide ?? _humanSide.opponent,
      winner: terminal.$2,
      isDraw: terminal.$3,
    );
    status = '本局已結束，可按「策略調整」套用學習';
  }

  String _capturedSummary(Side side) {
    final remaining = <Rank, int>{for (final r in Rank.values) r: 0};
    for (final pos in board.positions) {
      final piece = board.cell(pos).piece;
      if (piece != null && piece.side == side) {
        remaining[piece.rank] = (remaining[piece.rank] ?? 0) + 1;
      }
    }
    for (final piece in board.pieceBag) {
      if (piece.side == side) {
        remaining[piece.rank] = (remaining[piece.rank] ?? 0) + 1;
      }
    }

    final parts = <String>[];
    for (final rank in Rank.values) {
      final total = rankCounts[rank] ?? 0;
      final alive = remaining[rank] ?? 0;
      final captured = total - alive;
      if (captured <= 0) {
        continue;
      }
      final glyph = side == Side.red ? (redGlyph[rank] ?? '?') : (blackGlyph[rank] ?? '?');
      parts.add('$glyph×$captured');
    }
    return parts.isEmpty ? '無' : parts.join(' ');
  }
}
