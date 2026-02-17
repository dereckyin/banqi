import 'dart:convert';
import 'dart:math';

import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../core/ai/minimax_ai.dart';
import '../../core/engine/types.dart';

class AdaptiveLearningState {
  const AdaptiveLearningState({
    required this.totalGames,
    required this.aiWins,
    required this.aiLosses,
    required this.draws,
    required this.learningRate,
    required this.weights,
    required this.recentOutcomes,
  });

  final int totalGames;
  final int aiWins;
  final int aiLosses;
  final int draws;
  final double learningRate;
  final EvaluationWeights weights;
  final List<int> recentOutcomes;

  factory AdaptiveLearningState.initial() {
    return const AdaptiveLearningState(
      totalGames: 0,
      aiWins: 0,
      aiLosses: 0,
      draws: 0,
      learningRate: 0.06,
      weights: EvaluationWeights(),
      recentOutcomes: <int>[],
    );
  }

  factory AdaptiveLearningState.fromJson(Map<String, dynamic> json) {
    final outcomesDynamic = json['recentOutcomes'];
    final outcomes = <int>[];
    if (outcomesDynamic is List) {
      for (final value in outcomesDynamic) {
        if (value is num) {
          outcomes.add(value.toInt().clamp(-1, 1));
        }
      }
    }
    return AdaptiveLearningState(
      totalGames: (json['totalGames'] as num?)?.toInt() ?? 0,
      aiWins: (json['aiWins'] as num?)?.toInt() ?? 0,
      aiLosses: (json['aiLosses'] as num?)?.toInt() ?? 0,
      draws: (json['draws'] as num?)?.toInt() ?? 0,
      learningRate: ((json['learningRate'] as num?)?.toDouble() ?? 0.06).clamp(0.01, 0.2),
      weights: EvaluationWeights.fromJson(
        (json['weights'] is Map<String, dynamic>)
            ? json['weights'] as Map<String, dynamic>
            : <String, dynamic>{},
      ),
      recentOutcomes: outcomes.take(20).toList(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'totalGames': totalGames,
      'aiWins': aiWins,
      'aiLosses': aiLosses,
      'draws': draws,
      'learningRate': learningRate,
      'weights': weights.toJson(),
      'recentOutcomes': recentOutcomes,
    };
  }

  AdaptiveLearningState copyWith({
    int? totalGames,
    int? aiWins,
    int? aiLosses,
    int? draws,
    double? learningRate,
    EvaluationWeights? weights,
    List<int>? recentOutcomes,
  }) {
    return AdaptiveLearningState(
      totalGames: totalGames ?? this.totalGames,
      aiWins: aiWins ?? this.aiWins,
      aiLosses: aiLosses ?? this.aiLosses,
      draws: draws ?? this.draws,
      learningRate: learningRate ?? this.learningRate,
      weights: weights ?? this.weights,
      recentOutcomes: recentOutcomes ?? this.recentOutcomes,
    );
  }
}

class LearningStatsService {
  LearningStatsService._();

  static final LearningStatsService instance = LearningStatsService._();
  static const String _stateKey = 'adaptive_learning_state_v1';

  SharedPreferences? _prefs;
  AdaptiveLearningState _state = AdaptiveLearningState.initial();
  bool _storageAvailable = true;

  AdaptiveLearningState get state => _state;
  EvaluationWeights get currentWeights => _state.weights;

  Future<void> initialize() async {
    try {
      _prefs ??= await SharedPreferences.getInstance();
    } on MissingPluginException {
      _storageAvailable = false;
      _state = AdaptiveLearningState.initial();
      return;
    } catch (_) {
      _storageAvailable = false;
      _state = AdaptiveLearningState.initial();
      return;
    }
    final raw = _prefs?.getString(_stateKey);
    if (raw == null || raw.isEmpty) {
      _state = AdaptiveLearningState.initial();
      return;
    }
    try {
      final decoded = jsonDecode(raw);
      if (decoded is Map<String, dynamic>) {
        _state = AdaptiveLearningState.fromJson(decoded);
      }
    } catch (_) {
      _state = AdaptiveLearningState.initial();
    }
  }

  Future<void> saveState() async {
    if (!_storageAvailable) {
      return;
    }
    try {
      _prefs ??= await SharedPreferences.getInstance();
      await _prefs?.setString(_stateKey, jsonEncode(_state.toJson()));
    } on MissingPluginException {
      _storageAvailable = false;
    } catch (_) {
      _storageAvailable = false;
    }
  }

  Future<void> updateAfterGame({
    required Side aiSide,
    required Side? winner,
    required bool isDraw,
  }) async {
    final outcome = isDraw ? 0 : (winner == aiSide ? 1 : -1);
    _state = evolveState(_state, outcome: outcome);
    await saveState();
  }

  static AdaptiveLearningState evolveState(
    AdaptiveLearningState prev, {
    required int outcome, // 1: win, 0: draw, -1: loss
  }) {
    final boundedOutcome = outcome.clamp(-1, 1);
    final nextOutcomes = <int>[...prev.recentOutcomes, boundedOutcome];
    if (nextOutcomes.length > 20) {
      nextOutcomes.removeRange(0, nextOutcomes.length - 20);
    }

    final total = prev.totalGames + 1;
    final winCount = prev.aiWins + (boundedOutcome == 1 ? 1 : 0);
    final lossCount = prev.aiLosses + (boundedOutcome == -1 ? 1 : 0);
    final drawCount = prev.draws + (boundedOutcome == 0 ? 1 : 0);
    final step = prev.learningRate;

    var weights = prev.weights;
    if (boundedOutcome == 1) {
      weights = weights.copyWith(
        tactical: weights.tactical + (0.40 * step),
        safety: weights.safety + (0.15 * step),
        mobility: weights.mobility + (0.10 * step),
        flip: weights.flip + (0.12 * step),
      );
    } else if (boundedOutcome == -1) {
      weights = weights.copyWith(
        safety: weights.safety + (0.50 * step),
        tactical: weights.tactical - (0.10 * step),
        localSafety: weights.localSafety + (0.22 * step),
        localFlipBias: weights.localFlipBias - (0.08 * step),
      );
    } else {
      weights = weights.copyWith(
        tactical: weights.tactical + (0.22 * step),
        mobility: weights.mobility + (0.15 * step),
        safety: weights.safety - (0.05 * step),
      );
    }

    final recentDraws = nextOutcomes.where((v) => v == 0).length;
    final drawRate = nextOutcomes.isEmpty ? 0.0 : (recentDraws / nextOutcomes.length);
    if (drawRate >= 0.55) {
      weights = weights.copyWith(
        tactical: weights.tactical + (0.25 * step),
        mobility: weights.mobility + (0.12 * step),
        localFlipBias: weights.localFlipBias - (0.10 * step),
      );
    }

    return AdaptiveLearningState(
      totalGames: total,
      aiWins: winCount,
      aiLosses: lossCount,
      draws: drawCount,
      learningRate: max(0.015, prev.learningRate * 0.999),
      weights: weights.clamped(),
      recentOutcomes: nextOutcomes,
    );
  }
}
