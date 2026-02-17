import 'package:flutter_test/flutter_test.dart';

import 'package:flutter_banqi_app/core/ai/minimax_ai.dart';
import 'package:flutter_banqi_app/features/stats/learning_stats_service.dart';

void main() {
  test('evaluation weights json roundtrip and clamp', () {
    const weights = EvaluationWeights(
      mobility: 9.0,
      tactical: -2.0,
      safety: 1.5,
      flip: 1.1,
      localSafety: 0.9,
      localFlipBias: 4.0,
    );
    final encoded = weights.toJson();
    final decoded = EvaluationWeights.fromJson(encoded);
    expect(decoded.mobility, inInclusiveRange(0.6, 2.4));
    expect(decoded.tactical, inInclusiveRange(0.8, 3.5));
    expect(decoded.safety, closeTo(1.5, 1e-6));
    expect(decoded.localFlipBias, inInclusiveRange(0.5, 2.0));
  });

  test('adaptive evolve updates counters and learning state', () {
    final initial = AdaptiveLearningState.initial();
    final winState = LearningStatsService.evolveState(initial, outcome: 1);
    expect(winState.totalGames, 1);
    expect(winState.aiWins, 1);
    expect(winState.aiLosses, 0);
    expect(winState.draws, 0);
    expect(winState.recentOutcomes, [1]);
    expect(winState.weights.tactical, greaterThan(initial.weights.tactical));

    final lossState = LearningStatsService.evolveState(winState, outcome: -1);
    expect(lossState.totalGames, 2);
    expect(lossState.aiWins, 1);
    expect(lossState.aiLosses, 1);
    expect(lossState.recentOutcomes, [1, -1]);
    expect(lossState.weights.safety, greaterThan(winState.weights.safety));
  });

  test('draw-heavy sequence nudges aggression parameters', () {
    var state = AdaptiveLearningState.initial();
    for (var i = 0; i < 8; i++) {
      state = LearningStatsService.evolveState(state, outcome: 0);
    }
    expect(state.draws, 8);
    expect(state.weights.tactical, greaterThan(2.0));
    expect(state.weights.mobility, greaterThan(1.3));
  });
}
