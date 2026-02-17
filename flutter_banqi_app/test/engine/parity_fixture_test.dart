import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

import 'package:flutter_banqi_app/core/engine/banqi_move.dart';
import 'package:flutter_banqi_app/core/engine/board_state.dart';
import 'package:flutter_banqi_app/core/engine/types.dart';

void main() {
  test('initial fixture parity baseline', () {
    final fixturePath = File('test/fixtures/initial_position_fixture.json');
    final fixture =
        json.decode(fixturePath.readAsStringSync()) as Map<String, dynamic>;

    final board = BoardState.initial(seed: 123);
    final legal = board.legalMoves();
    final flipCount = legal.where((m) => m.kind == MoveKind.flip).length;
    final sideName = board.currentTurn == Side.red ? 'red' : 'black';

    expect(BoardState.rows, fixture['rows']);
    expect(BoardState.cols, fixture['cols']);
    expect(legal.length, fixture['expected_legal_moves']);
    expect(flipCount, fixture['expected_flip_moves']);
    expect(sideName, fixture['expected_current_turn']);
  });
}
