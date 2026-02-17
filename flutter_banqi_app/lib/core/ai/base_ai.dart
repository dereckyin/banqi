import '../engine/banqi_move.dart';
import '../engine/board_state.dart';

abstract class BaseAI {
  BanqiMove chooseMove(BoardState board);
}
