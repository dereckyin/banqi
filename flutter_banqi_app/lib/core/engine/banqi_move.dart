import 'types.dart';

enum MoveKind { flip, move }

class BanqiMove {
  const BanqiMove.flip(this.to) : kind = MoveKind.flip, from = null;
  const BanqiMove.move(this.from, this.to) : kind = MoveKind.move;

  final MoveKind kind;
  final Position? from;
  final Position to;

  @override
  bool operator ==(Object other) {
    return other is BanqiMove &&
        other.kind == kind &&
        other.from == from &&
        other.to == to;
  }

  @override
  int get hashCode => Object.hash(kind, from, to);

  @override
  String toString() => kind == MoveKind.flip ? 'flip:$to' : 'move:$from->$to';
}

class MoveResult {
  const MoveResult({
    this.capturedPiece,
    this.flippedPiece,
    this.winner,
    required this.isDraw,
  });

  final Piece? capturedPiece;
  final Piece? flippedPiece;
  final Side? winner;
  final bool isDraw;
}
