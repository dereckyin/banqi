import 'dart:math';

enum Side { red, black }

extension SideX on Side {
  Side get opponent => this == Side.red ? Side.black : Side.red;
}

enum Rank { general, advisor, elephant, chariot, horse, cannon, soldier }

const Map<Rank, int> rankStrength = {
  Rank.general: 7,
  Rank.advisor: 6,
  Rank.elephant: 5,
  Rank.chariot: 4,
  Rank.horse: 3,
  Rank.cannon: 2,
  Rank.soldier: 1,
};

const Map<Rank, int> rankValue = {
  Rank.general: 90,
  Rank.advisor: 20,
  Rank.elephant: 20,
  Rank.chariot: 50,
  Rank.horse: 40,
  Rank.cannon: 45,
  Rank.soldier: 10,
};

const Map<Rank, int> rankCounts = {
  Rank.general: 1,
  Rank.advisor: 2,
  Rank.elephant: 2,
  Rank.horse: 2,
  Rank.chariot: 2,
  Rank.cannon: 2,
  Rank.soldier: 5,
};

const Map<Rank, String> blackGlyph = {
  Rank.general: '將',
  Rank.advisor: '士',
  Rank.elephant: '象',
  Rank.chariot: '車',
  Rank.horse: '馬',
  Rank.cannon: '包',
  Rank.soldier: '卒',
};

const Map<Rank, String> redGlyph = {
  Rank.general: '帥',
  Rank.advisor: '士',
  Rank.elephant: '相',
  Rank.chariot: '俥',
  Rank.horse: '傌',
  Rank.cannon: '砲',
  Rank.soldier: '兵',
};

class Piece {
  const Piece({required this.side, required this.rank});

  final Side side;
  final Rank rank;

  int get value => rankValue[rank] ?? 0;

  String get glyph =>
      side == Side.red ? (redGlyph[rank] ?? '?') : (blackGlyph[rank] ?? '?');

  @override
  String toString() => '${side.name}-${rank.name}';
}

class Position {
  const Position(this.row, this.col);

  final int row;
  final int col;

  @override
  bool operator ==(Object other) =>
      other is Position && other.row == row && other.col == col;

  @override
  int get hashCode => Object.hash(row, col);

  @override
  String toString() => '($row,$col)';
}

class CellState {
  const CellState._({required this.hidden, required this.piece});

  const CellState.hidden() : this._(hidden: true, piece: null);
  const CellState.empty() : this._(hidden: false, piece: null);
  const CellState.revealed(Piece piece) : this._(hidden: false, piece: piece);

  final bool hidden;
  final Piece? piece;

  bool get isEmpty => !hidden && piece == null;
  bool get isHidden => hidden;
  bool get isRevealed => !hidden && piece != null;

  CellState copy() {
    if (isHidden) {
      return const CellState.hidden();
    }
    if (piece == null) {
      return const CellState.empty();
    }
    return CellState.revealed(Piece(side: piece!.side, rank: piece!.rank));
  }
}

List<Piece> buildFullPiecePool(Random rng) {
  final pool = <Piece>[];
  for (final side in Side.values) {
    for (final rank in Rank.values) {
      final count = rankCounts[rank] ?? 0;
      for (var i = 0; i < count; i++) {
        pool.add(Piece(side: side, rank: rank));
      }
    }
  }
  pool.shuffle(rng);
  return pool;
}
