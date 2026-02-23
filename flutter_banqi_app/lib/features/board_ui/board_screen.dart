import 'package:flutter/material.dart';

import '../../core/engine/board_state.dart';
import '../../core/engine/types.dart';
import '../game/game_controller.dart';

class BoardScreen extends StatefulWidget {
  const BoardScreen({super.key});

  @override
  State<BoardScreen> createState() => _BoardScreenState();
}

class _BoardScreenState extends State<BoardScreen> {
  late GameController controller;
  bool _dismissEndgameOverlay = false;

  @override
  void initState() {
    super.initState();
    controller = GameController(
      depth: 2,
      humanSide: Side.red,
      longChaseLimit: 7,
      drawNoProgressLimit: 50,
    );
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: controller,
      builder: (context, _) {
        final terminal = controller.board.gameOver();
        if (!terminal.$1 && _dismissEndgameOverlay) {
          WidgetsBinding.instance.addPostFrameCallback((_) {
            if (!mounted) {
              return;
            }
            setState(() {
              _dismissEndgameOverlay = false;
            });
          });
        }
        return Scaffold(
          body: Stack(
            children: [
              Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      Color(0xFF12212E),
                      Color(0xFF1F2F3E),
                      Color(0xFF2A3C4F),
                    ],
                  ),
                ),
                child: SafeArea(
                  child: LayoutBuilder(
                    builder: (context, constraints) {
                      final sideWidth = (constraints.maxWidth * 0.20).clamp(150.0, 188.0);
                      return Padding(
                        padding: const EdgeInsets.all(10),
                        child: Row(
                          children: [
                            Expanded(child: _boardPanel()),
                            const SizedBox(width: 8),
                            SizedBox(width: sideWidth, child: _sidePanel(terminal)),
                          ],
                        ),
                      );
                    },
                  ),
                ),
              ),
              if (terminal.$1 && !controller.spectatorMode && !_dismissEndgameOverlay)
                _endgameOverlay(terminal),
            ],
          ),
        );
      },
    );
  }

  Widget _boardPanel() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF9A6A3A), Color(0xFF7A4F27)],
        ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFE0C28E), width: 1.2),
        boxShadow: const [
          BoxShadow(
            color: Color(0x44000000),
            blurRadius: 10,
            offset: Offset(0, 5),
          ),
        ],
      ),
      child: _boardGrid(),
    );
  }

  Widget _sidePanel((bool, Side?, bool) terminal) {
    final winnerText = terminal.$1
        ? (terminal.$3 ? '和局' : '勝方: ${terminal.$2 == Side.red ? '紅方' : '黑方'}')
        : '對局進行中';
    return Column(
      children: [
        Row(
          children: [
            Expanded(
              child: FilledButton.icon(
                onPressed: controller.restart,
                icon: const Icon(Icons.refresh),
                label: const Text('重新開始'),
              ),
            ),
            const SizedBox(width: 4),
            Opacity(
              opacity: 0.72,
              child: IconButton(
                tooltip: '悔一步',
                onPressed: controller.canUndo &&
                        !controller.thinking &&
                        !controller.spectatorMode
                    ? controller.undoOneStep
                    : null,
                icon: const Icon(Icons.undo, size: 18),
                visualDensity: VisualDensity.compact,
                color: const Color(0xFFDDD8C9),
                disabledColor: const Color(0x66DDD8C9),
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: FilledButton.tonalIcon(
                onPressed: controller.spectatorMode
                    ? controller.stopSpectatorMode
                    : () async {
                        if (controller.thinking) {
                          return;
                        }
                        await controller.watchOneAiVsAiGame();
                      },
                icon: Icon(controller.spectatorMode ? Icons.stop : Icons.visibility),
                label: Text(controller.spectatorMode ? '停止對戰' : 'AI 對戰'),
              ),
            ),
          ],
        ),
        const SizedBox(height: 10),
        Expanded(
          child: SingleChildScrollView(
            child: _statusCard(terminal, winnerText),
          ),
        ),
      ],
    );
  }

  Widget _statusCard((bool, Side?, bool) terminal, String winnerText) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0x88FFFFFF),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'AI 暗棋 v1.0.0',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
          ),
          const Text(
            '手搓AI坊',
            style: TextStyle(fontSize: 12.5, color: Color(0xCC2C3E50)),
          ),
          const SizedBox(height: 6),
          Text(
            '紅方被吃: ${controller.redCapturedSummary}',
            style: const TextStyle(fontSize: 12.5, color: Color(0xFFB01212)),
          ),
          Text(
            '黑方被吃: ${controller.blackCapturedSummary}',
            style: const TextStyle(fontSize: 12.5, color: Color(0xFF1C326E)),
          ),
          const SizedBox(height: 6),
          Text(
            '狀態: ${controller.status}',
            style: const TextStyle(fontSize: 13),
          ),
          Text(
            '你方: ${controller.humanSide == Side.red ? '紅' : '黑'}',
            style: TextStyle(
              fontSize: 13,
              color: controller.humanSide == Side.red
                  ? const Color(0xFFB01212)
                  : const Color(0xFF1C326E),
              fontWeight: FontWeight.w600,
            ),
          ),
          Text(
            '進度: ${controller.board.noProgressPlies}/${controller.board.drawNoProgressLimit}',
            style: const TextStyle(fontSize: 13),
          ),
          Text(
            '長追: ${controller.longChaseLimit}',
            style: const TextStyle(fontSize: 13),
          ),
          const Divider(height: 14),
          Text(
            winnerText,
            style: TextStyle(
              fontWeight: FontWeight.w700,
              fontSize: 13,
              color: terminal.$1 ? Colors.deepOrange.shade700 : Colors.black87,
            ),
          ),
          const Divider(height: 14),
          FilledButton.tonalIcon(
            onPressed: controller.canApplyStrategyAdjustment
                ? () async {
                    await controller.applyStrategyAdjustmentForLastGame();
                  }
                : null,
            icon: const Icon(Icons.tune),
            label: const Text('策略調整'),
          ),
        ],
      ),
    );
  }

  Widget _boardGrid() {
    return AspectRatio(
      aspectRatio: 2.0,
      child: GridView.builder(
        physics: const NeverScrollableScrollPhysics(),
        itemCount: BoardState.rows * BoardState.cols,
        gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: BoardState.cols,
          childAspectRatio: 1.0,
        ),
        itemBuilder: (context, index) {
          final row = index ~/ BoardState.cols;
          final col = index % BoardState.cols;
          final pos = Position(row, col);
          final cell = controller.board.cell(pos);
          final selected = controller.selected == pos;
          final lastMove = controller.lastMove;
          final isLanding = lastMove != null && lastMove.to == pos;
          final face = _buildCellFace(cell);

          return Padding(
            padding: const EdgeInsets.all(3),
            child: InkWell(
              borderRadius: BorderRadius.circular(10),
              onTap: () => controller.tap(pos),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 140),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    colors: selected
                        ? const [Color(0xFFFFEEB8), Color(0xFFF2C85B)]
                        : const [Color(0xFFDFC190), Color(0xFFCDA066)],
                  ),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(
                    color: selected
                        ? const Color(0xFFE09A00)
                        : const Color(0xFF7E5930),
                    width: selected ? 2.4 : 1,
                  ),
                  boxShadow: [
                    const BoxShadow(
                      color: Color(0x2A000000),
                      blurRadius: 3,
                      offset: Offset(0, 1),
                    ),
                    if (selected)
                      const BoxShadow(
                        color: Color(0x66FFD45E),
                        blurRadius: 8,
                        offset: Offset(0, 0),
                      ),
                  ],
                ),
                child: Center(
                  child: _landingShake(
                    child: face,
                    enabled: isLanding,
                    tick: controller.moveAnimTick,
                    keyId: 'r${pos.row}c${pos.col}',
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildCellFace(CellState cell) {
    if (cell.isHidden) {
      return Container(
        width: 42,
        height: 42,
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFFB48A55), Color(0xFF8A6538)],
          ),
          shape: BoxShape.circle,
          border: Border.all(color: const Color(0xFF6D4A26), width: 1.5),
          boxShadow: const [
            BoxShadow(
              color: Color(0x33000000),
              blurRadius: 5,
              offset: Offset(0, 2),
            ),
          ],
        ),
        child: Stack(
          alignment: Alignment.center,
          children: [
            Container(
              width: 30,
              height: 30,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(color: const Color(0x8057351A)),
              ),
            ),
            Container(
              width: 12,
              height: 12,
              decoration: const BoxDecoration(
                shape: BoxShape.circle,
                color: Color(0xAA5C3B1E),
              ),
            ),
            const Align(
              alignment: Alignment(-0.35, -0.35),
              child: SizedBox(
                width: 10,
                height: 10,
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Color(0x35FFFFFF),
                  ),
                ),
              ),
            ),
          ],
        ),
      );
    }

    if (cell.piece == null) {
      return Container(
        width: 10,
        height: 10,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: const Color(0x70553418),
          boxShadow: const [
            BoxShadow(color: Color(0x22000000), blurRadius: 2, offset: Offset(0, 1)),
          ],
        ),
      );
    }

    final textColor = cell.piece!.side == Side.red
        ? const Color(0xFFB01212)
        : const Color(0xFF1C326E);
    return Container(
      width: 44,
      height: 44,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: const LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [Color(0xFFF6E8CC), Color(0xFFE5CFAB)],
        ),
        border: Border.all(color: const Color(0xFF8A6334), width: 1.4),
        boxShadow: const [
          BoxShadow(
            color: Color(0x35000000),
            blurRadius: 5,
            offset: Offset(0, 2),
          ),
        ],
      ),
      child: Stack(
        children: [
          const Align(
            alignment: Alignment(-0.3, -0.42),
            child: SizedBox(
              width: 12,
              height: 12,
              child: DecoratedBox(
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Color(0x35FFFFFF),
                ),
              ),
            ),
          ),
          Center(
            child: Transform.translate(
              offset: const Offset(0, -2),
              child: Text(
                cell.piece!.glyph,
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.w800,
                  color: textColor,
                  shadows: const [
                    Shadow(
                      color: Color(0x33000000),
                      blurRadius: 1.2,
                      offset: Offset(0, 1),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _landingShake({
    required Widget child,
    required bool enabled,
    required int tick,
    required String keyId,
  }) {
    if (!enabled) {
      return child;
    }
    return TweenAnimationBuilder<double>(
      key: ValueKey('$keyId-$tick'),
      tween: Tween<double>(begin: 0, end: 1),
      duration: const Duration(milliseconds: 180),
      builder: (context, t, c) {
        final envelope = (1 - t);
        final dx = (t < 0.25)
            ? -2.0 * envelope
            : (t < 0.5)
            ? 2.0 * envelope
            : (t < 0.75)
            ? -1.2 * envelope
            : 1.2 * envelope;
        return Transform.translate(offset: Offset(dx, 0), child: c);
      },
      child: child,
    );
  }

  Widget _endgameOverlay((bool, Side?, bool) terminal) {
    final text = terminal.$3 ? '和局' : (terminal.$2 == Side.red ? '紅方勝利' : '黑方勝利');
    return Positioned.fill(
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: () {
          setState(() {
            _dismissEndgameOverlay = true;
          });
        },
        child: AnimatedOpacity(
          opacity: 1,
          duration: const Duration(milliseconds: 220),
          child: Container(
            color: const Color(0x99000000),
            alignment: Alignment.center,
            child: GestureDetector(
              behavior: HitTestBehavior.opaque,
              onTap: () {},
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 18),
                decoration: BoxDecoration(
                  color: const Color(0xEDEBDCC7),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: const Color(0xFF8A6334), width: 2),
                  boxShadow: const [
                    BoxShadow(
                      color: Color(0x66000000),
                      blurRadius: 16,
                      offset: Offset(0, 6),
                    ),
                  ],
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text(
                      '對局結束',
                      style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      text,
                      style: TextStyle(
                        fontSize: 28,
                        fontWeight: FontWeight.w900,
                        color: terminal.$3
                            ? const Color(0xFF6A4A1D)
                            : (terminal.$2 == Side.red
                                  ? const Color(0xFFB01212)
                                  : const Color(0xFF1C326E)),
                      ),
                    ),
                    const SizedBox(height: 14),
                    Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        FilledButton.icon(
                          onPressed: controller.restart,
                          icon: const Icon(Icons.refresh),
                          label: const Text('再來一局'),
                        ),
                        const SizedBox(width: 10),
                        FilledButton.tonalIcon(
                          onPressed: () async {
                            await controller.watchOneAiVsAiGame();
                          },
                          icon: const Icon(Icons.visibility),
                          label: const Text('AI 對戰'),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
