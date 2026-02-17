import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'features/board_ui/board_screen.dart';
import 'features/stats/learning_stats_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await LearningStatsService.instance.initialize();
  await SystemChrome.setPreferredOrientations(const [
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);
  runApp(const BanqiApp());
}

class BanqiApp extends StatelessWidget {
  const BanqiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI 暗棋',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blueGrey),
        useMaterial3: true,
      ),
      home: const BoardScreen(),
    );
  }
}
