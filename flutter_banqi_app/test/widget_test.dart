import 'package:flutter_test/flutter_test.dart';

import 'package:flutter_banqi_app/main.dart';

void main() {
  testWidgets('app boots to Banqi screen', (WidgetTester tester) async {
    await tester.pumpWidget(const BanqiApp());
    expect(find.textContaining('狀態:'), findsOneWidget);
  });
}
