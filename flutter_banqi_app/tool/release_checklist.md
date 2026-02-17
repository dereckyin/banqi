# Release Hardening Checklist

1. `flutter pub get`
2. `flutter analyze`
3. `flutter test`
4. Manual smoke test:
   - Flip/move legality
   - Long-chase limit behavior
   - Draw/no-progress limit behavior
   - Minimax response latency on target device
5. Build artifacts:
   - `flutter build apk --release`
   - `flutter build ios --release`

## Latency budget

- Target: AI move under 500 ms at depth 3 on mid-tier Android.
- If over budget:
  - Lower depth default to 2
  - Reduce max branching in Minimax
  - Move AI turn to isolate for heavy settings
