# Banqi Offline Flutter

Offline Flutter mobile app for Taiwanese Dark Chess (Banqi), with:

- Dart rule engine (`lib/core/engine/`)
- On-device Minimax AI (`lib/core/ai/`)
- Mobile board UI (`lib/features/board_ui/`)
- Game controller flow (`lib/features/game/`)

## Run

```bash
flutter pub get
flutter run
```

## Test

```bash
flutter test
```

Included tests:

- Engine legality baseline
- Rule regression (horse/chariot capture ordering)
- Initial-state parity fixture
- Minimax tactical capture preference

## Performance notes

- First release is optimized for offline reliability.
- Search guard uses max branching to avoid deep-frame stalls on mid-tier devices.
- If AI feels slow, lower depth in app settings.

## Release checklist

```bash
flutter analyze
flutter test
flutter build apk --release
flutter build ios --release
```

## Roadmap

- Add parity fixtures exported from Python reference engine
- Add stronger iterative deepening and transposition cache
- Add optional on-device DQN inference (training remains in Python toolchain)
