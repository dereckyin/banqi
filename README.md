# Banqi AI (Taiwanese Dark Chess)

Modular Python project for a desktop CLI Banqi engine with both rule-based and reinforcement-learning AI foundations.

## Features in this milestone

- Full 4x8 Banqi engine with hidden state and legal move generation
- Taiwan-standard capture rules (`General` vs `Soldier` special case)
- Cannon jump-capture rule over exactly one intervening piece
- 50-ply no-progress draw rule (no capture and no flip)
- Minimax AI with alpha-beta pruning
- DQN skeleton with PyTorch policy/target networks and save/load
- Professional self-play pipeline with replay buffer, evaluator, Elo, and league promotion
- CLI gameplay versus Minimax AI

## Project structure

```text
engine/
  board.py
  pieces.py
  rules.py
ai/
  base_ai.py
  minimax_ai.py
  dqn_ai.py
  mcts_ai.py
training/
  self_play.py
  replay_buffer.py
  trainer.py
  evaluator.py
  elo.py
  league_manager.py
configs/
  training_config.json
models/
cli/
  main.py
  gui.py
  watch_selfplay.py
  train.py
```

## Setup

1. Install Python 3.10+.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Play in CLI

```bash
python -m cli.main --depth 3 --human-side red --seed 42
```

Commands:
- `flip <row> <col>`
- `move <r1> <c1> <r2> <c2>`
- `help`
- `quit`

## Play with Desktop GUI

```bash
python -m cli.gui --depth 3 --human-side red --seed 42
```

GUI controls:
- Click hidden cell to flip
- Click your revealed piece, then click destination to move/capture
- `New Game` button restarts current session

## Watch AI vs AI (Desktop)

```bash
python -m cli.watch_selfplay --red-ai minimax --black-ai minimax --red-depth 2 --black-depth 2 --games 5 --delay-ms 350 --seed 42
```

Options:
- `--delay-ms`: milliseconds per move (smaller is faster)
- `--games`: number of consecutive games to watch
- `--red-ai/--black-ai`: `minimax` or `dqn`
- `--red-checkpoint/--black-checkpoint`: DQN checkpoint path when using `dqn`

## Start self-play training

```bash
python -m cli.train --config configs/training_config.json --log-level INFO
```

Pipeline per iteration:
1. Generate AI-vs-AI self-play games
2. Store transitions in replay buffer
3. Train policy for K mini-batch steps
4. Evaluate candidate vs current best (Elo + win rate)
5. Promote to `models/model_best.pt` when threshold is met
6. Keep last N versions in league for opponent diversity

TensorBoard (optional):
```bash
tensorboard --logdir runs/banqi
```

## Programmatic training example

```python
from ai.dqn_ai import DQNAI
from training.trainer import DQNTrainer, TrainerConfig

agent = DQNAI(seed=42)
config = TrainerConfig.from_json("configs/training_config.json")
trainer = DQNTrainer(agent=agent, config=config)
trainer.run_training_loop()
```

## Deterministic behavior

Use `seed` arguments in `Board`, `MinimaxAI`, and `DQNAI` constructors for reproducible runs.

## Privacy Policy

- Traditional Chinese privacy policy page:
  - `docs/privacy-policy.html`
- After enabling GitHub Pages for this repository, the public URL will be:
  - `https://dereckyin.github.io/banqi/privacy-policy.html`
