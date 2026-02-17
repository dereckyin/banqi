"""CLI command to launch Banqi self-play training."""

from __future__ import annotations

import argparse
import logging

from ai.dqn_ai import DQNAI, EpsilonSchedule
from training.trainer import DQNTrainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Banqi DQN with self-play league pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.json",
        help="Path to training config JSON",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    config = TrainerConfig.from_json(args.config)
    agent = DQNAI(
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon_schedule=EpsilonSchedule(
            start=config.epsilon_start,
            end=config.epsilon_end,
            decay=config.epsilon_decay,
        ),
        seed=None if config.seed is None else int(config.seed),
    )

    trainer = DQNTrainer(agent=agent, config=config)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
