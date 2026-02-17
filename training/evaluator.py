"""Model evaluation for Banqi league promotion decisions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from training.elo import EloTracker
from training.self_play import PolicySpec, SelfPlayConfig, SelfPlayRunner

LOGGER = logging.getLogger(__name__)


@dataclass
class EvaluatorConfig:
    """Evaluation match settings."""

    games: int = 200
    promotion_win_rate: float = 0.55
    base_seed: int = 10_000
    parallel_workers: int = 1


@dataclass
class EvaluationReport:
    """Result of challenger vs best evaluation."""

    challenger_wins: int
    best_wins: int
    draws: int
    win_rate: float
    promoted: bool
    challenger_elo: float
    best_elo: float


class Evaluator:
    """Evaluates checkpoints and updates Elo tracker."""

    def __init__(self, config: EvaluatorConfig, elo_tracker: EloTracker) -> None:
        self.config = config
        self.elo_tracker = elo_tracker

    def evaluate_checkpoints(
        self,
        challenger_name: str,
        challenger_path: str | Path,
        best_name: str,
        best_path: str | Path,
    ) -> EvaluationReport:
        challenger_spec = PolicySpec(
            kind="dqn_checkpoint",
            checkpoint_path=str(challenger_path),
            epsilon=0.0,
        )
        best_spec = PolicySpec(
            kind="dqn_checkpoint",
            checkpoint_path=str(best_path),
            epsilon=0.0,
        )
        runner = SelfPlayRunner(
            SelfPlayConfig(
                base_seed=self.config.base_seed,
                parallel_workers=self.config.parallel_workers,
                log_every=max(1, self.config.games // 10),
            )
        )
        trajectories = runner.run_games_parallel_from_specs(
            red_spec=challenger_spec,
            black_spec=best_spec,
            n_games=self.config.games,
            replay_buffer=None,
        )
        summary = runner.summarize(trajectories)
        challenger_wins = summary["red_wins"]
        best_wins = summary["black_wins"]
        draws = summary["draws"]
        win_rate = challenger_wins / max(1, self.config.games)
        promoted = win_rate >= self.config.promotion_win_rate

        for _ in range(challenger_wins):
            self.elo_tracker.record_match(challenger_name, best_name, 1.0)
        for _ in range(best_wins):
            self.elo_tracker.record_match(challenger_name, best_name, 0.0)
        for _ in range(draws):
            self.elo_tracker.record_match(challenger_name, best_name, 0.5)

        challenger_elo = self.elo_tracker.get(challenger_name)
        best_elo = self.elo_tracker.get(best_name)

        LOGGER.info(
            "Eval %s vs %s | W:%d L:%d D:%d win_rate=%.3f promoted=%s elo=(%.1f, %.1f)",
            challenger_name,
            best_name,
            challenger_wins,
            best_wins,
            draws,
            win_rate,
            promoted,
            challenger_elo,
            best_elo,
        )
        return EvaluationReport(
            challenger_wins=challenger_wins,
            best_wins=best_wins,
            draws=draws,
            win_rate=win_rate,
            promoted=promoted,
            challenger_elo=challenger_elo,
            best_elo=best_elo,
        )
