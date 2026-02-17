"""Elo rating helpers for AI evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class EloConfig:
    """Elo configuration."""

    k_factor: float = 24.0
    initial_rating: float = 1200.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k_factor: float = 24.0,
) -> Tuple[float, float]:
    """
    Update Elo ratings.

    score_a: 1.0 win, 0.5 draw, 0.0 loss for player A.
    """
    exp_a = expected_score(rating_a, rating_b)
    exp_b = 1.0 - exp_a
    score_b = 1.0 - score_a

    new_a = rating_a + k_factor * (score_a - exp_a)
    new_b = rating_b + k_factor * (score_b - exp_b)
    return new_a, new_b


class EloTracker:
    """Track ratings for multiple model versions."""

    def __init__(self, config: EloConfig | None = None) -> None:
        self.config = config or EloConfig()
        self._ratings: Dict[str, float] = {}

    def get(self, model_name: str) -> float:
        if model_name not in self._ratings:
            self._ratings[model_name] = self.config.initial_rating
        return self._ratings[model_name]

    def record_match(self, model_a: str, model_b: str, score_a: float) -> None:
        rating_a = self.get(model_a)
        rating_b = self.get(model_b)
        new_a, new_b = update_elo(
            rating_a=rating_a,
            rating_b=rating_b,
            score_a=score_a,
            k_factor=self.config.k_factor,
        )
        self._ratings[model_a] = new_a
        self._ratings[model_b] = new_b

    def leaderboard(self) -> Dict[str, float]:
        return dict(sorted(self._ratings.items(), key=lambda item: item[1], reverse=True))
