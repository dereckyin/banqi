"""Self-play runner for scalable Banqi data generation."""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ai.base_ai import BaseAI
from ai.dqn_ai import DQNAI, EpsilonSchedule
from ai.minimax_ai import MinimaxAI
from engine.board import Board
from engine.pieces import Side
from training.replay_buffer import ReplayBuffer, Transition

LOGGER = logging.getLogger(__name__)


@dataclass
class EpisodeStats:
    """Summary from one self-play episode."""

    winner: Optional[Side]
    is_draw: bool
    plies: int


@dataclass
class RewardConfig:
    """Reward design for transition collection."""

    win_reward: float = 1.0
    loss_reward: float = -1.0
    draw_reward: float = 0.0
    capture_bonus: float = 0.01
    lose_piece_penalty: float = 0.01
    step_penalty: float = 0.0


@dataclass
class PolicySpec:
    """Serializable policy descriptor for workers."""

    kind: str  # dqn_checkpoint or minimax
    checkpoint_path: Optional[str] = None
    depth: int = 2
    epsilon: float = 0.05


@dataclass
class SelfPlayConfig:
    """Self-play generation config."""

    max_plies: int = 500
    no_progress_limit: int = 50
    parallel_workers: int = 1
    base_seed: Optional[int] = None
    log_every: int = 10


@dataclass
class GameTrajectory:
    """Trajectory for one full game."""

    transitions: List[Transition]
    stats: EpisodeStats


def _build_agent_from_spec(spec: PolicySpec, seed: Optional[int]) -> BaseAI:
    # Future extension point: add "mcts_checkpoint" or "hybrid_mcts_dqn" kinds here.
    # The runner API stays stable so training/evaluation orchestration does not change.
    if spec.kind == "minimax":
        return MinimaxAI(depth=spec.depth, seed=seed)
    if spec.kind == "dqn_checkpoint":
        agent = DQNAI(
            seed=seed,
            epsilon_schedule=EpsilonSchedule(start=spec.epsilon, end=spec.epsilon, decay=1.0),
        )
        if spec.checkpoint_path:
            agent.load(spec.checkpoint_path)
        agent.epsilon = spec.epsilon
        return agent
    raise ValueError(f"Unsupported PolicySpec kind: {spec.kind}")


def _simulate_single_game(
    red_ai: BaseAI,
    black_ai: BaseAI,
    reward_config: RewardConfig,
    seed: Optional[int],
    max_plies: int,
    no_progress_limit: int,
) -> GameTrajectory:
    board = Board(seed=seed, draw_no_progress_limit=no_progress_limit)
    transitions: List[Transition] = []
    done = False
    winner: Optional[Side] = None
    is_draw = False

    while not done and board.ply_count < max_plies:
        actor = red_ai if board.current_turn is Side.RED else black_ai
        actor_side = board.current_turn

        state = board.encode_state()
        legal_mask = board.legal_action_mask()
        move = actor.choose_move(board)
        action = board.move_to_action(move)

        # Track piece-count deltas for optional shaped penalties.
        own_before = board.piece_count_remaining(actor_side)
        opp_before = board.piece_count_remaining(actor_side.opponent())

        board.apply_move(move)
        next_state = board.encode_state()
        next_legal_mask = board.legal_action_mask()
        done, winner, is_draw = board.game_over()

        own_after = board.piece_count_remaining(actor_side)
        opp_after = board.piece_count_remaining(actor_side.opponent())

        reward = -reward_config.step_penalty
        if opp_after < opp_before:
            reward += reward_config.capture_bonus
        if own_after < own_before:
            reward -= reward_config.lose_piece_penalty

        if done:
            if is_draw:
                reward += reward_config.draw_reward
            elif winner is actor_side:
                reward += reward_config.win_reward
            else:
                reward += reward_config.loss_reward

        transitions.append(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                legal_mask=legal_mask,
                next_legal_mask=next_legal_mask,
            )
        )

    if board.ply_count >= max_plies and not done:
        is_draw = True
        winner = None

    stats = EpisodeStats(winner=winner, is_draw=is_draw, plies=board.ply_count)
    return GameTrajectory(transitions=transitions, stats=stats)


def _parallel_worker(
    game_index: int,
    red_spec: PolicySpec,
    black_spec: PolicySpec,
    reward_config: RewardConfig,
    max_plies: int,
    no_progress_limit: int,
    base_seed: Optional[int],
) -> GameTrajectory:
    seed = None if base_seed is None else base_seed + game_index
    red_ai = _build_agent_from_spec(red_spec, seed=seed)
    black_ai = _build_agent_from_spec(black_spec, seed=None if seed is None else seed + 9973)
    return _simulate_single_game(
        red_ai,
        black_ai,
        reward_config=reward_config,
        seed=seed,
        max_plies=max_plies,
        no_progress_limit=no_progress_limit,
    )


class SelfPlayRunner:
    """Runs AI-vs-AI matches and returns trajectories."""

    def __init__(self, config: SelfPlayConfig, reward_config: RewardConfig | None = None) -> None:
        self.config = config
        self.reward_config = reward_config or RewardConfig()

    def run_games(
        self,
        red_ai: BaseAI,
        black_ai: BaseAI,
        n_games: int,
        replay_buffer: Optional[ReplayBuffer] = None,
    ) -> List[GameTrajectory]:
        trajectories: List[GameTrajectory] = []
        for game_index in range(n_games):
            seed = None if self.config.base_seed is None else self.config.base_seed + game_index
            trajectory = _simulate_single_game(
                red_ai,
                black_ai,
                reward_config=self.reward_config,
                seed=seed,
                max_plies=self.config.max_plies,
                no_progress_limit=self.config.no_progress_limit,
            )
            trajectories.append(trajectory)
            self._record_trajectory(trajectory, replay_buffer)
            if (game_index + 1) % max(1, self.config.log_every) == 0:
                LOGGER.info(
                    "Self-play serial game %d/%d | winner=%s draw=%s plies=%d",
                    game_index + 1,
                    n_games,
                    trajectory.stats.winner,
                    trajectory.stats.is_draw,
                    trajectory.stats.plies,
                )
        return trajectories

    def run_games_parallel_from_specs(
        self,
        red_spec: PolicySpec,
        black_spec: PolicySpec,
        n_games: int,
        replay_buffer: Optional[ReplayBuffer] = None,
    ) -> List[GameTrajectory]:
        if self.config.parallel_workers <= 1:
            red_ai = _build_agent_from_spec(red_spec, seed=self.config.base_seed)
            black_ai = _build_agent_from_spec(black_spec, seed=None if self.config.base_seed is None else self.config.base_seed + 1)
            return self.run_games(red_ai, black_ai, n_games=n_games, replay_buffer=replay_buffer)

        # Deep-copy config objects once so workers get independent immutable payloads.
        reward_cfg = copy.deepcopy(self.reward_config)
        args = [
            (
                idx,
                red_spec,
                black_spec,
                reward_cfg,
                self.config.max_plies,
                self.config.no_progress_limit,
                self.config.base_seed,
            )
            for idx in range(n_games)
        ]
        with mp.Pool(processes=self.config.parallel_workers) as pool:
            trajectories = pool.starmap(_parallel_worker, args)

        for idx, trajectory in enumerate(trajectories):
            self._record_trajectory(trajectory, replay_buffer)
            if (idx + 1) % max(1, self.config.log_every) == 0:
                LOGGER.info(
                    "Self-play parallel game %d/%d | winner=%s draw=%s plies=%d",
                    idx + 1,
                    n_games,
                    trajectory.stats.winner,
                    trajectory.stats.is_draw,
                    trajectory.stats.plies,
                )
        return trajectories

    @staticmethod
    def summarize(trajectories: Sequence[GameTrajectory]) -> Dict[str, int]:
        summary = {"red_wins": 0, "black_wins": 0, "draws": 0}
        for trajectory in trajectories:
            if trajectory.stats.is_draw:
                summary["draws"] += 1
            elif trajectory.stats.winner is Side.RED:
                summary["red_wins"] += 1
            elif trajectory.stats.winner is Side.BLACK:
                summary["black_wins"] += 1
        return summary

    def _record_trajectory(self, trajectory: GameTrajectory, replay_buffer: Optional[ReplayBuffer]) -> None:
        if replay_buffer is None:
            return
        for transition in trajectory.transitions:
            replay_buffer.push(transition)


# Backward-compatible helper wrappers.
def play_episode(
    red_ai: BaseAI,
    black_ai: BaseAI,
    replay_buffer: Optional[ReplayBuffer] = None,
    seed: Optional[int] = None,
    max_plies: int = 500,
) -> EpisodeStats:
    runner = SelfPlayRunner(SelfPlayConfig(max_plies=max_plies, no_progress_limit=50, base_seed=seed))
    trajectories = runner.run_games(red_ai, black_ai, n_games=1, replay_buffer=replay_buffer)
    return trajectories[0].stats


def run_self_play(
    red_ai: BaseAI,
    black_ai: BaseAI,
    episodes: int,
    replay_buffer: Optional[ReplayBuffer] = None,
    base_seed: Optional[int] = None,
) -> Dict[str, int]:
    """Run multiple episodes and return aggregate results."""
    runner = SelfPlayRunner(SelfPlayConfig(base_seed=base_seed))
    trajectories = runner.run_games(red_ai, black_ai, n_games=episodes, replay_buffer=replay_buffer)
    return runner.summarize(trajectories)
