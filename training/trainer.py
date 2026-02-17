"""Professional self-play DQN training pipeline for Banqi."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

from ai.dqn_ai import DQNAI
from training.evaluator import Evaluator, EvaluatorConfig
from training.league_manager import LeagueManager
from training.replay_buffer import ReplayBuffer
from training.self_play import PolicySpec, RewardConfig, SelfPlayConfig, SelfPlayRunner

LOGGER = logging.getLogger(__name__)


class TrainerConfig:
    """Container for trainer hyperparameters loaded from config file."""

    def __init__(self, payload: Dict[str, object]) -> None:
        self.iterations = int(payload.get("iterations", 200))
        self.games_per_iteration = int(payload.get("games_per_iteration", 32))
        self.training_steps_per_iteration = int(payload.get("training_steps_per_iteration", 64))
        self.batch_size = int(payload.get("batch_size", 128))
        self.target_sync_interval = int(payload.get("target_sync_interval", 200))
        self.gradient_clip_norm = float(payload.get("gradient_clip_norm", 5.0))
        self.checkpoint_every_steps = int(payload.get("checkpoint_every_steps", 500))
        self.save_every_iteration = bool(payload.get("save_every_iteration", True))
        self.learning_rate_gamma = float(payload.get("learning_rate_gamma", 0.9995))
        self.seed = payload.get("seed")
        self.deterministic = bool(payload.get("deterministic", False))
        self.tensorboard = bool(payload.get("tensorboard", True))

        dqn = payload.get("dqn", {})
        self.learning_rate = float(dqn.get("learning_rate", 1e-3))
        self.gamma = float(dqn.get("gamma", 0.99))
        self.epsilon_start = float(dqn.get("epsilon_start", 1.0))
        self.epsilon_end = float(dqn.get("epsilon_end", 0.05))
        self.epsilon_decay = float(dqn.get("epsilon_decay", 0.9995))

        replay = payload.get("replay_buffer", {})
        self.replay_capacity = int(replay.get("capacity", 100_000))
        self.replay_prioritized = bool(replay.get("prioritized", False))
        self.replay_alpha = float(replay.get("alpha", 0.6))
        self.replay_beta_start = float(replay.get("beta_start", 0.4))
        self.replay_beta_increment = float(replay.get("beta_increment", 1e-4))

        selfplay = payload.get("self_play", {})
        self.max_plies = int(selfplay.get("max_plies", 500))
        self.no_progress_limit = int(selfplay.get("no_progress_limit", 50))
        self.parallel_workers = int(selfplay.get("parallel_workers", 1))
        self.win_reward = float(selfplay.get("win_reward", 1.0))
        self.loss_reward = float(selfplay.get("loss_reward", -1.0))
        self.capture_bonus = float(selfplay.get("capture_bonus", 0.01))
        self.lose_piece_penalty = float(selfplay.get("lose_piece_penalty", 0.01))
        self.draw_penalty = float(selfplay.get("draw_penalty", 0.1))
        self.step_penalty = float(selfplay.get("step_penalty", 0.001))
        self.base_seed = selfplay.get("base_seed")
        self.log_every_games = int(selfplay.get("log_every_games", 16))

        evaluation = payload.get("evaluation", {})
        self.eval_games = int(evaluation.get("games", 200))
        self.eval_win_rate_threshold = float(evaluation.get("promotion_win_rate", 0.55))
        self.eval_parallel_workers = int(evaluation.get("parallel_workers", 1))

        league = payload.get("league", {})
        self.max_versions = int(league.get("max_versions", 8))
        self.sample_best_probability = float(league.get("sample_best_probability", 0.5))

        output = payload.get("output", {})
        self.model_dir = str(output.get("model_dir", "models"))
        self.log_dir = str(output.get("log_dir", "runs/banqi"))

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainerConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload)


class DQNTrainer:
    """Coordinates self-play, optimization, evaluation, and model promotion."""

    def __init__(
        self,
        agent: DQNAI,
        config: TrainerConfig,
    ) -> None:
        self.agent = agent
        self.config = config
        self.replay_buffer = ReplayBuffer(
            capacity=config.replay_capacity,
            prioritized=config.replay_prioritized,
            alpha=config.replay_alpha,
            beta_start=config.replay_beta_start,
            beta_increment=config.replay_beta_increment,
        )
        self.optimize_steps = 0
        self.train_iterations = 0
        self.loss_history: List[float] = []

        self.scheduler = ExponentialLR(self.agent.optimizer, gamma=config.learning_rate_gamma)
        self.elo_tracker = None
        self.league = LeagueManager(model_dir=config.model_dir, max_versions=config.max_versions, seed=config.seed)
        self.evaluator = Evaluator(
            EvaluatorConfig(
                games=config.eval_games,
                promotion_win_rate=config.eval_win_rate_threshold,
                base_seed=10_000 if config.base_seed is None else int(config.base_seed) + 10_000,
                parallel_workers=config.eval_parallel_workers,
            ),
            elo_tracker=self._build_elo_tracker(),
        )

        self.model_dir = Path(config.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.writer = self._build_writer()
        self._configure_determinism()

    def optimize_step(self) -> Optional[float]:
        """Run one optimization step from replay buffer."""
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch, indices, is_weights = self.replay_buffer.sample(self.config.batch_size)
        tensors = ReplayBuffer.to_tensors(batch, weights=is_weights, device=self.agent.device)

        q_values = self.agent.policy_net(tensors["states"]).gather(1, tensors["actions"].unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.agent.target_net(tensors["next_states"])
            next_q[~tensors["next_legal_masks"]] = -1e9
            max_next_q = next_q.max(dim=1).values
            target_q = tensors["rewards"] + (1.0 - tensors["dones"]) * self.agent.gamma * max_next_q

        td_errors = target_q - q_values
        self.replay_buffer.update_priorities(indices, np.abs(td_errors.detach().cpu().numpy()) + 1e-6)
        per_sample_loss = F.smooth_l1_loss(q_values, target_q, reduction="none")
        loss = (per_sample_loss * tensors["weights"]).mean()
        self.agent.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), self.config.gradient_clip_norm)
        self.agent.optimizer.step()
        self.scheduler.step()

        self.optimize_steps += 1
        if self.optimize_steps % self.config.target_sync_interval == 0:
            self.agent.sync_target_network()
        self.agent.decay_epsilon()
        loss_value = float(loss.item())
        self.loss_history.append(loss_value)

        if self.writer is not None:
            self.writer.add_scalar("train/loss", loss_value, self.optimize_steps)
            self.writer.add_scalar("train/epsilon", self.agent.epsilon, self.optimize_steps)
            self.writer.add_scalar("train/lr", self.agent.optimizer.param_groups[0]["lr"], self.optimize_steps)

        if self.optimize_steps % self.config.checkpoint_every_steps == 0:
            self._save_training_checkpoint(tag=f"step_{self.optimize_steps}")

        return loss_value

    def run_training_loop(self) -> None:
        """
        Main training loop:
        1) self-play generation
        2) replay storage
        3) optimization
        4) evaluation vs best
        5) promotion decision
        """
        bootstrap_path = self.model_dir / "bootstrap.pt"
        if not self.league.has_best():
            self.agent.save(
                bootstrap_path,
                metadata={"step": 0, "elo": 1200.0, "iteration": 0, "tag": "bootstrap"},
            )
            bootstrap_entry = self.league.add_version_from_checkpoint(
                source_checkpoint=bootstrap_path,
                elo=1200.0,
                step=0,
                promote_best=True,
            )
            LOGGER.info("Initialized league with %s", bootstrap_entry.name)

        for iteration in range(1, self.config.iterations + 1):
            self.train_iterations = iteration
            summary = self._generate_self_play_data(iteration)
            loss_values = self._run_optimization_steps(self.config.training_steps_per_iteration)
            avg_loss = float(np.mean(loss_values)) if loss_values else float("nan")

            candidate_path = self.model_dir / "candidate_latest.pt"
            self.agent.save(
                candidate_path,
                metadata={
                    "step": self.optimize_steps,
                    "iteration": iteration,
                    "elo": 1200.0,  # updated after evaluation
                    "buffer_size": len(self.replay_buffer),
                },
            )

            report = self._evaluate_and_maybe_promote(candidate_path, iteration)
            LOGGER.info(
                "Iteration %d | selfplay=%s buffer=%d avg_loss=%.6f win_rate=%.3f promoted=%s",
                iteration,
                summary,
                len(self.replay_buffer),
                avg_loss,
                report["win_rate"],
                report["promoted"],
            )

            if self.writer is not None:
                self.writer.add_scalar("iter/avg_loss", avg_loss, iteration)
                self.writer.add_scalar("iter/replay_size", len(self.replay_buffer), iteration)
                self.writer.add_scalar("iter/win_rate_vs_best", report["win_rate"], iteration)
                self.writer.add_scalar("iter/challenger_elo", report["challenger_elo"], iteration)
                self.writer.add_scalar("iter/best_elo", report["best_elo"], iteration)
                self.writer.add_scalar("iter/red_wins", summary["red_wins"], iteration)
                self.writer.add_scalar("iter/black_wins", summary["black_wins"], iteration)
                self.writer.add_scalar("iter/draws", summary["draws"], iteration)

            if self.config.save_every_iteration:
                self._save_training_checkpoint(tag=f"iter_{iteration}")

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def _generate_self_play_data(self, iteration: int) -> Dict[str, int]:
        reward_cfg = RewardConfig(
            win_reward=self.config.win_reward,
            loss_reward=self.config.loss_reward,
            capture_bonus=self.config.capture_bonus,
            lose_piece_penalty=self.config.lose_piece_penalty,
            draw_reward=-self.config.draw_penalty,
            step_penalty=self.config.step_penalty,
        )
        runner = SelfPlayRunner(
            SelfPlayConfig(
                max_plies=self.config.max_plies,
                no_progress_limit=self.config.no_progress_limit,
                parallel_workers=self.config.parallel_workers,
                base_seed=None if self.config.base_seed is None else int(self.config.base_seed) + iteration * 1000,
                log_every=self.config.log_every_games,
            ),
            reward_config=reward_cfg,
        )

        best_entry = self.league.get_best()
        sample_best = best_entry is None or random.random() < self.config.sample_best_probability
        if sample_best:
            opponent = best_entry
        else:
            opponent = self.league.sample_opponent(include_best=True)

        current_checkpoint = self.model_dir / "selfplay_current.pt"
        self.agent.save(current_checkpoint, metadata={"step": self.optimize_steps, "iteration": iteration})
        current_spec = PolicySpec(kind="dqn_checkpoint", checkpoint_path=str(current_checkpoint), epsilon=self.agent.epsilon)

        if opponent is None:
            opponent_spec = current_spec
        else:
            opponent_spec = PolicySpec(kind="dqn_checkpoint", checkpoint_path=opponent.path, epsilon=0.05)

        trajectories = runner.run_games_parallel_from_specs(
            red_spec=current_spec,
            black_spec=opponent_spec,
            n_games=self.config.games_per_iteration,
            replay_buffer=self.replay_buffer,
        )
        return runner.summarize(trajectories)

    def _run_optimization_steps(self, steps: int) -> List[float]:
        losses: List[float] = []
        for _ in range(steps):
            loss = self.optimize_step()
            if loss is not None:
                losses.append(loss)
        return losses

    def _evaluate_and_maybe_promote(self, candidate_path: Path, iteration: int) -> Dict[str, float | bool]:
        best_entry = self.league.get_best()
        if best_entry is None:
            entry = self.league.add_version_from_checkpoint(candidate_path, elo=1200.0, step=self.optimize_steps, promote_best=True)
            self.league.promote_existing_entry(entry.name)
            return {"win_rate": 1.0, "promoted": True, "challenger_elo": 1200.0, "best_elo": 1200.0}

        challenger_name = f"candidate_iter_{iteration}"
        report = self.evaluator.evaluate_checkpoints(
            challenger_name=challenger_name,
            challenger_path=candidate_path,
            best_name=best_entry.name,
            best_path=best_entry.path,
        )
        entry = self.league.add_version_from_checkpoint(
            source_checkpoint=candidate_path,
            elo=report.challenger_elo,
            step=self.optimize_steps,
            promote_best=report.promoted,
        )
        self.league.set_entry_elo(best_entry.name, report.best_elo)
        self.league.set_entry_elo(entry.name, report.challenger_elo)
        if report.promoted:
            self.league.promote_existing_entry(entry.name)
        return {
            "win_rate": report.win_rate,
            "promoted": report.promoted,
            "challenger_elo": report.challenger_elo,
            "best_elo": report.best_elo,
        }

    def _save_training_checkpoint(self, tag: str) -> None:
        path = self.model_dir / f"trainer_{tag}.pt"
        best_elo = 1200.0
        best_entry = self.league.get_best()
        if best_entry is not None:
            best_elo = best_entry.elo
        self.agent.save(
            path,
            metadata={
                "tag": tag,
                "step": self.optimize_steps,
                "iteration": self.train_iterations,
                "elo": best_elo,
            },
        )

    def _configure_determinism(self) -> None:
        if self.config.seed is None:
            return
        seed = int(self.config.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_writer(self):
        if not self.config.tensorboard:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter  # local import to avoid hard dependency at CLI parse time
        except Exception:
            LOGGER.warning("TensorBoard writer unavailable; continuing without TensorBoard logging.")
            return None
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir))

    def _build_elo_tracker(self):
        from training.elo import EloTracker

        if self.elo_tracker is None:
            self.elo_tracker = EloTracker()
        return self.elo_tracker
