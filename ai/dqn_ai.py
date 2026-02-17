"""DQN-based AI skeleton for Banqi."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

from ai.base_ai import BaseAI
from engine.board import ACTION_SPACE_SIZE, Board, Move

LOGGER = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """Simple convolutional Q-network over encoded board state."""

    def __init__(self, input_channels: int = 17, action_size: int = ACTION_SPACE_SIZE) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


@dataclass
class EpsilonSchedule:
    """Epsilon-greedy schedule parameters."""

    start: float = 1.0
    end: float = 0.05
    decay: float = 0.995


class DQNAI(BaseAI):
    """DQN policy wrapper with target network and model persistence."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_schedule: Optional[EpsilonSchedule] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self.gamma = gamma
        self.epsilon_schedule = epsilon_schedule or EpsilonSchedule()
        self.epsilon = self.epsilon_schedule.start
        self._rng = random.Random(seed)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

    def choose_move(self, board: Board) -> Move:
        """Choose action with epsilon-greedy policy over legal actions."""
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            raise RuntimeError("No legal moves available.")

        if self._rng.random() < self.epsilon:
            return self._rng.choice(legal_moves)

        state = self._encode_state_tensor(board)
        legal_mask = board.legal_action_mask()
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze(0)
            action = self._masked_argmax(q_values, legal_mask)
        move = board.action_to_move(action)
        if move is None or move not in legal_moves:
            return self._rng.choice(legal_moves)
        return move

    def _encode_state_tensor(self, board: Board) -> torch.Tensor:
        encoded = board.encode_state()
        return torch.from_numpy(encoded).float().unsqueeze(0).to(self.device)

    def _masked_argmax(self, q_values: torch.Tensor, legal_mask: np.ndarray) -> int:
        mask_tensor = torch.from_numpy(legal_mask).to(self.device)
        masked_q = q_values.clone()
        masked_q[~mask_tensor] = -1e9
        return int(torch.argmax(masked_q).item())

    def decay_epsilon(self) -> None:
        """Decay epsilon after training step/episode."""
        self.epsilon = max(self.epsilon_schedule.end, self.epsilon * self.epsilon_schedule.decay)

    def sync_target_network(self) -> None:
        """Copy policy network weights into target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str | Path, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "metadata": metadata or {},
        }
        torch.save(
            payload,
            path,
        )
        LOGGER.info("Saved DQN model to %s", path)

    def load(self, path: str | Path) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_state"])
        self.target_net.load_state_dict(checkpoint["target_state"])
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon_schedule.start))
        self.gamma = float(checkpoint.get("gamma", self.gamma))
        LOGGER.info("Loaded DQN model from %s", path)
        return dict(checkpoint.get("metadata", {}))
