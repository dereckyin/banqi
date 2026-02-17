"""Experience replay buffer with optional prioritized sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class Transition:
    """One RL transition."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    legal_mask: np.ndarray
    next_legal_mask: np.ndarray


class ReplayBuffer:
    """Circular replay buffer supporting uniform and prioritized replay."""

    def __init__(
        self,
        capacity: int = 50_000,
        prioritized: bool = False,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_increment: float = 1e-4,
        epsilon: float = 1e-6,
    ) -> None:
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self._buffer: List[Optional[Transition]] = [None] * capacity
        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._next_idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(self, transition: Transition, priority: Optional[float] = None) -> None:
        idx = self._next_idx
        self._buffer[idx] = transition

        if self.prioritized:
            max_priority = float(self._priorities[: self._size].max()) if self._size > 0 else 1.0
            self._priorities[idx] = max(max_priority, priority if priority is not None else max_priority)
        else:
            self._priorities[idx] = 1.0

        self._next_idx = (self._next_idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Return (samples, indices, is_weights)."""
        if batch_size > self._size:
            raise ValueError("Not enough samples in replay buffer.")

        if not self.prioritized:
            indices = np.random.choice(self._size, size=batch_size, replace=False)
            weights = np.ones(batch_size, dtype=np.float32)
        else:
            priorities = self._priorities[: self._size] + self.epsilon
            scaled = priorities ** self.alpha
            probs = scaled / scaled.sum()
            indices = np.random.choice(self._size, size=batch_size, replace=False, p=probs)

            self.beta = min(1.0, self.beta + self.beta_increment)
            weights = (self._size * probs[indices]) ** (-self.beta)
            weights /= weights.max()
            weights = weights.astype(np.float32)

        samples = [self._buffer[int(i)] for i in indices]
        return [s for s in samples if s is not None], indices.astype(np.int64), weights

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float]) -> None:
        """Update sampled transition priorities from TD errors."""
        if not self.prioritized:
            return
        for idx, priority in zip(indices, priorities):
            self._priorities[int(idx)] = max(float(priority), self.epsilon)

    def clear(self) -> None:
        self._buffer = [None] * self.capacity
        self._priorities.fill(0.0)
        self._next_idx = 0
        self._size = 0

    def all_transitions(self) -> Tuple[Transition, ...]:
        return tuple(t for t in self._buffer[: self._size] if t is not None)

    @staticmethod
    def to_tensors(
        transitions: Sequence[Transition],
        weights: np.ndarray,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Convert sampled transitions to batched tensors on device."""
        return {
            "states": torch.from_numpy(np.stack([t.state for t in transitions])).float().to(device),
            "actions": torch.from_numpy(np.array([t.action for t in transitions], dtype=np.int64)).to(device),
            "rewards": torch.from_numpy(np.array([t.reward for t in transitions], dtype=np.float32)).to(device),
            "next_states": torch.from_numpy(np.stack([t.next_state for t in transitions])).float().to(device),
            "dones": torch.from_numpy(np.array([t.done for t in transitions], dtype=np.float32)).to(device),
            "next_legal_masks": torch.from_numpy(np.stack([t.next_legal_mask for t in transitions])).to(device),
            "weights": torch.from_numpy(weights.astype(np.float32)).to(device),
        }
