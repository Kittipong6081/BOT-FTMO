"""
v6.9 E2 — Auxiliary-aware RolloutBuffer (PPO + auxiliary task)

Extends `stable_baselines3.common.buffers.RolloutBuffer` with an extra
per-step `aux_targets` field used by `AuxAwarePPO` to compute regression
auxiliary loss alongside the standard PPO objective.

Reference: arXiv 2411.01456 — auxiliary task improves DRL forex performance
by giving the policy network an extra inductive bias on signal outcome.
"""
from __future__ import annotations

from typing import Generator, NamedTuple

import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer


class AuxRolloutBufferSamples(NamedTuple):
    """Mirror of RolloutBufferSamples + aux_targets."""
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    aux_targets: th.Tensor


class AuxRolloutBuffer(RolloutBuffer):
    """RolloutBuffer + per-step aux_targets (regression target)."""

    def reset(self) -> None:
        super().reset()
        # shape (buffer_size, n_envs) — same layout as rewards
        self.aux_targets = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, *args, aux_target: np.ndarray | None = None, **kwargs) -> None:
        """Accept extra `aux_target` kwarg before delegating to parent.

        aux_target shape: (n_envs,) per step. Default 0.0 if missing
        (graceful degradation when env doesn't supply info['aux_target']).
        """
        if aux_target is None:
            aux_target = np.zeros((self.n_envs,), dtype=np.float32)
        # Record at current pos (parent .add increments pos)
        self.aux_targets[self.pos] = np.asarray(aux_target, dtype=np.float32).flatten()
        super().add(*args, **kwargs)

    def get(self, batch_size: int | None = None) -> Generator[AuxRolloutBufferSamples, None, None]:
        """Override: also flatten aux_targets along with parent tensors."""
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "aux_targets",  # NEW: include in flatten step
            ]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env=None,
    ) -> AuxRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.aux_targets[batch_inds].flatten(),  # NEW
        )
        return AuxRolloutBufferSamples(*tuple(map(self.to_torch, data)))
