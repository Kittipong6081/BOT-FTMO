"""
v6.9 E2 — Auxiliary-aware ActorCriticPolicy

Adds a 3rd head (aux_head) that predicts signal outcome (regression) from the
actor's latent representation. The auxiliary loss in `AuxAwarePPO.train()` uses
`predict_aux()` and forces the trunk to encode features useful for both
action selection AND outcome prediction.

Reference: arXiv 2411.01456 — auxiliary task gives the policy net an extra
inductive bias toward representations informative of trade outcomes.
"""
from __future__ import annotations

import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


class AuxAwareACPolicy(ActorCriticPolicy):
    """ActorCriticPolicy + aux regression head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add aux head: trunk dim → 1 scalar (regression on outcome_pnl_ratio)
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.aux_head = nn.Linear(latent_dim_pi, 1)

    def predict_aux(self, obs: th.Tensor) -> th.Tensor:
        """Forward pass through actor trunk → aux_head only.

        Returns shape (batch_size,) — squeezed scalar prediction.
        Used by AuxAwarePPO.train() to compute MSE vs aux_targets.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
        return self.aux_head(latent_pi).squeeze(-1)
