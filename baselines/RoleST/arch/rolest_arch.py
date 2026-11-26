"""BasicTS-compatible RoleST architecture."""

from __future__ import annotations

import torch
from torch import nn

from .rolest_layers import DynamicRoleMixer, ForecastHead, TemporalEncoder


class RoleST(nn.Module):
    """Graph-free RoleST forecaster compatible with BasicTS v0.5.8 runner API."""

    def __init__(
        self,
        input_len: int = 12,
        short_len: int = 6,
        output_len: int = 12,
        hidden_dim: int = 192,
        num_roles: int = 48,
        phase_harmonics: int = 4,
        mixer_strength: float = 0.15,
        spatial_gain_jitter: float = 0.5,
        norm_mean: float = 0.0,
        norm_scale: float = 1.0,
        clip_value: float = 10.0,
        min_observations: int = 2,
        null_val: float | None = 0.0,
    ) -> None:
        super().__init__()
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.num_roles = int(num_roles)
        self.mixer_strength = float(mixer_strength)
        self.spatial_gain_jitter = float(spatial_gain_jitter)
        self.norm_mean = float(norm_mean)
        self.norm_scale = float(norm_scale)
        self.null_val = null_val

        self.encoder = TemporalEncoder(
            input_len=input_len,
            short_len=short_len,
            output_len=output_len,
            hidden_dim=hidden_dim,
            phase_harmonics=phase_harmonics,
            norm_mean=norm_mean,
            norm_scale=norm_scale,
            clip_value=clip_value,
            min_observations=min_observations,
            null_val=null_val,
        )
        self.mixer = DynamicRoleMixer(hidden_dim=hidden_dim, num_roles=num_roles)
        self.head = ForecastHead(
            input_len=input_len,
            short_len=short_len,
            output_len=output_len,
            hidden_dim=hidden_dim,
            phase_harmonics=phase_harmonics,
        )

    def forward_with_roles(self, history_data: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(history_data)
        state = encoded["h"] + encoded["u"]
        mixed = self.mixer(state, encoded["u"], encoded["available"], encoded["observed_fraction"])

        gain = self.mixer_strength
        if self.training and self.spatial_gain_jitter > 0.0:
            w = self.spatial_gain_jitter
            gain = gain * (1.0 + (2.0 * torch.rand(state.shape[0], 1, 1, device=state.device, dtype=state.dtype) - 1.0) * w)
        fused = state + gain * mixed["message"]
        prediction = self.head(fused, encoded)
        return {
            "prediction": prediction,
            "role_logits": mixed["role_logits"],
            "assignment": mixed["assignment"],
            "transition": mixed["transition"],
            "available": encoded["available"],
        }

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor | None = None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """BasicTS standard forward signature returning [B, F, N, 1]."""
        return self.forward_with_roles(history_data)["prediction"]
