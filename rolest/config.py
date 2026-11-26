"""Clean configuration for RoleST (no legacy or ablation flags)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RoleSTConfig:
    input_len: int = 12
    short_len: int = 6
    output_len: int = 12
    hidden_dim: int = 192
    num_roles: int = 48
    phase_harmonics: int = 4
    mixer_strength: float = 0.15
    spatial_gain_jitter: float = 0.5
    norm_mean: float = 0.0
    norm_scale: float = 1.0
    clip_value: float = 10.0
    min_observations: int = 2
    null_val: float | None = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RoleSTConfig":
        valid = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**valid)
