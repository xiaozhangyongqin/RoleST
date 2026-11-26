"""RoleST layers: TemporalEncoder, DynamicRoleMixer, ForecastHead."""

from __future__ import annotations

import math

import torch
from torch import nn


def masked_softmax(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    allowed = torch.broadcast_to(allowed, logits.shape)
    scores = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
    prob = scores.softmax(dim=-1) * allowed.to(scores.dtype)
    return prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class TemporalEncoder(nn.Module):
    """Dual-path node-shared encoder with window-mean anchoring and relative-level embedding."""

    def __init__(
        self,
        input_len: int = 12,
        short_len: int = 6,
        output_len: int = 12,
        hidden_dim: int = 192,
        phase_harmonics: int = 4,
        norm_mean: float = 0.0,
        norm_scale: float = 1.0,
        clip_value: float = 10.0,
        min_observations: int = 2,
        null_val: float | None = 0.0,
    ) -> None:
        super().__init__()
        self.input_len = int(input_len)
        self.short_len = int(short_len)
        self.output_len = int(output_len)
        self.hidden_dim = int(hidden_dim)
        self.phase_harmonics = int(phase_harmonics)
        self.norm_mean = float(norm_mean)
        self.norm_scale = float(norm_scale)
        self.clip_value = float(clip_value)
        self.min_observations = int(min_observations)
        self.null_val = null_val

        d = self.hidden_dim
        self.short = nn.Sequential(nn.Linear(2 * self.short_len, d), nn.GELU(), nn.Linear(d, d))
        self.context = nn.Sequential(nn.Conv1d(2, d, kernel_size=5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(8))
        self.context_proj = nn.Linear(8 * d, d)
        self.rel_level_proj = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))

    @staticmethod
    def _window_stats(values: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = valid.to(values.dtype)
        count = w.sum(dim=1, keepdim=True)
        mean = (values * w).sum(dim=1, keepdim=True) / count.clamp_min(1.0)
        var = (((values - mean) * w).square()).sum(dim=1, keepdim=True) / count.clamp_min(1.0)
        return mean.detach(), var.clamp_min(1e-12).sqrt().detach(), count

    def _calendar_phases(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tod, dow = history[:, -1, :, 1], history[:, -1, :, 2]
        delta = (history[:, -1, :, 1] - history[:, -2, :, 1]).remainder(1.0)
        delta = torch.where(delta > 1e-5, delta, torch.full_like(delta, 1.0 / 96.0))
        slots_per_day = torch.round(1.0 / delta).clamp_min(1.0).unsqueeze(-1)
        offsets = torch.arange(1, self.output_len + 1, device=history.device, dtype=history.dtype).view(1, 1, -1)
        slot_fut = torch.round(tod.unsqueeze(-1) * slots_per_day) + offsets
        tod_fut = slot_fut.remainder(slots_per_day) / slots_per_day
        days = torch.div(slot_fut, slots_per_day, rounding_mode="floor")
        dow_fut = (dow.unsqueeze(-1) + days / 7.0).remainder(1.0)

        two_pi = 2.0 * math.pi
        cur_list, fut_list = [], []
        for k in range(1, max(1, self.phase_harmonics) + 1):
            cur_list.extend((
                torch.sin(k * two_pi * tod), torch.cos(k * two_pi * tod),
                torch.sin(k * two_pi * dow), torch.cos(k * two_pi * dow),
            ))
            fut_list.extend((
                torch.sin(k * two_pi * tod_fut), torch.cos(k * two_pi * tod_fut),
                torch.sin(k * two_pi * dow_fut), torch.cos(k * two_pi * dow_fut),
            ))
        return torch.stack(cur_list, dim=-1), torch.stack(fut_list, dim=-1)

    def forward(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        traffic = history[..., 0]
        valid = torch.isfinite(traffic)
        if self.null_val is not None:
            valid = valid & (traffic != self.null_val)
        values = torch.where(valid, traffic, torch.zeros_like(traffic))

        mean_all, std_all, count_all = self._window_stats(values, valid)
        mean = torch.where(count_all > 0, mean_all, torch.full_like(mean_all, self.norm_mean))
        scale = torch.full_like(mean, self.norm_scale)
        normalized = torch.where(
            valid,
            ((values - mean) / scale).clamp(-self.clip_value, self.clip_value),
            torch.zeros_like(values),
        )

        b, length, n = values.shape
        short_in = torch.cat(
            (normalized[:, -self.short_len:].transpose(1, 2),
             valid[:, -self.short_len:].transpose(1, 2).to(values.dtype)),
            dim=-1,
        )
        h = self.short(short_in)

        ctx_in = torch.stack((normalized.transpose(1, 2), valid.transpose(1, 2).to(values.dtype)), dim=2)
        u = self.context_proj(self.context(ctx_in.reshape(b * n, 2, length)).flatten(1)).reshape(b, n, -1)

        available = count_all.squeeze(1) >= self.min_observations
        mean_node, scale_node = mean.transpose(1, 2), scale.transpose(1, 2)
        ref_vals = (mean_node.abs() + scale_node).squeeze(-1)
        per_sample = ref_vals.masked_fill(~available, float("nan"))
        level_ref = torch.nanmedian(per_sample, dim=1).values.nan_to_num(nan=1.0).clamp_min(1e-3).unsqueeze(-1).detach()

        rel_lvl = torch.log1p(mean_node.squeeze(-1).clamp_min(0.0)) / torch.log1p(level_ref).clamp_min(1e-3)
        std_ratio = torch.log1p(std_all.squeeze(1).clamp_min(0.0)) / torch.log1p(level_ref).clamp_min(1e-3)
        u = u + self.rel_level_proj(torch.stack((rel_lvl, std_ratio), dim=-1))

        current_phase, future_phase = self._calendar_phases(history)
        return {
            "h": h, "u": u, "normalized": normalized, "valid": valid, "available": available,
            "observed_fraction": valid.to(values.dtype).mean(dim=1).unsqueeze(-1),
            "mean": mean_node, "scale": scale_node, "level_reference": level_ref,
            "time_features": history[:, -1, :, 1:3],
            "current_phase": current_phase, "future_phase": future_phase,
        }


class DynamicRoleMixer(nn.Module):
    """Low-rank O(N P d + P^2 d) dynamic structural-role mixer."""

    def __init__(self, hidden_dim: int = 192, num_roles: int = 48) -> None:
        super().__init__()
        d, p = int(hidden_dim), int(num_roles)
        self.role_head = nn.Linear(d, p)
        self.value = nn.Linear(d, d, bias=False)
        self.base_relation = nn.Parameter(torch.zeros(p, p))
        self.role_embedding = nn.Parameter(torch.randn(p, d) / math.sqrt(d))
        self.query = nn.Linear(2 * d, d, bias=False)
        self.key = nn.Linear(2 * d, d, bias=False)
        self.gate = nn.Linear(2 * d + 1, d)

    def forward(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        available: torch.Tensor,
        observed_fraction: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, _, d = state.shape
        logits = self.role_head(context)
        q = logits.softmax(dim=-1)
        values = self.value(state)
        write = q * available.unsqueeze(-1).to(state.dtype)
        mass = write.sum(dim=1)
        z = torch.bmm(write.transpose(1, 2), values) / mass.unsqueeze(-1).clamp_min(1e-12)

        tokens = torch.cat((z, self.role_embedding.unsqueeze(0).expand(b, -1, -1)), dim=-1)
        relation = self.base_relation.unsqueeze(0).expand(b, -1, -1)
        relation = relation + torch.bmm(self.query(tokens), self.key(tokens).transpose(1, 2)) / math.sqrt(d)
        k = masked_softmax(relation, (mass > 1e-12).unsqueeze(1))

        raw = torch.bmm(q, torch.bmm(k, z))
        gate = torch.sigmoid(self.gate(torch.cat((state, context, observed_fraction), dim=-1)))
        return {"message": gate * raw, "role_logits": logits, "assignment": q, "transition": k, "role_state": z}


class ForecastHead(nn.Module):
    """Shared level- and multi-harmonic calendar-conditioned 3-path forecast head."""

    def __init__(
        self,
        input_len: int = 12,
        short_len: int = 6,
        output_len: int = 12,
        hidden_dim: int = 192,
        phase_harmonics: int = 4,
    ) -> None:
        super().__init__()
        self.input_len = int(input_len)
        self.short_len = int(short_len)
        self.output_len = int(output_len)
        d = int(hidden_dim)
        p_dim = 4 * max(1, int(phase_harmonics))

        self.output = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.output_len))
        self.level_net = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, 2 * d))
        self.recent_net = nn.Sequential(nn.Linear(2 * self.short_len, d), nn.GELU(), nn.Linear(d, d))
        self.history_direct = nn.Sequential(nn.Linear(2 * self.input_len, d), nn.GELU(), nn.Linear(d, self.output_len))
        self.horizon_step = nn.Embedding(self.output_len, d)
        self.delta_net = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.time_net = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, 2 * d))
        self.current_phase_net = nn.Sequential(nn.Linear(p_dim, d), nn.GELU(), nn.Linear(d, 2 * d))
        self.phase_net = nn.Sequential(nn.Linear(p_dim, d), nn.GELU(), nn.Linear(d, 2))

        for mod in (
            self.level_net, self.recent_net, self.history_direct,
            self.delta_net, self.time_net, self.current_phase_net, self.phase_net,
        ):
            nn.init.zeros_(mod[-1].weight)
            nn.init.zeros_(mod[-1].bias)

    @staticmethod
    def _level_features(encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        mean = encoded["mean"].squeeze(-1)
        scale = encoded["scale"].squeeze(-1)
        ref = encoded["level_reference"].clamp_min(1e-3)
        recent = encoded["normalized"][:, -4:].mean(dim=1)
        level = torch.log1p(mean.clamp_min(0.0)) / torch.log1p(ref).clamp_min(1e-3)
        return torch.stack(
            (level, torch.log(scale.clamp_min(1e-3)) - torch.log1p(mean.abs()),
             recent, encoded["observed_fraction"].squeeze(-1)),
            dim=-1,
        )

    def forward(self, fused: torch.Tensor, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        lvl_g, lvl_b = self.level_net(self._level_features(encoded)).chunk(2, dim=-1)
        fused = fused * (1.0 + lvl_g) + lvl_b

        t_g, t_b = self.time_net(encoded["time_features"]).chunk(2, dim=-1)
        fused = fused * (1.0 + t_g) + t_b

        cp_g, cp_b = self.current_phase_net(encoded["current_phase"]).chunk(2, dim=-1)
        fused = fused * (1.0 + cp_g) + cp_b

        fp_g, fp_b = self.phase_net(encoded["future_phase"]).chunk(2, dim=-1)
        fp_g, fp_b = fp_g.transpose(1, 2), fp_b.transpose(1, 2)

        normalized, valid = encoded["normalized"], encoded["valid"]
        recent = torch.cat(
            (normalized[:, -self.short_len:].transpose(1, 2),
             valid[:, -self.short_len:].transpose(1, 2).to(fused.dtype)),
            dim=-1,
        )
        fused = fused + self.recent_net(recent)

        b, n, _ = fused.shape
        steps = self.horizon_step.weight.unsqueeze(0).unsqueeze(0).expand(b, n, -1, -1)
        delta_in = torch.cat((fused.unsqueeze(2).expand(-1, -1, self.output_len, -1), steps), dim=-1)
        correction = torch.cumsum(self.delta_net(delta_in).squeeze(-1), dim=-1).transpose(1, 2).unsqueeze(-1)

        base = self.output(fused).transpose(1, 2).unsqueeze(-1)
        hist = torch.cat((normalized.transpose(1, 2), valid.transpose(1, 2).to(fused.dtype)), dim=-1)
        direct = self.history_direct(hist).transpose(1, 2).unsqueeze(-1)

        forecast = (base + correction + direct) * (1.0 + fp_g) + fp_b
        return forecast * encoded["scale"].unsqueeze(1) + encoded["mean"].unsqueeze(1)
