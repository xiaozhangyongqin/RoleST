# RoleST: Structural Role Distillation for Graph-Free Cross-City Traffic Forecasting

Clean, standalone PyTorch implementation of **RoleST** with zero legacy flags or ablation variants.

## Architecture Overview
- **TemporalEncoder** (`model.py`): Dual-path node-shared encoder (short-term MLP + full-window 1D CNN) with per-window mean anchoring, frozen source scale $\sigma_s$, scale-free relative-level descriptors, and $K=4$ harmonic Fourier calendar phases.
- **DynamicRoleMixer** (`model.py`): Low-rank $O(NPd + P^2 d)$ dynamic role mixer ($P = 48$ structural roles, hidden width $d = 192$) that requires no target road graph or node embeddings at inference.
- **ForecastHead** (`model.py`): Shared level- and calendar-conditioned 3-path prediction head (base + cumulative horizon delta + direct history residual).
- **Training Supervision** (`roles.py`, `train.py`): Source-only training with inference-observable Flow-6 role classification (`CE`) and static source-graph role-relation alignment (`KL`).

## Quick Start

```bash
python -m rolest.train \
    --source SD \
    --targets GBA GLA GBA_2020 GLA_2020 CA \
    --data-root datasets \
    --output-dir runs/RoleST_SD
```

## Unified Hyperparameters
- `input_len = output_len = 12`, `short_len = 6`
- `hidden_dim = 192`, `num_roles = 48`, `phase_harmonics = 4` (1.05M parameters)
- `mixer_strength = 0.15`, `spatial_gain_jitter = 0.5`, `city_scale_jitter = 0.10`
- `AdamW` (`lr = 1.5e-3`, `weight_decay = 1e-4`), `MultiStepLR([40, 65, 85], gamma = 0.3)`, `batch_size = 32`, `epochs = 100`
