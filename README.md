# RoleST: Structural Role Distillation for Graph-Free Cross-City Traffic Forecasting

Official clean PyTorch & BasicTS-compatible implementation of **RoleST**.

## Overview

**RoleST** uses source-city road topology as training-time privileged information and performs **graph-free zero-shot traffic forecasting** at inference in $O(N P d + P^2 d)$ time:

- **Temporal Encoder (`arch/rolest_layers.py`)**: Dual-path node-shared encoder (short-term MLP + full-window 1D CNN) with per-window mean anchoring, frozen source scale $\sigma_s$, scale-free relative-level descriptors, and $K=4$ harmonic Fourier calendar phases.
- **Dynamic Role Mixer (`arch/rolest_layers.py`)**: Low-rank dynamic structural-role attention ($P=48$ roles, hidden width $d=192$) requiring **no target road graph, no sensor coordinates, and no node embeddings** at inference.
- **Forecast Head (`arch/rolest_layers.py`)**: Level- and calendar-conditioned 3-path prediction head (base + cumulative horizon delta + direct history residual).
- **Training-Time Role Supervision (`roles.py`, `train.py`)**: Source-only Flow-6 role classification (`CE`) + static source-graph role-relation alignment (`KL`).

## Repository Structure (BasicTS Layout)

```text
RoleST/
├── README.md
├── requirements.txt
└── baselines/
    └── RoleST/
        ├── __init__.py
        ├── arch/
        │   ├── __init__.py
        │   ├── rolest_arch.py      # BasicTS-compatible RoleST model
        │   └── rolest_layers.py    # TemporalEncoder, DynamicRoleMixer, ForecastHead
        ├── roles.py                # Flow-6 role clustering & static source-graph prior K_T
        ├── data.py                 # Chronological 6:2:2 dataset & masked MAE/RMSE
        ├── train.py                # End-to-end zero-shot cross-city training & evaluation
        ├── config_common.py        # BasicTS v0.5.8 EasyDict config builder
        ├── SD.py                   # BasicTS config (San Diego)
        ├── GBA.py                  # BasicTS config (Greater Bay Area)
        └── GLA.py                  # BasicTS config (Greater Los Angeles)
```

## Installation

```bash
pip install -r requirements.txt
```

Prepare LargeST datasets (`SD`, `GBA`, `GLA`, `GBA_2020`, `GLA_2020`, `CA`) in BasicTS format under `datasets/<CITY>/` (`data.dat`, `desc.json`, `adj_mx.pkl`).

## Quick Start

### 1. Zero-Shot Cross-City Training & Evaluation (Recommended)

```bash
# Train on SD and evaluate zero-shot on GBA, GLA, GBA_2020, GLA_2020, and CA
python -m baselines.RoleST.train \
    --source SD \
    --targets GBA GLA GBA_2020 GLA_2020 CA \
    --data-root datasets \
    --output-dir runs/RoleST_SD

# Train on GBA and evaluate zero-shot on SD, GLA, and CA
python -m baselines.RoleST.train \
    --source GBA \
    --targets SD GLA CA \
    --data-root datasets \
    --output-dir runs/RoleST_GBA

# Train on GLA and evaluate zero-shot on SD, GBA, and CA
python -m baselines.RoleST.train \
    --source GLA \
    --targets SD GBA CA \
    --data-root datasets \
    --output-dir runs/RoleST_GLA
```

### 2. Drop-In Usage Inside BasicTS

Copy `baselines/RoleST/` into your `BasicTS/baselines/` directory and launch via BasicTS:

```bash
python experiments/train.py -c baselines/RoleST/SD.py --gpus '0'
```

## Unified Hyperparameters

| Parameter | Value |
| :--- | :---: |
| Horizon ($L \to F$) | $12 \to 12$ (3h at 15-min) |
| Hidden width $d$ / Roles $P$ | $192$ / $48$ (1.05M params) |
| Calendar harmonics $K$ | $4$ |
| `mixer_strength` / `spatial_gain_jitter` | $0.15$ / $0.5$ |
| Optimizer & Schedule | AdamW (`lr=1.5e-3`, `wd=1e-4`), MultiStepLR `[40, 65, 85]` ($\times 0.3$), `bs=32`, `100` epochs |

## Zero-Shot Benchmark Results (LargeST, MAE / RMSE)

| Transfer Direction | RoleST (Graph-Free) |
| :--- | :---: |
| **SD → GBA** (2019) | **33.70 / 55.42** |
| **SD → GLA** (2019) | **37.13 / 61.89** |
| **GBA → SD** (2019) | **31.86 / 52.47** |
| **GBA → GLA** (2019) | **32.63 / 52.20** |
| **GLA → SD** (2019) | **33.61 / 55.45** |
| **GLA → GBA** (2019) | **32.06 / 50.87** |
| **SD → GBA20** (2020) | **25.61 / 42.08** |
| **SD → GLA20** (2020) | **30.01 / 50.61** |
| **SD / GBA / GLA → CA** (8,600 sensors) | **32.20 / 29.00 / 28.52** |
