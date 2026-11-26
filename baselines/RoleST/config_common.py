"""Shared BasicTS v0.5.8 EasyDict configuration builder for RoleST."""

from __future__ import annotations

import os
import sys
from easydict import EasyDict

from .arch import RoleST


def build_basicts_cfg(
    data_name: str,
    norm_mean: float,
    norm_scale: float,
    num_epochs: int = 100,
    batch_size: int = 32,
) -> EasyDict:
    sys.path.append(os.path.abspath(__file__ + "/../../.."))
    from basicts.data import TimeSeriesForecastingDataset
    from basicts.metrics import masked_mae, masked_mape, masked_rmse
    from basicts.runners import SimpleTimeSeriesForecastingRunner

    cfg = EasyDict()
    cfg.DESCRIPTION = f"RoleST ({data_name}) BasicTS configuration"
    cfg.GPU_NUM = 1
    cfg.RUNNER = SimpleTimeSeriesForecastingRunner

    cfg.DATASET = EasyDict()
    cfg.DATASET.NAME = data_name
    cfg.DATASET.TYPE = TimeSeriesForecastingDataset
    cfg.DATASET.PARAM = EasyDict(
        dataset_name=data_name,
        train_val_test_ratio=[0.6, 0.2, 0.2],
        input_len=12,
        output_len=12,
    )

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = "RoleST"
    cfg.MODEL.ARCH = RoleST
    cfg.MODEL.PARAM = {
        "input_len": 12,
        "short_len": 6,
        "output_len": 12,
        "hidden_dim": 192,
        "num_roles": 48,
        "phase_harmonics": 4,
        "mixer_strength": 0.15,
        "spatial_gain_jitter": 0.5,
        "norm_mean": norm_mean,
        "norm_scale": norm_scale,
    }
    cfg.MODEL.FORWARD_FEATURES = [0, 1, 2]
    cfg.MODEL.TARGET_FEATURES = [0]

    cfg.METRICS = EasyDict()
    cfg.METRICS.FUNCS = EasyDict(MAE=masked_mae, MAPE=masked_mape, RMSE=masked_rmse)
    cfg.METRICS.TARGET = "MAE"
    cfg.METRICS.NULL_VAL = 0.0

    cfg.TRAIN = EasyDict()
    cfg.TRAIN.NUM_EPOCHS = num_epochs
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "RoleST", f"{data_name}_{num_epochs}_12_12")
    cfg.TRAIN.LOSS = masked_mae
    cfg.TRAIN.OPTIM = EasyDict()
    cfg.TRAIN.OPTIM.TYPE = "AdamW"
    cfg.TRAIN.OPTIM.PARAM = {"lr": 1.5e-3, "weight_decay": 1e-4}
    cfg.TRAIN.LR_SCHEDULER = EasyDict()
    cfg.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
    cfg.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [40, 65, 85], "gamma": 0.3}
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    cfg.TRAIN.DATA = EasyDict(BATCH_SIZE=batch_size, SHUFFLE=True)

    cfg.VAL = EasyDict(INTERVAL=1, DATA=EasyDict(BATCH_SIZE=batch_size))
    cfg.TEST = EasyDict(INTERVAL=1, DATA=EasyDict(BATCH_SIZE=batch_size))
    return cfg
