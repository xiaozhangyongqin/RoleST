"""Clean training and zero-shot cross-city evaluation entrypoint for RoleST.

Example:
    python -m rolest.train \
        --source SD --targets GBA GLA GBA_2020 GLA_2020 \
        --data-root datasets --output-dir runs/RoleST_SD
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from .config import RoleSTConfig
from .data import MaskedMetrics, TrafficWindowDataset, compute_source_stats
from .model import RoleST
from .roles import StaticSourceGraphPrior, fit_flow_roles, load_adjacency


def city_scale_jitter(x: torch.Tensor, y: torch.Tensor, width: float) -> tuple[torch.Tensor, torch.Tensor]:
    if width <= 0.0:
        return x, y
    factors = torch.empty((x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype).uniform_(-width, width).exp_()
    x_aug = x.clone()
    x_aug[..., 0] = x[..., 0] * factors[..., 0]
    return x_aug, y * factors


def role_st_loss(
    out: dict[str, torch.Tensor],
    target: torch.Tensor,
    labels: torch.Tensor,
    kt: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    norm_scale: float,
    role_weight: float = 0.1,
    transition_weight: float = 0.1,
    null_val: float = 0.0,
) -> torch.Tensor:
    mask = torch.isfinite(target) & (target != null_val)
    pred_mae = (out["prediction"][mask] - target[mask]).abs().mean() / max(norm_scale, 1e-6)

    logits = out["role_logits"]  # [B, N, P]
    avail = out["available"]     # [B, N]
    target_labels = labels.unsqueeze(0).expand_as(avail)
    role_ce = F.cross_entropy(logits[avail], target_labels[avail]) if avail.any() else logits.new_tensor(0.0)

    kl_loss = logits.new_tensor(0.0)
    if transition_weight > 0.0 and active_rows.any():
        ks = out["transition"].clamp_min(1e-8)
        kt_safe = kt.clamp_min(1e-8)
        row_kl = (kt_safe * (kt_safe.log() - ks.log())).sum(dim=-1)
        kl_loss = row_kl[active_rows].mean()

    return pred_mae + role_weight * role_ce + transition_weight * kl_loss


@torch.no_grad()
def evaluate_dataset(model: RoleST, ds: TrafficWindowDataset, batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    metrics = MaskedMetrics(null_val=model.config.null_val or 0.0)
    for x, y in DataLoader(ds, batch_size=batch_size, shuffle=False):
        pred = model.predict(x.to(device, non_blocking=True))
        metrics.update(pred, y.to(device, non_blocking=True))
    return metrics.compute()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and zero-shot evaluate RoleST.")
    ap.add_argument("--source", required=True, help="Source city directory name (e.g., SD, GBA, GLA)")
    ap.add_argument("--targets", nargs="+", required=True, help="Unseen target city directory names")
    ap.add_argument("--data-root", default="datasets")
    ap.add_argument("--output-dir", default="runs/RoleST")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--steps-per-epoch", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--city-scale-jitter", type=float, default=0.10)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TrafficWindowDataset(root / args.source, split="train")
    valid_ds = TrafficWindowDataset(root / args.source, split="valid")
    mean_s, std_s = compute_source_stats(train_ds.raw_flow)

    cfg = RoleSTConfig(norm_mean=mean_s, norm_scale=std_s)
    labels_np = fit_flow_roles(train_ds.raw_flow, num_roles=cfg.num_roles, seed=args.seed)
    adj = load_adjacency(root / args.source / "adj_mx.pkl")
    prior = StaticSourceGraphPrior(adj, labels_np, cfg.num_roles, device)

    model = RoleST(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 65, 85], gamma=0.3)

    best_val, best_epoch, bad = float("inf"), 0, 0
    best_path = out_dir / "rolest_best.pt"
    n_samples = args.steps_per_epoch * args.batch_size

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        gen = torch.Generator().manual_seed(args.seed + epoch)
        sampler = RandomSampler(train_ds, replacement=True, num_samples=n_samples, generator=gen)
        loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)

        # Linear warm-up for transition KL weight: 0 for ep<=5, ramps over ep 6..10 to 0.1
        if epoch <= 5:
            tw = 0.0
        elif epoch <= 10:
            tw = 0.1 * ((epoch - 5) / 5.0)
        else:
            tw = 0.1

        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, y = city_scale_jitter(x, y, args.city_scale_jitter)
            out = model(x)
            kt, active_rows = prior.coarse_target(out["available"])
            loss = role_st_loss(
                out, y, prior.labels, kt, active_rows,
                norm_scale=cfg.norm_scale, role_weight=0.1, transition_weight=tw,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())

        scheduler.step()
        val_m = evaluate_dataset(model, valid_ds, args.batch_size, device)
        val_mae = val_m["MAE"]
        if val_mae < best_val:
            best_val, best_epoch, bad = val_mae, epoch, 0
            torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict(), "epoch": epoch}, best_path)
        else:
            bad += 1

        if epoch <= 5 or epoch % 5 == 0 or epoch == best_epoch:
            print(f"[ep {epoch:3d}] loss={total_loss/len(loader):.4f} val_mae={val_mae:.2f} best={best_val:.2f} ({time.perf_counter()-t0:.1f}s)", flush=True)
        if bad >= args.patience:
            print(f"Early stopping at epoch {epoch} (best={best_val:.2f} @ ep {best_epoch})", flush=True)
            break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    results = {}
    for city in [args.source, *args.targets]:
        test_ds = TrafficWindowDataset(root / city, split="test")
        m = evaluate_dataset(model, test_ds, args.batch_size, device)
        results[city] = {"MAE": round(m["MAE"], 2), "RMSE": round(m["RMSE"], 2)}
        print(f"[Zero-Shot {args.source} -> {city}] MAE={m['MAE']:.2f}, RMSE={m['RMSE']:.2f}", flush=True)

    (out_dir / "results.json").write_text(json.dumps({"source": args.source, "best_epoch": best_epoch, "scores": results}, indent=2))


if __name__ == "__main__":
    main()
