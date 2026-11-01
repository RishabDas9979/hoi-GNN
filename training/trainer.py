"""Trainer (Phase 5.2 / 5.3): AdamW, cosine schedule with warmup, AMP, gradient
accumulation, early stopping on validation mAP, best/last checkpoints.

Models with no trainable parameters (the prior baseline) skip the optimiser:
fit() just evaluates once and saves, so `train.py --model prior` still produces
a results folder in the same format as every other model.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from evaluation.metrics import evaluate_all
from training.batch_adapter import get_targets
from utils.checkpoint import save_checkpoint


def _scalars(d: Dict, prefix: str = "") -> Dict[str, float]:
    return {f"{prefix}{k}": v for k, v in d.items() if isinstance(v, (int, float))}


class Trainer:
    def __init__(self, model, loss_fn, tcfg: dict, device: torch.device, out_dir,
                 meta: dict, train_pos_counts: Optional[List[int]] = None):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.cfg, self.device = tcfg, device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.meta = meta  # model_name, runtime, cfg, seed
        self.train_pos_counts = train_pos_counts
        self.use_amp = bool(tcfg.get("amp", False)) and device.type == "cuda"

    # ---------- evaluation ----------
    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        ys, ss, ms = [], [], []
        loss_sum, count = 0.0, 0.0
        for batch in loader:
            batch = batch.to(self.device)
            y, mask = get_targets(batch)
            logits = self.model(batch).float()
            if logits.numel():
                n = float(mask.sum().item())
                loss_sum += float(self.loss_fn(logits, y, mask).item()) * n
                count += n
            ys.append(y.detach().cpu().numpy())
            ss.append(torch.sigmoid(logits).cpu().numpy())
            ms.append(mask.detach().cpu().numpy())
        a = self.meta["runtime"]["num_actions"]
        cat = lambda xs: np.concatenate(xs, 0) if xs else np.zeros((0, a))
        return cat(ys), cat(ss), cat(ms), (loss_sum / count if count else float("nan"))

    def evaluate(self, loader) -> Dict:
        y, s, m, loss = self.predict(loader)
        out = evaluate_all(y, s, m, self.train_pos_counts,
                           self.cfg.get("rare_below", 10), self.cfg.get("threshold", 0.5))
        out["loss"] = loss
        return out

    # ---------- training ----------
    def _lr_lambda(self, epochs: int, warmup: int):
        def f(e: int) -> float:
            if e < warmup:
                return (e + 1) / max(warmup, 1)
            t = (e - warmup) / max(epochs - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * min(t, 1.0))) * 0.99 + 0.01
        return f

    def _train_epoch(self, loader, opt, scaler) -> float:
        self.model.train()
        accum = max(int(self.cfg.get("grad_accum", 1)), 1)
        clip = self.cfg.get("grad_clip", 0.0)
        total, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = batch.to(self.device)
            y, mask = get_targets(batch)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(batch)
            loss = self.loss_fn(logits.float(), y, mask)
            if loss.requires_grad:
                scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(loader):
                if clip:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total += float(loss.item())
            n += 1
        return total / max(n, 1)

    def _save(self, name, epoch, score, history, opt=None, sched=None):
        save_checkpoint(self.out_dir / name, self.model, model_name=self.meta["model_name"],
                        runtime=self.meta["runtime"], cfg=self.meta["cfg"], seed=self.meta["seed"],
                        epoch=epoch, best_score=score, history=history, optimizer=opt, scheduler=sched)

    def fit(self, train_loader, val_loader) -> List[Dict]:
        params = [p for p in self.model.parameters() if p.requires_grad]
        history: List[Dict] = []

        if not params:  # e.g. prior baseline: nothing to optimise
            val = self.evaluate(val_loader)
            history.append({"epoch": 0, "train_loss": float("nan"), **_scalars(val, "val_")})
            score = val["mAP"] if not np.isnan(val["mAP"]) else float("-inf")
            for name in ("best.pt", "last.pt"):
                self._save(name, 0, score, history)
            return history

        epochs = int(self.cfg["epochs"])
        opt = torch.optim.AdamW(params, lr=self.cfg["lr"], weight_decay=self.cfg.get("weight_decay", 0.0))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, self._lr_lambda(epochs, int(self.cfg.get("warmup_epochs", 0))))
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        best, bad = float("-inf"), 0
        patience, min_delta = int(self.cfg.get("patience", 0)), float(self.cfg.get("min_delta", 0.0))

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, opt, scaler)
            val = self.evaluate(val_loader)
            sched.step()
            # monitor mAP; fall back to -val_loss if mAP is undefined (no positives in val)
            score = val["mAP"] if not np.isnan(val["mAP"]) else -val["loss"]
            history.append({"epoch": epoch, "train_loss": train_loss, "lr": opt.param_groups[0]["lr"],
                            **_scalars(val, "val_")})
            if score > best + min_delta:
                best, bad = score, 0
                self._save("best.pt", epoch, best, history, opt, sched)
            else:
                bad += 1
            self._save("last.pt", epoch, best, history, opt, sched)
            if patience and bad >= patience:
                break
        return history
