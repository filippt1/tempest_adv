from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    def tqdm(iterable, **_: object):  # type: ignore[misc]
        return iterable

from dataset import ExhaustiveTempestDataset, FullImageTempestDataset
from network_unet import UNetResTime


LOGGER_NAME = "train_flow_drunet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Flow-Matching DRUNet (UNetResTime) for TEMPEST denoising")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root containing train/ val/ test/")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Checkpoint and log directory")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps; omit to disable accumulation",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--cosine-eta-min", type=float, default=1e-6, help="Minimum LR for cosine annealing")
    parser.add_argument(
        "--cosine-t-max",
        type=int,
        default=0,
        help="Cosine cycle length in epochs; <=0 uses --epochs",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument(
        "--data-loading-mode",
        type=str,
        choices=("exhaustive-patches", "full-images"),
        default="exhaustive-patches",
        help="Use exhaustive patch crops or full clean/noisy images",
    )
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-center-crop", action="store_true", help="Use center crop for validation")
    parser.add_argument("--resume", type=str, default="", help="Path to a checkpoint to resume")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    data_root = Path(args.data_root)

    if args.data_loading_mode == "full-images":
        train_ds = FullImageTempestDataset(
            split_dir=data_root / "train",
            normalize_to_neg_one_one=True,
        )
        val_ds = FullImageTempestDataset(
            split_dir=data_root / "val",
            normalize_to_neg_one_one=True,
        )
    else:
        train_ds = ExhaustiveTempestDataset(
            split_dir=data_root / "train",
            patch_size=args.patch_size,
            normalize_to_neg_one_one=True,
        )
        val_ds = ExhaustiveTempestDataset(
            split_dir=data_root / "val",
            patch_size=args.patch_size,
            normalize_to_neg_one_one=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def _prepare_batch(batch: Dict[str, Tensor | str], device: torch.device) -> tuple[Tensor, Tensor]:
    clean = batch["clean"].to(device, non_blocking=True)
    condition = batch["condition"].to(device, non_blocking=True)
    return clean, condition


def _count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _setup_logging(output_dir: Path, resume: bool) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_mode = "a" if resume else "w"
    file_handler = logging.FileHandler(output_dir / "train.log", mode=file_mode, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
    use_amp: bool,
    accumulation_steps: int,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")

    total_loss = 0.0
    total_count = 0
    accumulated_display_loss = 0.0
    steps_in_current_accum = 0

    context = torch.enable_grad if is_train else torch.no_grad
    with context():
        pbar = tqdm(loader, desc="train" if is_train else "val", leave=False)
        num_batches = len(loader)

        if is_train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(pbar):
            clean, condition = _prepare_batch(batch, device)
            bsz = clean.shape[0]

            t = torch.rand((bsz,), device=device, dtype=torch.float32)
            t_reshaped = t.view(bsz, 1, 1, 1)
            x_t = (1.0 - t_reshaped) * condition + t_reshaped * clean
            target_velocity = clean - condition

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                pred_velocity = model(x_t, t * 1000, condition)
                loss = F.mse_loss(pred_velocity, target_velocity)

            should_step = ((i + 1) % accumulation_steps == 0) or ((i + 1) == num_batches)

            if is_train:
                assert optimizer is not None
                window_start = (i // accumulation_steps) * accumulation_steps
                steps_in_window = min(
                    accumulation_steps,
                    num_batches - window_start
                )
                loss_for_backward = loss / steps_in_window
                loss_for_backward.backward()

                if should_step:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            loss_item = loss.detach().item()
            total_loss += loss_item * bsz
            total_count += bsz
            accumulated_display_loss += loss_item
            steps_in_current_accum += 1

            if should_step and steps_in_current_accum > 0:
                effective_loss = accumulated_display_loss / steps_in_current_accum
                pbar.set_postfix(eff_loss=f"{effective_loss:.5f}")
                accumulated_display_loss = 0.0
                steps_in_current_accum = 0

    return total_loss / max(1, total_count)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "model_name": "flow_drunet",
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def _init_metric_logs(output_dir: Path, resume: bool) -> tuple[Path, Path]:
    jsonl_path = output_dir / "metrics.jsonl"
    csv_path = output_dir / "metrics.csv"

    if not resume:
        if jsonl_path.exists():
            jsonl_path.unlink()
        csv_path.write_text("epoch,train_loss,val_loss,best_val_loss,is_best,lr,timestamp_utc\n", encoding="utf-8")
    elif not csv_path.exists():
        csv_path.write_text("epoch,train_loss,val_loss,best_val_loss,is_best,lr,timestamp_utc\n", encoding="utf-8")

    return jsonl_path, csv_path


def _append_metric_log(
    jsonl_path: Path,
    csv_path: Path,
    *,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    is_best: bool,
    lr: float,
    logger: logging.Logger,
) -> None:
    payload = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "is_best": is_best,
        "lr": lr,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with jsonl_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload) + "\n")

        with csv_path.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    payload["epoch"],
                    payload["train_loss"],
                    payload["val_loss"],
                    payload["best_val_loss"],
                    payload["is_best"],
                    payload["lr"],
                    payload["timestamp_utc"],
                ]
            )
    except OSError as exc:
        logger.warning("Failed to write metric logs: %s", exc)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_accumulation_steps = args.accumulation_steps if args.accumulation_steps is not None else 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(output_dir, resume=bool(args.resume))
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    logger.info("Saved config to %s", output_dir / "config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders(args)

    model = UNetResTime(in_nc=2, out_nc=1).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cosine_t_max = args.cosine_t_max if args.cosine_t_max > 0 else args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=cosine_t_max, eta_min=args.cosine_eta_min)

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        ckpt_model_name = ckpt.get("model_name")
        if ckpt_model_name and ckpt_model_name != "flow_drunet":
            raise ValueError(
                f"Checkpoint model '{ckpt_model_name}' does not match the required 'flow_drunet' model."
            )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        logger.info(
            "Resumed from %s (next_epoch=%d, best_val_loss=%.6f)",
            args.resume,
            start_epoch,
            best_val_loss,
        )

    metrics_jsonl_path, metrics_csv_path = _init_metric_logs(output_dir, resume=bool(args.resume))

    logger.info("Device: %s", device)
    train_count = len(train_loader.dataset)  # type: ignore[arg-type]
    val_count = len(val_loader.dataset)  # type: ignore[arg-type]
    logger.info("Train samples: %d | Val samples: %d", train_count, val_count)
    logger.info("Model: flow_drunet")
    logger.info("Objective: Optimal Transport Flow Matching (velocity)")
    logger.info("Model params: %s", f"{_count_trainable_parameters(model):,}")
    logger.info("LR scheduler: cosine annealing (T_max=%d, eta_min=%.2e)", cosine_t_max, args.cosine_eta_min)
    logger.info("Data loading mode: %s", args.data_loading_mode)
    if args.accumulation_steps is None:
        logger.info("Gradient accumulation: disabled")
    else:
        logger.info("Gradient accumulation: %d step(s)", args.accumulation_steps)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            use_amp=args.amp and device.type == "cuda",
            accumulation_steps=train_accumulation_steps,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            use_amp=args.amp and device.type == "cuda",
            accumulation_steps=1,
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        epoch_msg = f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        print(epoch_msg)
        logger.info(epoch_msg)

        current_lr = float(optimizer.param_groups[0]["lr"])
        _append_metric_log(
            metrics_jsonl_path,
            metrics_csv_path,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            is_best=is_best,
            lr=current_lr,
            logger=logger,
        )
        scheduler.step()

        latest_path = output_dir / "checkpoint_latest.pt"
        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best_val_loss, args)
        logger.info("Saved latest checkpoint: %s", latest_path)

        if is_best:
            best_path = output_dir / "checkpoint_best.pt"
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val_loss, args)
            logger.info("New best checkpoint saved: %s", best_path)


if __name__ == "__main__":
    main()

