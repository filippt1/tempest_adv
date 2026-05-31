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
except ImportError:
    def tqdm(iterable, **_: object):
        return iterable

from dataset import ExhaustiveTempestDataset, FullImageTempestDataset
from NAFNet_arch import NAFNet


LOGGER_NAME = "train_nafnet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NAFNet for TEMPEST denoising")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root containing train/ val/ test/")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Checkpoint and log directory")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument(
        "--data-loading-mode",
        type=str,
        choices=("exhaustive-patches", "full-images"),
        default="exhaustive-patches",
    )
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--seed", type=int, default=42)
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
            normalize_to_neg_one_one=False,
        )
        val_ds = FullImageTempestDataset(
            split_dir=data_root / "val",
            normalize_to_neg_one_one=False,
        )
    else:
        train_ds = ExhaustiveTempestDataset(
            split_dir=data_root / "train",
            patch_size=args.patch_size,
            normalize_to_neg_one_one=False,
        )
        val_ds = ExhaustiveTempestDataset(
            split_dir=data_root / "val",
            patch_size=args.patch_size,
            normalize_to_neg_one_one=False,
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
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    use_amp: bool,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_count = 0

    context = torch.enable_grad if is_train else torch.no_grad
    with context():
        pbar = tqdm(loader, desc="train" if is_train else "val", leave=False)

        for batch in pbar:
            clean, condition = _prepare_batch(batch, device)
            bsz = clean.shape[0]

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                pred_clean = model(condition)
                loss = F.l1_loss(pred_clean, clean)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            loss_item = loss.detach().item()
            total_loss += loss_item * bsz
            total_count += bsz
            pbar.set_postfix(loss=f"{loss_item:.5f}")

    return total_loss / max(1, total_count)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "model_name": "nafnet",
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
            writer.writerow([
                payload["epoch"], payload["train_loss"], payload["val_loss"],
                payload["best_val_loss"], payload["is_best"], payload["lr"],
                payload["timestamp_utc"],
            ])
    except OSError as exc:
        logger.warning("Failed to write metric logs: %s", exc)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(output_dir, resume=bool(args.resume))
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders(args)

    model = NAFNet(
        img_channel=1,
        width=32,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp) if device.type == "cuda" else None

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        ckpt_model_name = ckpt.get("model_name")
        if ckpt_model_name and ckpt_model_name != "nafnet":
            raise ValueError(f"Checkpoint model '{ckpt_model_name}' != 'nafnet'.")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["lr_scheduler"])
        if "scaler" in ckpt and ckpt["scaler"] and scaler:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    metrics_jsonl_path, metrics_csv_path = _init_metric_logs(output_dir, resume=bool(args.resume))

    logger.info("Starting NAFNet training")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=args.amp and device.type == "cuda",
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            scaler=None,
            device=device,
            use_amp=args.amp and device.type == "cuda",
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        epoch_msg = f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        print(epoch_msg)
        logger.info(epoch_msg)

        current_lr = float(optimizer.param_groups[0]["lr"])
        _append_metric_log(
            metrics_jsonl_path, metrics_csv_path,
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            best_val_loss=best_val_loss, is_best=is_best, lr=current_lr,
            logger=logger,
        )
        scheduler.step()

        save_checkpoint(output_dir / "checkpoint_latest.pt", model, optimizer, scheduler, scaler, epoch, best_val_loss, args)
        if is_best:
            save_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer, scheduler, scaler, epoch, best_val_loss, args)


if __name__ == "__main__":
    main()

