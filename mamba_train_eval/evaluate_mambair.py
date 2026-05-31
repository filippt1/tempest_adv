from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Set, Union

import numpy as np
from PIL import Image
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FullImageTempestDataset
from basicsr.archs.mambair_arch import MambaIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TEMPEST denoising models")

    # Checkpoint for MambaIR.
    parser.add_argument("--mambair-checkpoint", type=str, required=True, help="MambaIR checkpoint path")

    # Evaluation-specific arguments.
    parser.add_argument("--test-dir", type=str, required=True, help="Dataset split directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Evaluation outputs")
    parser.add_argument(
        "--visualize-count",
        type=int,
        default=10,
        help="Save N random [Noisy|Restored|GT] triplets for each model",
    )
    parser.add_argument("--visualize-seed", type=int, default=42, help="Random seed for visualization sampling")
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print intermediate patch metrics every N patches (1 = every patch)",
    )
    parser.add_argument(
        "--mambair-img-size",
        type=int,
        default=128,
        help="MambaIR img_size (should match training patch size)",
    )
    return parser.parse_args()


def _to_uint8_image(tensor: torch.Tensor, value_range: str = "neg_one_one") -> np.ndarray:
    """Convert [1, 1, H, W] or [1, H, W] tensor to uint8 [H, W]."""
    image = tensor.detach().float().cpu()
    if value_range == "neg_one_one":
        image = image.clamp(-1.0, 1.0)
        image = (image + 1.0) * 0.5
    elif value_range == "zero_one":
        image = image.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported value_range: {value_range}")

    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3:
        image = image[0]
    image = (image * 255.0).round().numpy().astype(np.uint8)
    return image


def _window_sum(image: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(image, pad_width=pad, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(0).cumsum(1)
    h, w = image.shape
    wsize = window
    return (
        integral[wsize : wsize + h, wsize : wsize + w]
        - integral[:h, wsize : wsize + w]
        - integral[wsize : wsize + h, :w]
        + integral[:h, :w]
    )


def _local_variance(image: np.ndarray, window: int = 7) -> np.ndarray:
    window_area = float(window * window)
    sum_x = _window_sum(image, window)
    sum_x2 = _window_sum(image * image, window)
    mean = sum_x / window_area
    mean2 = sum_x2 / window_area
    return np.maximum(0.0, mean2 - mean * mean)


def _three_component_ssim(gt_u8: np.ndarray, pred_u8: np.ndarray) -> float:
    gt = gt_u8.astype(np.float32) / 255.0
    pred = pred_u8.astype(np.float32) / 255.0

    ssim_score, ssim_map = structural_similarity(gt, pred, data_range=1.0, full=True)

    grad_y, grad_x = np.gradient(gt)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    edge_thresh = np.percentile(grad_mag, 90)

    local_var = _local_variance(gt, window=7)
    texture_thresh = np.percentile(local_var, 60)

    edge_mask = grad_mag >= edge_thresh
    texture_mask = (grad_mag < edge_thresh) & (local_var >= texture_thresh)
    smooth_mask = ~(edge_mask | texture_mask)

    def _masked_mean(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return float(ssim_score)
        return float(ssim_map[mask].mean())

    edge_ssim = _masked_mean(edge_mask)
    texture_ssim = _masked_mean(texture_mask)
    smooth_ssim = _masked_mean(smooth_mask)

    weights = np.array([edge_mask.mean(), texture_mask.mean(), smooth_mask.mean()], dtype=np.float32)
    if weights.sum() <= 0:
        return float(ssim_score)
    weights = weights / weights.sum()
    return float(weights[0] * edge_ssim + weights[1] * texture_ssim + weights[2] * smooth_ssim)


def _compute_patch_metrics(gt_u8: np.ndarray, pred_u8: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((gt_u8.astype(np.float32) - pred_u8.astype(np.float32)) ** 2))
    psnr = float(peak_signal_noise_ratio(gt_u8, pred_u8, data_range=255))
    ssim = float(structural_similarity(gt_u8, pred_u8, data_range=255))
    tssim = _three_component_ssim(gt_u8, pred_u8)

    return {"mse": mse, "psnr": psnr, "ssim": ssim, "tssim": tssim}


def _save_triplet(noisy_u8: np.ndarray, restored_u8: np.ndarray, gt_u8: np.ndarray, path: Path) -> None:
    concat = np.concatenate([noisy_u8, restored_u8, gt_u8], axis=1)
    Image.fromarray(concat).save(path)


def _save_single(image_u8: np.ndarray, path: Path) -> None:
    Image.fromarray(image_u8).save(path)


def _format_summary_row(name: str, metrics: Dict[str, float]) -> str:
    return (
        f"{name:<10} | "
        f"MSE: {metrics['mse']:>10.4f} | "
        f"PSNR: {metrics['psnr']:>8.4f} | "
        f"SSIM: {metrics['ssim']:>8.4f} | "
        f"3-SSIM: {metrics['tssim']:>8.4f}"
    )


def _average_metrics(metrics_sum: Dict[str, float], count: int) -> Dict[str, float]:
    if count <= 0:
        return {"mse": float("nan"), "psnr": float("nan"), "ssim": float("nan"), "tssim": float("nan")}
    return {k: v / count for k, v in metrics_sum.items()}


def _compute_baseline_metrics(
    loader: Iterable[Dict[str, Union[torch.Tensor, str]]],
    device: torch.device,
) -> Dict[str, float]:
    baseline_sum = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "tssim": 0.0}
    count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Baseline", unit="image"):
            gt = batch["clean"].to(device, non_blocking=True)
            noisy = batch["condition"].to(device, non_blocking=True)

            gt_u8 = _to_uint8_image(gt, value_range="neg_one_one")
            noisy_u8 = _to_uint8_image(noisy, value_range="neg_one_one")

            metrics = _compute_patch_metrics(gt_u8, noisy_u8)
            for key in baseline_sum:
                baseline_sum[key] += metrics[key]
            count += 1

    return _average_metrics(baseline_sum, count)


def _extract_sample_name(batch_name: object) -> str:
    if isinstance(batch_name, list) and batch_name:
        return str(batch_name[0])
    return str(batch_name)


def _load_state_dict(model: torch.nn.Module, checkpoint_path: str, expected_name: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        ckpt_model_name = ckpt.get("model_name")
        if ckpt_model_name and ckpt_model_name != expected_name:
            raise ValueError(
                f"Checkpoint model '{ckpt_model_name}' does not match the required '{expected_name}' model."
            )
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()


def _select_visual_indices(total: int, count: int, seed: int) -> Set[int]:
    if total <= 0 or count <= 0:
        return set()
    rng = np.random.RandomState(seed)
    chosen = rng.choice(total, size=min(count, total), replace=False)
    return set(int(idx) for idx in chosen.tolist())


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation script.")

    device = torch.device("cuda" if torch.cuda.device_count() > 0 else "cpu")

    dataset = FullImageTempestDataset(split_dir=args.test_dir, normalize_to_neg_one_one=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visual_indices = _select_visual_indices(len(dataset), args.visualize_count, args.visualize_seed)

    baseline_avg = _compute_baseline_metrics(loader, device)

    results: List[Dict[str, Union[float, str]]] = [{"model": "baseline", **baseline_avg}]

    model = MambaIR(
        upscale=1,
        in_chans=1,
        img_size=args.mambair_img_size,
        img_range=1.0,
        d_state=16,
        depths=[6, 6, 6, 6],
        embed_dim=180,
        mlp_ratio=1.2,
    ).to(device)
    _load_state_dict(model, args.mambair_checkpoint, "mambair", device)

    model_output_dir = output_dir / "mambair"
    visuals_dir = model_output_dir / "visuals"
    restored_dir = model_output_dir / "restored"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    restored_dir.mkdir(parents=True, exist_ok=True)

    restored_sum = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "tssim": 0.0}

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating mambair", unit="image")
        for idx, batch in enumerate(pbar):
            gt = batch["clean"].to(device, non_blocking=True)
            noisy = batch["condition"].to(device, non_blocking=True)

            noisy_input = (noisy + 1.0) * 0.5
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=noisy.is_cuda):
                restored = model(noisy_input)
            restored = restored.clamp(0.0, 1.0)
            restored_u8 = _to_uint8_image(restored, value_range="zero_one")

            gt_u8 = _to_uint8_image(gt, value_range="neg_one_one")
            noisy_u8 = _to_uint8_image(noisy, value_range="neg_one_one")

            restored_metrics = _compute_patch_metrics(gt_u8, restored_u8)
            for key in restored_sum:
                restored_sum[key] += restored_metrics[key]

            sample_name = _extract_sample_name(batch["name"])
            out_name = f"{idx:06d}_{sample_name}.png"
            _save_single(restored_u8, restored_dir / out_name)

            if idx in visual_indices:
                _save_triplet(noisy_u8, restored_u8, gt_u8, visuals_dir / out_name)

            if args.log_every > 0 and ((idx + 1) % args.log_every == 0):
                tqdm.write(
                    (
                        f"model=mambair image={idx + 1}/{len(dataset)} name={sample_name} | "
                        f"restored[mse={restored_metrics['mse']:.3f}, psnr={restored_metrics['psnr']:.3f}, "
                        f"ssim={restored_metrics['ssim']:.4f}, 3ssim={restored_metrics['tssim']:.4f}]"
                    )
                )

            pbar.set_postfix(psnr=f"{restored_metrics['psnr']:.3f}", ssim=f"{restored_metrics['ssim']:.4f}")

    restored_avg = _average_metrics(restored_sum, len(dataset))
    results.append({"model": "mambair", **restored_avg})

    print("\n=== Evaluation Summary ===")
    print("Model: mambair")
    print(f"Total images evaluated: {len(dataset)}")
    print(_format_summary_row("Baseline", baseline_avg))
    print(_format_summary_row("Restored", restored_avg))
    print(f"Visualizations saved to: {visuals_dir}")
    print(f"Restored outputs saved to: {restored_dir}")

    summary_path = output_dir / "metrics_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["model", "mse", "psnr", "ssim", "3ssim"])
        for row in results:
            writer.writerow(
                [
                    row["model"],
                    f"{row['mse']:.6f}",
                    f"{row['psnr']:.6f}",
                    f"{row['ssim']:.6f}",
                    f"{row['tssim']:.6f}",
                ]
            )


if __name__ == "__main__":
    main()

