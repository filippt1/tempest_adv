"""Inference utilities for full-resolution Independent Conditional Flow Matching."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image
import torch
from torch import Tensor
from torchvision import transforms

from model_advanced import AdvancedConditionalUNet
from scheduler import ConditionalFlowMatchingScheduler


def load_grayscale_tensor(path: str | Path, device: torch.device) -> Tensor:
    image = Image.open(path).convert("L")
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return tensor * 2.0 - 1.0


def save_grayscale_tensor(tensor: Tensor, path: str | Path) -> None:
    image = tensor.detach().cpu().clamp(-1.0, 1.0)
    image = (image + 1.0) * 0.5
    image = image.squeeze(0).squeeze(0)
    pil = transforms.ToPILImage()(image)
    pil.save(path)


def _positions(length: int, tile_size: int, stride: int) -> List[int]:
    if tile_size > length:
        raise ValueError(f"tile_size={tile_size} larger than image dimension={length}")

    positions = list(range(0, length - tile_size + 1, stride))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def gaussian_weight_mask(tile_size: int, sigma_scale: float = 0.125, device: torch.device | None = None) -> Tensor:
    """Build a 2D Gaussian blend mask used for overlap-tile stitching."""

    coords = torch.linspace(-1.0, 1.0, tile_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    sigma = sigma_scale
    weight = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    weight = weight / weight.max().clamp(min=1e-8)
    return weight.unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def _predict_model_output(
    model: AdvancedConditionalUNet,
    x_t: Tensor,
    condition: Tensor,
    time: Tensor,
) -> Tensor:
    use_amp = x_t.is_cuda
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        return model(x_t, time * 1000.0, condition)


@torch.no_grad()
def infer_direct(
    model: AdvancedConditionalUNet,
    scheduler: ConditionalFlowMatchingScheduler,
    condition: Tensor,
) -> Tensor:
    """Restore by running full-resolution tensors directly through the CFM model."""

    model.eval()
    bsz = condition.shape[0]

    x_t = condition.clone()
    for timestep_idx in range(int(scheduler.time_schedule.numel()) - 1):
        t_val = scheduler.time_schedule[timestep_idx]
        tt = torch.full((bsz,), t_val, device=condition.device, dtype=torch.float32)
        pred_v = _predict_model_output(model, x_t, condition, tt)
        x_t = scheduler.step(pred_v, timestep_idx, x_t)

    return x_t


@torch.no_grad()
def infer_tiled(
    model: AdvancedConditionalUNet,
    scheduler: ConditionalFlowMatchingScheduler,
    condition: Tensor,
    tile_size: int = 256,
    stride: int = 128,
    sigma_scale: float = 0.125,
) -> Tensor:
    """Synchronized tiled CFM sampling with Gaussian blending at every timestep."""

    model.eval()
    bsz, _, height, width = condition.shape
    if bsz != 1:
        raise ValueError("infer_tiled currently expects batch_size=1 for full-resolution inference")

    ys = _positions(height, tile_size, stride)
    xs = _positions(width, tile_size, stride)

    mask = gaussian_weight_mask(tile_size, sigma_scale=sigma_scale, device=condition.device)

    x_t = condition.clone()

    for timestep_idx in range(int(scheduler.time_schedule.numel()) - 1):
        tt = scheduler.time_schedule[timestep_idx].view(1)

        velocity_accum = torch.zeros_like(x_t)
        weight_accum = torch.zeros_like(x_t)

        for y in ys:
            for x in xs:
                x_tile = x_t[:, :, y : y + tile_size, x : x + tile_size]
                c_tile = condition[:, :, y : y + tile_size, x : x + tile_size]

                pred_tile = _predict_model_output(model, x_tile, c_tile, tt)

                # Blend overlapping tile predictions using smooth Gaussian weights.
                velocity_accum[:, :, y : y + tile_size, x : x + tile_size] += pred_tile * mask
                weight_accum[:, :, y : y + tile_size, x : x + tile_size] += mask

        pred_v = velocity_accum / weight_accum.clamp(min=1e-8)
        x_t = scheduler.step(pred_v, timestep_idx, x_t)

    return x_t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Independent Conditional Flow Matching inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--condition", type=str, required=True, help="Noisy TEMPEST image path")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--mode", choices=["direct", "tiled"], default="tiled")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--sigma-scale", type=float, default=0.125)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    model = AdvancedConditionalUNet().to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict):
        ckpt_model_name = ckpt.get("model_name")
        if ckpt_model_name and ckpt_model_name != "advanced":
            raise ValueError(
                f"Checkpoint model '{ckpt_model_name}' does not match the required 'advanced' model."
            )
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)

    scheduler = ConditionalFlowMatchingScheduler(device=device)
    scheduler.set_integration_schedule(num_inference_steps=args.inference_steps, device=device)

    condition = load_grayscale_tensor(args.condition, device=device)

    if args.mode == "direct":
        denoised = infer_direct(model, scheduler, condition)
    else:
        denoised = infer_tiled(
            model,
            scheduler,
            condition,
            tile_size=args.tile_size,
            stride=args.stride,
            sigma_scale=args.sigma_scale,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_grayscale_tensor(denoised, output_path)
    print(f"Saved denoised output to: {output_path}")


if __name__ == "__main__":
    main()

