"""Dataset utilities for conditional TEMPEST screenshot denoising."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTENSIONS: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class SampleEntry:
    """One clean screenshot and all available noisy capture paths."""

    name: str
    clean_path: Path
    noisy_paths: List[Path]


@dataclass(frozen=True)
class ExhaustivePatchInstruction:
    """One deterministic crop instruction for a specific clean/noisy pair."""

    sample_index: int
    noisy_index: int
    top: int
    left: int


@dataclass(frozen=True)
class FullImageInstruction:
    """One deterministic full-image clean/noisy pairing."""

    sample_index: int
    noisy_index: int


class TempestConditionalDataset(Dataset):
    """Loads clean/noisy TEMPEST pairs and applies synchronized random patch crops.

    The key trick is to stack clean + noisy tensors as channels, crop once, then split.
    That guarantees identical random crop coordinates for both images.
    """

    def __init__(
        self,
        split_dir: str | Path,
        patch_size: int = 256,
        normalize_to_neg_one_one: bool = True,
        random_crop: bool = True,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.patch_size = patch_size
        self.normalize_to_neg_one_one = normalize_to_neg_one_one
        self.random_crop = random_crop

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")

        self.samples = self._discover_samples(self.split_dir)
        if not self.samples:
            raise RuntimeError(f"No valid samples found in: {self.split_dir}")

        self.random_crop_op = transforms.RandomCrop(self.patch_size)
        self.center_crop_op = transforms.CenterCrop(self.patch_size)
        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def _is_image_file(path: Path) -> bool:
        return path.suffix.lower() in IMG_EXTENSIONS

    def _discover_samples(self, split_dir: Path) -> List[SampleEntry]:
        samples: List[SampleEntry] = []
        clean_files = [
            p for p in sorted(split_dir.iterdir()) if p.is_file() and self._is_image_file(p)
        ]

        for clean_path in clean_files:
            stem = clean_path.stem
            noisy_dir = split_dir / stem
            if not noisy_dir.is_dir():
                continue

            noisy_paths = [
                p for p in sorted(noisy_dir.iterdir()) if p.is_file() and self._is_image_file(p)
            ]
            if not noisy_paths:
                continue

            samples.append(SampleEntry(name=stem, clean_path=clean_path, noisy_paths=noisy_paths))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_grayscale_tensor(self, path: Path) -> Tensor:
        image = Image.open(path).convert("L")
        return self.to_tensor(image)

    def _apply_joint_crop(self, clean: Tensor, noisy: Tensor) -> tuple[Tensor, Tensor]:
        if clean.shape != noisy.shape:
            raise ValueError(
                f"Shape mismatch between clean {tuple(clean.shape)} and noisy {tuple(noisy.shape)}"
            )

        if clean.shape[-2] < self.patch_size or clean.shape[-1] < self.patch_size:
            raise ValueError(
                f"Image too small for patch_size={self.patch_size}: {tuple(clean.shape)}"
            )

        # Crucial: crop both images in lockstep by stacking them as channels first.
        stacked = torch.cat([clean, noisy], dim=0)  # [2, H, W]
        if self.random_crop:
            stacked = self.random_crop_op(stacked)
        else:
            stacked = self.center_crop_op(stacked)

        clean_crop = stacked[0:1]
        noisy_crop = stacked[1:2]
        return clean_crop, noisy_crop

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        entry = self.samples[index]

        clean = self._load_grayscale_tensor(entry.clean_path)
        noisy_path = random.choice(entry.noisy_paths)
        noisy = self._load_grayscale_tensor(noisy_path)

        clean_crop, noisy_crop = self._apply_joint_crop(clean, noisy)

        if self.normalize_to_neg_one_one:
            clean_crop = clean_crop * 2.0 - 1.0
            noisy_crop = noisy_crop * 2.0 - 1.0

        return {
            "clean": clean_crop,
            "condition": noisy_crop,
            "name": entry.name,
        }


class ExhaustiveTempestDataset(Dataset):
    """Deterministic exhaustive 2D grid crops for all clean/noisy pairings."""

    def __init__(
        self,
        split_dir: str | Path,
        patch_size: int = 256,
        normalize_to_neg_one_one: bool = True,
        canvas_width: int = 1024,
        canvas_height: int = 768,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.patch_size = patch_size
        self.normalize_to_neg_one_one = normalize_to_neg_one_one
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.canvas_width % self.patch_size != 0 or self.canvas_height % self.patch_size != 0:
            raise ValueError(
                "canvas dimensions must be divisible by patch_size for exhaustive non-overlapping grid"
            )

        self.samples = self._discover_samples(self.split_dir)
        if not self.samples:
            raise RuntimeError(f"No valid samples found in: {self.split_dir}")

        # Row-major coordinate order keeps indexing deterministic across runs.
        self.grid_coords = [
            (top, left)
            for top in range(0, self.canvas_height, self.patch_size)
            for left in range(0, self.canvas_width, self.patch_size)
        ]

        self.instructions: List[ExhaustivePatchInstruction] = []
        for sample_index, sample in enumerate(self.samples):
            for noisy_index, _ in enumerate(sample.noisy_paths):
                for top, left in self.grid_coords:
                    self.instructions.append(
                        ExhaustivePatchInstruction(
                            sample_index=sample_index,
                            noisy_index=noisy_index,
                            top=top,
                            left=left,
                        )
                    )

        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def _is_image_file(path: Path) -> bool:
        return path.suffix.lower() in IMG_EXTENSIONS

    def _discover_samples(self, split_dir: Path) -> List[SampleEntry]:
        samples: List[SampleEntry] = []
        clean_files = [
            p for p in sorted(split_dir.iterdir()) if p.is_file() and self._is_image_file(p)
        ]

        for clean_path in clean_files:
            stem = clean_path.stem
            noisy_dir = split_dir / stem
            if not noisy_dir.is_dir():
                continue

            noisy_paths = [
                p for p in sorted(noisy_dir.iterdir()) if p.is_file() and self._is_image_file(p)
            ]
            if not noisy_paths:
                continue

            samples.append(SampleEntry(name=stem, clean_path=clean_path, noisy_paths=noisy_paths))

        return samples

    def __len__(self) -> int:
        return len(self.instructions)

    def _load_grayscale_tensor(self, path: Path) -> Tensor:
        image = Image.open(path).convert("L")
        return self.to_tensor(image)

    def _crop_at(self, tensor: Tensor, top: int, left: int) -> Tensor:
        if tensor.shape[-2:] != (self.canvas_height, self.canvas_width):
            raise ValueError(
                f"Expected image shape (1, {self.canvas_height}, {self.canvas_width}), got {tuple(tensor.shape)}"
            )

        bottom = top + self.patch_size
        right = left + self.patch_size
        return tensor[:, top:bottom, left:right]

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        instruction = self.instructions[index]
        entry = self.samples[instruction.sample_index]

        clean = self._load_grayscale_tensor(entry.clean_path)
        noisy = self._load_grayscale_tensor(entry.noisy_paths[instruction.noisy_index])
        if clean.shape != noisy.shape:
            raise ValueError(
                f"Shape mismatch between clean {tuple(clean.shape)} and noisy {tuple(noisy.shape)}"
            )

        clean_crop = self._crop_at(clean, instruction.top, instruction.left)
        noisy_crop = self._crop_at(noisy, instruction.top, instruction.left)

        if self.normalize_to_neg_one_one:
            clean_crop = clean_crop * 2.0 - 1.0
            noisy_crop = noisy_crop * 2.0 - 1.0

        return {
            "clean": clean_crop,
            "condition": noisy_crop,
            "name": entry.name,
        }

class FullImageTempestDataset(Dataset):
    """Deterministic full-image dataset over all clean/noisy pairings."""

    def __init__(
        self,
        split_dir: str | Path,
        normalize_to_neg_one_one: bool = True,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.normalize_to_neg_one_one = normalize_to_neg_one_one

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")

        self.samples = self._discover_samples(self.split_dir)
        if not self.samples:
            raise RuntimeError(f"No valid samples found in: {self.split_dir}")

        self.instructions: List[FullImageInstruction] = []
        for sample_index, sample in enumerate(self.samples):
            for noisy_index, _ in enumerate(sample.noisy_paths):
                self.instructions.append(
                    FullImageInstruction(sample_index=sample_index, noisy_index=noisy_index)
                )

        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def _is_image_file(path: Path) -> bool:
        return path.suffix.lower() in IMG_EXTENSIONS

    def _discover_samples(self, split_dir: Path) -> List[SampleEntry]:
        samples: List[SampleEntry] = []
        clean_files = [
            p for p in sorted(split_dir.iterdir()) if p.is_file() and self._is_image_file(p)
        ]

        for clean_path in clean_files:
            stem = clean_path.stem
            noisy_dir = split_dir / stem
            if not noisy_dir.is_dir():
                continue

            noisy_paths = [
                p for p in sorted(noisy_dir.iterdir()) if p.is_file() and self._is_image_file(p)
            ]
            if not noisy_paths:
                continue

            samples.append(SampleEntry(name=stem, clean_path=clean_path, noisy_paths=noisy_paths))

        return samples

    def __len__(self) -> int:
        return len(self.instructions)

    def _load_grayscale_tensor(self, path: Path) -> Tensor:
        image = Image.open(path).convert("L")
        return self.to_tensor(image)

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        instruction = self.instructions[index]
        entry = self.samples[instruction.sample_index]

        clean = self._load_grayscale_tensor(entry.clean_path)
        noisy = self._load_grayscale_tensor(entry.noisy_paths[instruction.noisy_index])
        if clean.shape != noisy.shape:
            raise ValueError(
                f"Shape mismatch between clean {tuple(clean.shape)} and noisy {tuple(noisy.shape)}"
            )

        if self.normalize_to_neg_one_one:
            clean = clean * 2.0 - 1.0
            noisy = noisy * 2.0 - 1.0

        return {
            "clean": clean,
            "condition": noisy,
            "name": entry.name,
        }


