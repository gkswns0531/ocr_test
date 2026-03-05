"""Image preprocessing: GPU/CPU decode+resize for layout, region cropping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

if TYPE_CHECKING:
    pass


def gpu_decode_and_resize(
    image_paths: list[str],
    target_size: tuple[int, int],
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Decode images from disk and resize on GPU.

    JPEG: nvJPEG hardware decode via torchvision (GPU-accelerated).
    PNG/other: cv2 CPU decode + GPU F.interpolate.

    Returns:
        (batch_tensor [B,3,H,W] normalized, original_sizes [(w,h), ...])
    """
    import torchvision.io

    target_h, target_w = target_size
    original_sizes: list[tuple[int, int]] = []
    tensors: list[torch.Tensor] = []

    for path in image_paths:
        if path.lower().endswith((".jpg", ".jpeg")):
            with open(path, "rb") as f:
                data = f.read()
            jt = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            decoded = torchvision.io.decode_jpeg(jt, device=device)  # (3,H,W) uint8
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            decoded = torch.from_numpy(img_rgb).permute(2, 0, 1).to(device)

        _, h, w = decoded.shape
        original_sizes.append((w, h))
        resized = F.interpolate(
            decoded.unsqueeze(0).to(dtype=dtype),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        tensors.append(resized.squeeze(0))

    pixel_values = torch.stack(tensors).div_(255.0)
    return pixel_values, original_sizes


def cpu_decode_and_resize(
    images: list[Image.Image],
    target_size: tuple[int, int],
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """CPU fallback: PIL/cv2 decode + resize, transfer to GPU.

    Returns:
        (batch_tensor [B,3,H,W] normalized, original_sizes [(w,h), ...])
    """
    target_h, target_w = target_size
    original_sizes: list[tuple[int, int]] = []
    arrays: list[np.ndarray] = []

    for img in images:
        rgb = img if img.mode == "RGB" else img.convert("RGB")
        original_sizes.append(rgb.size)  # (w, h)
        arr = np.asarray(rgb)
        if arr.shape[0] != target_h or arr.shape[1] != target_w:
            arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        arrays.append(arr)

    batch = np.stack(arrays)
    pixel_values = (
        torch.from_numpy(batch)
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device, dtype=dtype, non_blocking=True)
        .div_(255.0)
    )
    return pixel_values, original_sizes


def crop_regions(
    image: Image.Image,
    detections: list,
    image_path: str | None = None,
) -> list[Image.Image]:
    """Crop detected regions from the original image.

    Uses cv2.imread once (if path available) to avoid repeated PIL→numpy conversion.

    Args:
        image: Original PIL image.
        detections: List of Detection objects with .bbox (x1,y1,x2,y2 pixel coords).
        image_path: Optional file path for fast cv2 read.

    Returns:
        List of cropped PIL Images (same order as detections).
    """
    if image_path:
        arr = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    else:
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        arr = np.asarray(rgb)

    crops: list[Image.Image] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        region = arr[y1:y2, x1:x2]
        if region.size == 0:
            crops.append(None)
        else:
            crops.append(Image.fromarray(region.copy()))
    return crops
