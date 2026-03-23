from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class InkComparison:
    edt_expert: np.ndarray
    edt_user: np.ndarray
    diff_map: np.ndarray
    heatmap_rgb: np.ndarray


def colorize_diff_map(diff_map: np.ndarray) -> np.ndarray:
    normalized = diff_map.astype(np.float32)
    if normalized.max() > 0:
        normalized /= normalized.max()

    low = np.array([0, 176, 80], dtype=np.float32)
    mid = np.array([255, 235, 132], dtype=np.float32)
    high = np.array([220, 53, 69], dtype=np.float32)

    rgb = np.zeros((*normalized.shape, 3), dtype=np.float32)
    lower_mask = normalized <= 0.5
    upper_mask = ~lower_mask

    lower_ratio = np.clip(normalized[lower_mask] / 0.5, 0.0, 1.0)[:, None]
    upper_ratio = np.clip((normalized[upper_mask] - 0.5) / 0.5, 0.0, 1.0)[:, None]

    rgb[lower_mask] = low + (mid - low) * lower_ratio
    rgb[upper_mask] = mid + (high - mid) * upper_ratio
    return np.clip(rgb, 0, 255).astype(np.uint8)


def compare_ink(expert_binary: np.ndarray, user_binary: np.ndarray) -> InkComparison:
    edt_expert = distance_transform_edt(expert_binary > 0)
    edt_user = distance_transform_edt(user_binary > 0)
    diff_map = np.abs(edt_user - edt_expert)
    heatmap_rgb = colorize_diff_map(diff_map)
    return InkComparison(
        edt_expert=edt_expert,
        edt_user=edt_user,
        diff_map=diff_map,
        heatmap_rgb=heatmap_rgb,
    )
