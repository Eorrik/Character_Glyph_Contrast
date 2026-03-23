from __future__ import annotations

from dataclasses import dataclass

import cv2
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
    heat_uint8 = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(255 - heat_uint8, cv2.COLORMAP_RDYLGN)


def compare_ink(expert_binary: np.ndarray, user_binary: np.ndarray) -> InkComparison:
    edt_expert = distance_transform_edt(expert_binary > 0)
    edt_user = distance_transform_edt(user_binary > 0)
    diff_map = np.abs(edt_user - edt_expert)
    heatmap_rgb = cv2.cvtColor(colorize_diff_map(diff_map), cv2.COLOR_BGR2RGB)
    return InkComparison(
        edt_expert=edt_expert,
        edt_user=edt_user,
        diff_map=diff_map,
        heatmap_rgb=heatmap_rgb,
    )
