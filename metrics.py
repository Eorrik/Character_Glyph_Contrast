from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ink_module import InkComparison
from skeleton_module import SkeletonComparison


@dataclass
class ComparisonMetrics:
    hausdorff_distance: float
    chamfer_distance: float
    mean_edt_difference: float
    max_edt_difference: float
    overlap_ratio: float
    expert_skeleton_pixels: int
    user_skeleton_pixels: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_overlap_ratio(expert_binary: np.ndarray, user_binary: np.ndarray) -> float:
    expert_mask = expert_binary > 0
    user_mask = user_binary > 0
    union = np.count_nonzero(expert_mask | user_mask)
    if union == 0:
        return 0.0
    intersection = np.count_nonzero(expert_mask & user_mask)
    return float(intersection / union)


def summarize_metrics(
    expert_binary: np.ndarray,
    user_binary: np.ndarray,
    skeleton_comparison: SkeletonComparison,
    ink_comparison: InkComparison,
) -> ComparisonMetrics:
    return ComparisonMetrics(
        hausdorff_distance=skeleton_comparison.hausdorff_distance,
        chamfer_distance=skeleton_comparison.chamfer_distance,
        mean_edt_difference=float(np.mean(ink_comparison.diff_map)),
        max_edt_difference=float(np.max(ink_comparison.diff_map)),
        overlap_ratio=compute_overlap_ratio(expert_binary, user_binary),
        expert_skeleton_pixels=int(np.count_nonzero(skeleton_comparison.expert.skeleton)),
        user_skeleton_pixels=int(np.count_nonzero(skeleton_comparison.user.skeleton)),
    )
