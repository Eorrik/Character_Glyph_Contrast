from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff
from skimage.morphology import medial_axis, skeletonize


@dataclass
class SkeletonResult:
    skeleton: np.ndarray
    medial_axis_skeleton: np.ndarray
    medial_distance: np.ndarray
    points: np.ndarray


@dataclass
class SkeletonComparison:
    expert: SkeletonResult
    user: SkeletonResult
    overlay_rgb: np.ndarray
    hausdorff_distance: float
    chamfer_distance: float


def extract_skeleton(binary_image: np.ndarray) -> SkeletonResult:
    binary_bool = binary_image.astype(bool)
    skeleton = skeletonize(binary_bool)
    medial_skeleton, medial_distance = medial_axis(binary_bool, return_distance=True)
    points = np.column_stack(np.nonzero(skeleton))
    return SkeletonResult(
        skeleton=skeleton.astype(np.uint8),
        medial_axis_skeleton=medial_skeleton.astype(np.uint8),
        medial_distance=medial_distance,
        points=points,
    )


def compute_hausdorff_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if len(points_a) == 0 or len(points_b) == 0:
        return float("inf")
    forward = directed_hausdorff(points_a, points_b)[0]
    backward = directed_hausdorff(points_b, points_a)[0]
    return float(max(forward, backward))


def compute_chamfer_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if len(points_a) == 0 or len(points_b) == 0:
        return float("inf")
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    distances_a, _ = tree_b.query(points_a, k=1)
    distances_b, _ = tree_a.query(points_b, k=1)
    return float((distances_a.mean() + distances_b.mean()) / 2.0)


def create_skeleton_overlay(expert_skeleton: np.ndarray, user_skeleton: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*expert_skeleton.shape, 3), dtype=np.uint8)
    overlay[..., 1] = expert_skeleton * 255
    overlay[..., 0] = user_skeleton * 255

    overlap = (expert_skeleton > 0) & (user_skeleton > 0)
    overlay[overlap] = np.array([255, 255, 0], dtype=np.uint8)
    return overlay


def compare_skeletons(expert_binary: np.ndarray, user_binary: np.ndarray) -> SkeletonComparison:
    expert = extract_skeleton(expert_binary)
    user = extract_skeleton(user_binary)
    overlay = create_skeleton_overlay(expert.skeleton, user.skeleton)
    hausdorff = compute_hausdorff_distance(expert.points, user.points)
    chamfer = compute_chamfer_distance(expert.points, user.points)
    return SkeletonComparison(
        expert=expert,
        user=user,
        overlay_rgb=overlay,
        hausdorff_distance=hausdorff,
        chamfer_distance=chamfer,
    )
