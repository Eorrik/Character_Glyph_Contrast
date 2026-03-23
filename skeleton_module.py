from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve
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


def extract_skeleton(binary_image: np.ndarray, denoise_threshold: float = 2.0) -> SkeletonResult:
    binary_bool = binary_image.astype(bool)
    skeleton = skeletonize(binary_bool)
    skeleton = prune_short_branches(skeleton, min_branch_length=denoise_threshold)
    medial_skeleton, medial_distance = medial_axis(binary_bool, return_distance=True)
    points = np.column_stack(np.nonzero(skeleton))
    return SkeletonResult(
        skeleton=skeleton.astype(np.uint8),
        medial_axis_skeleton=medial_skeleton.astype(np.uint8),
        medial_distance=medial_distance,
        points=points,
    )


def compute_neighbor_count(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )
    return convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)


def get_8_neighbors(row: int, col: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    neighbors: list[tuple[int, int]] = []
    h, w = shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = row + dr
            cc = col + dc
            if 0 <= rr < h and 0 <= cc < w:
                neighbors.append((rr, cc))
    return neighbors


def trace_branch(start: tuple[int, int], skeleton: np.ndarray, neighbor_count: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    path = [start]
    previous = None
    current = start

    while True:
        candidates = []
        for rr, cc in get_8_neighbors(current[0], current[1], skeleton.shape):
            if not skeleton[rr, cc]:
                continue
            if previous is not None and (rr, cc) == previous:
                continue
            candidates.append((rr, cc))

        if len(candidates) != 1:
            break

        nxt = candidates[0]
        path.append(nxt)
        previous = current
        current = nxt
        if neighbor_count[current[0], current[1]] != 2:
            break

    return path, current


def prune_short_branches(skeleton: np.ndarray, min_branch_length: float) -> np.ndarray:
    skeleton_bool = skeleton.astype(bool)
    if not np.any(skeleton_bool):
        return skeleton_bool

    threshold = max(0.0, float(min_branch_length))
    if threshold < 1e-6:
        return skeleton_bool

    max_passes = 32
    for _ in range(max_passes):
        neighbor_count = compute_neighbor_count(skeleton_bool)
        endpoints = np.argwhere(skeleton_bool & (neighbor_count == 1))
        if len(endpoints) == 0:
            break

        remove_mask = np.zeros_like(skeleton_bool)
        removed = False
        for row, col in endpoints:
            if not skeleton_bool[row, col]:
                continue
            path, end_node = trace_branch((int(row), int(col)), skeleton_bool, neighbor_count)
            end_degree = int(neighbor_count[end_node[0], end_node[1]])
            branch_length = len(path)
            if end_degree >= 3 and branch_length <= threshold:
                for rr, cc in path:
                    remove_mask[rr, cc] = True
                removed = True

        if not removed:
            break
        skeleton_bool[remove_mask] = False

    return skeleton_bool


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


def compare_skeletons(expert_binary: np.ndarray, user_binary: np.ndarray, denoise_threshold: float = 2.0) -> SkeletonComparison:
    expert = extract_skeleton(expert_binary, denoise_threshold=denoise_threshold)
    user = extract_skeleton(user_binary, denoise_threshold=denoise_threshold)
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
