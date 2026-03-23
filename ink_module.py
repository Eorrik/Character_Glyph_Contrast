from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.draw import ellipse
from skimage.measure import approximate_polygon, find_contours, subdivide_polygon


@dataclass
class InkComparison:
    edt_expert: np.ndarray
    edt_user: np.ndarray
    diff_map: np.ndarray
    heatmap_rgb: np.ndarray
    contour_expert_mask: np.ndarray
    contour_user_mask: np.ndarray
    contour_overlay_rgb: np.ndarray
    contour_expert_points: np.ndarray
    contour_user_points: np.ndarray
    contour_expert_deviation: np.ndarray
    contour_user_deviation: np.ndarray
    heatmap_max_value: float


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


def ensure_closed_contour(contour: np.ndarray) -> np.ndarray:
    if contour.shape[0] == 0:
        return contour
    if np.allclose(contour[0], contour[-1]):
        return contour
    return np.vstack([contour, contour[0]])


def ensure_closed_series(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32).reshape(-1)
    if values.size == 0:
        return values
    if np.isclose(values[0], values[-1]):
        return values
    return np.concatenate([values, values[:1]])


def contour_area(contour: np.ndarray) -> float:
    if contour.shape[0] < 3:
        return 0.0
    points = ensure_closed_contour(contour)
    x = points[:, 1]
    y = points[:, 0]
    return float(0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])))


def build_fallback_contour(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    rr, cc = ellipse(
        h // 2,
        w // 2,
        max(2, h // 8),
        max(2, w // 8),
        shape=(h, w),
    )
    mask[rr, cc] = 1
    contours = find_contours(mask, level=0.5)
    if contours:
        return ensure_closed_contour(contours[0])
    return np.array([[h // 2, w // 2]], dtype=np.float32)


def extract_outer_contour(binary_image: np.ndarray) -> np.ndarray:
    contours = find_contours(binary_image.astype(np.uint8), level=0.5)
    if not contours:
        return build_fallback_contour(binary_image.shape)

    outer = max(contours, key=contour_area)
    outer = ensure_closed_contour(outer)
    tolerance = max(binary_image.shape) * 0.004
    simplified = approximate_polygon(outer, tolerance=tolerance)
    if simplified.shape[0] >= 4:
        refined = simplified
    else:
        refined = outer
    refined = ensure_closed_contour(refined)
    refined = subdivide_polygon(refined, degree=2, preserve_ends=False)
    refined = subdivide_polygon(refined, degree=2, preserve_ends=False)
    refined[:, 0] = np.clip(refined[:, 0], 0, binary_image.shape[0] - 1)
    refined[:, 1] = np.clip(refined[:, 1], 0, binary_image.shape[1] - 1)
    return ensure_closed_contour(refined.astype(np.float32))


def rasterize_contour(contour: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    contour_mask = np.zeros(shape, dtype=np.uint8)
    points_xy = np.column_stack([contour[:, 1], contour[:, 0]])
    points_xy = np.round(points_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(contour_mask, [points_xy], isClosed=True, color=1, thickness=1)
    return contour_mask


def resample_closed_contour(contour: np.ndarray, sample_count: int = 720) -> np.ndarray:
    contour_closed = ensure_closed_contour(contour.astype(np.float32))
    if len(contour_closed) < 3:
        return contour_closed

    segment_vectors = contour_closed[1:] - contour_closed[:-1]
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    perimeter = float(cumulative[-1])
    if perimeter <= 1e-6:
        return contour_closed

    targets = np.linspace(0.0, perimeter, sample_count, endpoint=False)
    sampled = np.zeros((sample_count, 2), dtype=np.float32)
    segment_index = 0

    for idx, t in enumerate(targets):
        while segment_index + 1 < len(cumulative) and cumulative[segment_index + 1] < t:
            segment_index += 1
        start = contour_closed[segment_index]
        end = contour_closed[segment_index + 1]
        length = segment_lengths[segment_index]
        if length <= 1e-6:
            sampled[idx] = start
        else:
            ratio = (t - cumulative[segment_index]) / length
            sampled[idx] = start + (end - start) * ratio

    return ensure_closed_contour(sampled)


def align_contours_cyclic(expert_contour: np.ndarray, user_contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(expert_contour), len(user_contour)) - 1
    if n <= 2:
        return expert_contour, user_contour

    expert = expert_contour[:n]
    user = user_contour[:n]
    coarse_step = max(1, n // 180)
    best_shift = 0
    best_score = float("inf")

    for shift in range(0, n, coarse_step):
        rolled = np.roll(expert, shift, axis=0)
        diff = user - rolled
        score = float(np.mean(np.sum(diff * diff, axis=1)))
        if score < best_score:
            best_score = score
            best_shift = shift

    refine_radius = coarse_step * 2
    for delta in range(-refine_radius, refine_radius + 1):
        shift = (best_shift + delta) % n
        rolled = np.roll(expert, shift, axis=0)
        diff = user - rolled
        score = float(np.mean(np.sum(diff * diff, axis=1)))
        if score < best_score:
            best_score = score
            best_shift = shift

    aligned_expert = np.roll(expert, best_shift, axis=0)
    aligned_user = user
    return ensure_closed_contour(aligned_expert), ensure_closed_contour(aligned_user)


def compute_ordered_contour_deviation(expert_contour: np.ndarray, user_contour: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expert_sampled = resample_closed_contour(expert_contour)
    user_sampled = resample_closed_contour(user_contour, sample_count=max(8, len(expert_sampled) - 1))
    aligned_expert, aligned_user = align_contours_cyclic(expert_sampled, user_sampled)

    n = min(len(aligned_expert), len(aligned_user)) - 1
    aligned_expert = aligned_expert[:n]
    aligned_user = aligned_user[:n]
    deviation = np.linalg.norm(aligned_user - aligned_expert, axis=1).astype(np.float32)
    return (
        ensure_closed_contour(aligned_expert.astype(np.float32)),
        ensure_closed_contour(aligned_user.astype(np.float32)),
        ensure_closed_series(deviation),
        ensure_closed_series(deviation),
    )


def create_contour_overlay(expert_contour_mask: np.ndarray, user_contour_mask: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*expert_contour_mask.shape, 3), dtype=np.uint8)
    overlay[..., 1] = expert_contour_mask * 255
    overlay[..., 0] = user_contour_mask * 255
    overlap = (expert_contour_mask > 0) & (user_contour_mask > 0)
    overlay[overlap] = np.array([255, 255, 0], dtype=np.uint8)
    return overlay


def compare_ink(expert_binary: np.ndarray, user_binary: np.ndarray) -> InkComparison:
    expert_contour = extract_outer_contour(expert_binary > 0)
    user_contour = extract_outer_contour(user_binary > 0)
    expert_contour, user_contour, expert_deviation, user_deviation = compute_ordered_contour_deviation(expert_contour, user_contour)
    expert_contour_mask = rasterize_contour(expert_contour, expert_binary.shape)
    user_contour_mask = rasterize_contour(user_contour, user_binary.shape)

    edt_expert = distance_transform_edt(expert_contour_mask == 0)
    edt_user = distance_transform_edt(user_contour_mask == 0)
    raw_diff_map = np.abs(edt_user - edt_expert)

    focus = (expert_binary > 0) | (user_binary > 0)
    focus_band = distance_transform_edt(~focus) <= 56
    diff_map = raw_diff_map * focus_band.astype(np.float32)
    active_values = diff_map[focus_band]
    if active_values.size == 0:
        heatmap_max_value = 1.0
    else:
        heatmap_max_value = float(np.percentile(active_values, 97))
        if heatmap_max_value < 1e-6:
            heatmap_max_value = 1.0
    heatmap_rgb = colorize_diff_map(diff_map)
    contour_overlay_rgb = create_contour_overlay(expert_contour_mask, user_contour_mask)
    return InkComparison(
        edt_expert=edt_expert,
        edt_user=edt_user,
        diff_map=diff_map,
        heatmap_rgb=heatmap_rgb,
        contour_expert_mask=expert_contour_mask,
        contour_user_mask=user_contour_mask,
        contour_overlay_rgb=contour_overlay_rgb,
        contour_expert_points=expert_contour,
        contour_user_points=user_contour,
        contour_expert_deviation=expert_deviation,
        contour_user_deviation=user_deviation,
        heatmap_max_value=heatmap_max_value,
    )
