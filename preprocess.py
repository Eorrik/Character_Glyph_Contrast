from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    original_gray: np.ndarray
    binary: np.ndarray
    normalized_binary: np.ndarray


def load_grayscale_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")
    return image


def ensure_white_foreground(binary_image: np.ndarray) -> np.ndarray:
    white_pixels = int(np.count_nonzero(binary_image))
    black_pixels = int(binary_image.size - white_pixels)
    if white_pixels > black_pixels:
        binary_image = cv2.bitwise_not(binary_image)
    return binary_image


def crop_to_foreground(binary_image: np.ndarray, padding: int = 12) -> np.ndarray:
    points = cv2.findNonZero(binary_image)
    if points is None:
        raise ValueError("No foreground pixels found after thresholding.")

    x, y, w, h = cv2.boundingRect(points)
    x0 = max(x - padding, 0)
    y0 = max(y - padding, 0)
    x1 = min(x + w + padding, binary_image.shape[1])
    y1 = min(y + h + padding, binary_image.shape[0])
    return binary_image[y0:y1, x0:x1]


def normalize_binary_image(binary_image: np.ndarray, canvas_size: int = 512, margin: int = 36) -> np.ndarray:
    cropped = crop_to_foreground(binary_image)
    height, width = cropped.shape
    target = canvas_size - 2 * margin
    scale = min(target / max(height, 1), target / max(width, 1))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    y_offset = (canvas_size - new_height) // 2
    x_offset = (canvas_size - new_width) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
    return (canvas > 0).astype(np.uint8)


def preprocess_image(image_path: str | Path, canvas_size: int = 512) -> PreprocessResult:
    gray = load_grayscale_image(image_path)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = ensure_white_foreground(binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    normalized = normalize_binary_image(binary, canvas_size=canvas_size)
    return PreprocessResult(original_gray=gray, binary=(binary > 0).astype(np.uint8), normalized_binary=normalized)
