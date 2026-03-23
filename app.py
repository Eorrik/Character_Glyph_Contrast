from __future__ import annotations

from pathlib import Path

import cv2

from ink_module import compare_ink
from metrics import summarize_metrics
from preprocess import preprocess_image
from skeleton_module import compare_skeletons
from visualization import build_dash_app, export_static_html


ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample"
OUTPUT_DIR = ROOT / "outputs"
EXPERT_IMAGE = SAMPLE_DIR / "expert.jpg"
USER_IMAGE = SAMPLE_DIR / "user.jpg"


def save_rgb_image(path: Path, rgb_image):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr_image)


def build_pipeline():
    expert = preprocess_image(EXPERT_IMAGE)
    user = preprocess_image(USER_IMAGE)
    skeleton_comparison = compare_skeletons(expert.normalized_binary, user.normalized_binary)
    ink_comparison = compare_ink(expert.normalized_binary, user.normalized_binary)
    metrics = summarize_metrics(
        expert.normalized_binary,
        user.normalized_binary,
        skeleton_comparison,
        ink_comparison,
    )

    save_rgb_image(OUTPUT_DIR / "skeleton_overlay.png", skeleton_comparison.overlay_rgb)
    save_rgb_image(OUTPUT_DIR / "heatmap_diff.png", ink_comparison.heatmap_rgb)
    export_static_html(OUTPUT_DIR / "glyph_contrast_report.html", expert.normalized_binary, skeleton_comparison, ink_comparison)

    return expert, user, skeleton_comparison, ink_comparison, metrics


expert_result, user_result, skeleton_result, ink_result, comparison_metrics = build_pipeline()
app = build_dash_app(expert_result.normalized_binary, skeleton_result, ink_result, comparison_metrics)
server = app.server


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
