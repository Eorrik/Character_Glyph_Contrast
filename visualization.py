from __future__ import annotations

from pathlib import Path

import dash
from dash import Dash, Input, Output, dcc, html
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import distance_transform_edt

from ink_module import InkComparison
from metrics import ComparisonMetrics, summarize_metrics
from skeleton_module import SkeletonComparison, compare_skeletons


VIEW_OPTIONS = [
    {"label": "骨架偏差视图", "value": "skeleton"},
    {"label": "轮廓偏差视图", "value": "heatmap"},
    {"label": "综合偏差视图", "value": "comparison"},
]


def binary_to_grayscale_image(binary_image: np.ndarray) -> np.ndarray:
    canvas = np.where(binary_image > 0, 20, 245).astype(np.uint8)
    return np.stack([canvas, canvas, canvas], axis=-1)


def build_metrics_panel(metrics: ComparisonMetrics) -> html.Div:
    items = [
        ("Hausdorff 距离", f"{metrics.hausdorff_distance:.2f}"),
        ("Chamfer 距离", f"{metrics.chamfer_distance:.2f}"),
        ("平均轮廓距离差异", f"{metrics.mean_edt_difference:.2f}"),
        ("最大轮廓距离差异", f"{metrics.max_edt_difference:.2f}"),
        ("像素重叠率", f"{metrics.overlap_ratio:.2%}"),
        ("Expert 骨架像素", str(metrics.expert_skeleton_pixels)),
        ("User 骨架像素", str(metrics.user_skeleton_pixels)),
    ]
    return html.Div(
        [
            html.H3("对比指标"),
            html.Ul([html.Li([html.Strong(f"{label}: "), value]) for label, value in items]),
        ],
        style={"padding": "12px 18px", "background": "#f7f7f7", "borderRadius": "10px"},
    )


def create_figure(
    view_mode: str,
    expert_alpha: float,
    user_alpha: float,
    expert_base_rgb: np.ndarray,
    user_base_rgb: np.ndarray,
    skeleton_comparison: SkeletonComparison,
    ink_comparison: InkComparison,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Image(z=expert_base_rgb, name="expert base", opacity=expert_alpha))
    figure.add_trace(go.Image(z=user_base_rgb, name="user base", opacity=user_alpha))
    contour_expert_distances = np.asarray(ink_comparison.contour_expert_deviation, dtype=np.float32).reshape(-1)
    contour_user_distances = np.asarray(ink_comparison.contour_user_deviation, dtype=np.float32).reshape(-1)
    contour_distance_all = np.concatenate([contour_expert_distances, contour_user_distances])
    if contour_distance_all.size == 0:
        contour_zmax = 0.8
    else:
        contour_zmax = max(float(np.percentile(contour_distance_all, 82)) * 0.72, 0.8)

    skeleton_distance_to_expert = distance_transform_edt(skeleton_comparison.expert.skeleton == 0)
    user_points = skeleton_comparison.user.points
    if len(user_points) > 0:
        skeleton_user_distances = skeleton_distance_to_expert[user_points[:, 0], user_points[:, 1]]
        skeleton_zmax = max(float(np.percentile(skeleton_user_distances, 97)), 1.0)
    else:
        skeleton_user_distances = np.array([], dtype=np.float32)
        skeleton_zmax = 1.0

    def deviation_to_rgb(distance: float) -> str:
        ratio = float(np.clip(distance / contour_zmax, 0.0, 1.0))
        if ratio <= 0.5:
            local = ratio / 0.5
            start = np.array([0, 176, 80], dtype=np.float32)
            end = np.array([255, 235, 132], dtype=np.float32)
        else:
            local = (ratio - 0.5) / 0.5
            start = np.array([255, 235, 132], dtype=np.float32)
            end = np.array([220, 53, 69], dtype=np.float32)
        color = start + (end - start) * local
        return f"rgb({int(color[0])},{int(color[1])},{int(color[2])})"

    def add_gradient_contour(points: np.ndarray, distances: np.ndarray, name: str, opacity: float, showlegend: bool) -> None:
        segment_count = min(len(points) - 1, len(distances) - 1)
        if segment_count <= 0:
            return
        for idx in range(segment_count):
            p0 = points[idx]
            p1 = points[idx + 1]
            segment_distance = 0.5 * (float(distances[idx]) + float(distances[idx + 1]))
            figure.add_trace(
                go.Scatter(
                    x=[p0[1], p1[1]],
                    y=[p0[0], p1[0]],
                    mode="lines",
                    line={"color": deviation_to_rgb(segment_distance), "width": 3},
                    opacity=opacity,
                    name=name if idx == 0 else f"{name} ",
                    showlegend=showlegend and idx == 0,
                    hovertemplate=f"{name} 偏差={segment_distance:.2f}<extra></extra>",
                )
            )

    def add_solid_contour(points: np.ndarray, name: str, opacity: float, color: str) -> None:
        if len(points) < 2:
            return
        figure.add_trace(
            go.Scatter(
                x=points[:, 1],
                y=points[:, 0],
                mode="lines",
                line={"color": color, "width": 3},
                opacity=opacity,
                name=name,
                hovertemplate=f"{name}<extra></extra>",
            )
        )

    def add_contour_deviation(opacity: float) -> None:
        add_solid_contour(
            ink_comparison.contour_expert_points,
            "expert contour deviation",
            opacity,
            "rgb(128,128,128)",
        )
        add_gradient_contour(
            ink_comparison.contour_user_points,
            contour_user_distances,
            "user contour deviation",
            opacity,
            True,
        )

    def add_skeleton_deviation(opacity: float) -> None:
        expert_points = skeleton_comparison.expert.points
        if len(expert_points) > 0:
            figure.add_trace(
                go.Scatter(
                    x=expert_points[:, 1],
                    y=expert_points[:, 0],
                    mode="markers",
                    marker={"size": 3, "color": "rgb(128,128,128)", "opacity": max(opacity, 0.35)},
                    name="expert skeleton",
                    hovertemplate="expert x=%{x}<br>y=%{y}<extra></extra>",
                )
            )
        if len(user_points) > 0:
            figure.add_trace(
                go.Scatter(
                    x=user_points[:, 1],
                    y=user_points[:, 0],
                    mode="markers",
                    marker={
                        "size": 4,
                        "color": skeleton_user_distances,
                        "colorscale": "RdYlGn_r",
                        "cmin": 0.0,
                        "cmax": skeleton_zmax,
                        "opacity": opacity,
                        "colorbar": {"title": "骨架偏差"},
                    },
                    name="user skeleton deviation",
                    hovertemplate="user x=%{x}<br>y=%{y}<br>偏差=%{marker.color:.2f}<extra></extra>",
                )
            )

    deviation_alpha = max(0.35, (expert_alpha + user_alpha) * 0.5)

    if view_mode == "skeleton":
        add_skeleton_deviation(deviation_alpha)
    elif view_mode == "heatmap":
        add_contour_deviation(max(deviation_alpha, 0.45))
    else:
        add_contour_deviation(max(deviation_alpha, 0.45))
        add_skeleton_deviation(max(deviation_alpha * 0.9, 0.35))

    figure.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        dragmode="pan",
        title="字形轮廓与骨架偏差交互查看",
        legend={"orientation": "h"},
    )
    figure.update_xaxes(showgrid=False, visible=False)
    figure.update_yaxes(showgrid=False, visible=False, scaleanchor="x", autorange="reversed")
    return figure


def build_dash_app(
    expert_binary: np.ndarray,
    user_binary: np.ndarray,
    skeleton_comparison: SkeletonComparison,
    ink_comparison: InkComparison,
    metrics: ComparisonMetrics,
) -> Dash:
    app = dash.Dash(__name__)
    expert_base_rgb = binary_to_grayscale_image(expert_binary)
    user_base_rgb = binary_to_grayscale_image(user_binary)

    app.layout = html.Div(
        [
            html.H1("Character Glyph Contrast App"),
            html.P("基于 HTML 的交互式字形分析界面：支持骨架偏差、轮廓偏差和综合偏差查看。"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("视图模式"),
                            dcc.RadioItems(
                                id="view-mode",
                                options=VIEW_OPTIONS,
                                value="comparison",
                                inline=False,
                            ),
                            html.Br(),
                            html.Label("专家图层透明度"),
                            dcc.Slider(id="expert-alpha-slider", min=0.0, max=1.0, step=0.05, value=0.85),
                            html.Br(),
                            html.Label("用户图层透明度"),
                            dcc.Slider(id="user-alpha-slider", min=0.0, max=1.0, step=0.05, value=0.65),
                            html.Br(),
                            html.Label("骨架降噪阈值"),
                            dcc.Slider(id="skeleton-denoise-slider", min=15.0, max=25.0, step=0.2, value=20.0),
                        ],
                        style={"width": "24%", "minWidth": "260px"},
                    ),
                    html.Div(id="metrics-panel", children=build_metrics_panel(metrics), style={"width": "32%", "minWidth": "280px"}),
                ],
                style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "18px"},
            ),
            dcc.Graph(
                id="glyph-graph",
                figure=create_figure("comparison", 0.85, 0.65, expert_base_rgb, user_base_rgb, skeleton_comparison, ink_comparison),
                style={"height": "75vh"},
                config={"displaylogo": False, "scrollZoom": True},
            ),
        ],
        style={"padding": "20px 24px", "fontFamily": "Arial, sans-serif"},
    )

    @app.callback(
        Output("glyph-graph", "figure"),
        Output("metrics-panel", "children"),
        Input("view-mode", "value"),
        Input("expert-alpha-slider", "value"),
        Input("user-alpha-slider", "value"),
        Input("skeleton-denoise-slider", "value"),
    )
    def update_figure(view_mode: str, expert_alpha: float, user_alpha: float, denoise_threshold: float):
        current_skeleton = compare_skeletons(
            expert_binary,
            user_binary,
            denoise_threshold=float(denoise_threshold),
        )
        current_metrics = summarize_metrics(
            expert_binary,
            user_binary,
            current_skeleton,
            ink_comparison,
        )
        figure = create_figure(
            view_mode,
            float(expert_alpha),
            float(user_alpha),
            expert_base_rgb,
            user_base_rgb,
            current_skeleton,
            ink_comparison,
        )
        return figure, build_metrics_panel(current_metrics)

    return app


def export_static_html(
    output_path: str | Path,
    expert_binary: np.ndarray,
    user_binary: np.ndarray,
    skeleton_comparison: SkeletonComparison,
    ink_comparison: InkComparison,
) -> Path:
    expert_base_rgb = binary_to_grayscale_image(expert_binary)
    user_base_rgb = binary_to_grayscale_image(user_binary)
    figure = create_figure(
        "comparison",
        0.85,
        0.65,
        expert_base_rgb,
        user_base_rgb,
        skeleton_comparison,
        ink_comparison,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output), include_plotlyjs="cdn")
    return output
