from __future__ import annotations

from pathlib import Path

import dash
from dash import Dash, Input, Output, dcc, html
import numpy as np
import plotly.graph_objects as go

from ink_module import InkComparison
from metrics import ComparisonMetrics
from skeleton_module import SkeletonComparison


VIEW_OPTIONS = [
    {"label": "结构骨架叠加", "value": "skeleton"},
    {"label": "墨迹差异热力图", "value": "heatmap"},
    {"label": "综合对比", "value": "comparison"},
]


def binary_to_grayscale_image(binary_image: np.ndarray) -> np.ndarray:
    canvas = np.where(binary_image > 0, 20, 245).astype(np.uint8)
    return np.stack([canvas, canvas, canvas], axis=-1)


def build_metrics_panel(metrics: ComparisonMetrics) -> html.Div:
    items = [
        ("Hausdorff 距离", f"{metrics.hausdorff_distance:.2f}"),
        ("Chamfer 距离", f"{metrics.chamfer_distance:.2f}"),
        ("平均 EDT 差异", f"{metrics.mean_edt_difference:.2f}"),
        ("最大 EDT 差异", f"{metrics.max_edt_difference:.2f}"),
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
    alpha: float,
    base_rgb: np.ndarray,
    skeleton_comparison: SkeletonComparison,
    ink_comparison: InkComparison,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Image(z=base_rgb, name="base"))

    if view_mode == "skeleton":
        figure.add_trace(go.Image(z=skeleton_comparison.overlay_rgb, name="skeleton overlay", opacity=alpha))
    elif view_mode == "heatmap":
        figure.add_trace(
            go.Heatmap(
                z=ink_comparison.diff_map,
                colorscale="RdYlGn_r",
                opacity=alpha,
                colorbar={"title": "EDT 差异"},
                hovertemplate="x=%{x}<br>y=%{y}<br>diff=%{z:.2f}<extra></extra>",
            )
        )
    else:
        figure.add_trace(go.Image(z=skeleton_comparison.overlay_rgb, name="skeleton overlay", opacity=max(alpha * 0.9, 0.2)))
        figure.add_trace(
            go.Heatmap(
                z=ink_comparison.diff_map,
                colorscale="RdYlGn_r",
                opacity=min(alpha, 0.75),
                colorbar={"title": "EDT 差异"},
                hovertemplate="x=%{x}<br>y=%{y}<br>diff=%{z:.2f}<extra></extra>",
            )
        )

    figure.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        dragmode="pan",
        title="字形结构与墨迹差异交互查看",
        legend={"orientation": "h"},
    )
    figure.update_xaxes(showgrid=False, visible=False)
    figure.update_yaxes(showgrid=False, visible=False, scaleanchor="x", autorange="reversed")
    return figure


def build_dash_app(
    expert_binary: np.ndarray,
    skeleton_comparison: SkeletonComparison,
    ink_comparison: InkComparison,
    metrics: ComparisonMetrics,
) -> Dash:
    app = dash.Dash(__name__)
    base_rgb = binary_to_grayscale_image(expert_binary)

    app.layout = html.Div(
        [
            html.H1("Character Glyph Contrast App"),
            html.P("基于 HTML 的交互式字形分析界面：支持结构骨架、墨迹热力和综合对比查看。"),
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
                            html.Label("图层透明度"),
                            dcc.Slider(id="alpha-slider", min=0.2, max=1.0, step=0.05, value=0.7),
                        ],
                        style={"width": "24%", "minWidth": "260px"},
                    ),
                    html.Div(build_metrics_panel(metrics), style={"width": "32%", "minWidth": "280px"}),
                ],
                style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "18px"},
            ),
            dcc.Graph(
                id="glyph-graph",
                figure=create_figure("comparison", 0.7, base_rgb, skeleton_comparison, ink_comparison),
                style={"height": "75vh"},
                config={"displaylogo": False, "scrollZoom": True},
            ),
        ],
        style={"padding": "20px 24px", "fontFamily": "Arial, sans-serif"},
    )

    @app.callback(
        Output("glyph-graph", "figure"),
        Input("view-mode", "value"),
        Input("alpha-slider", "value"),
    )
    def update_figure(view_mode: str, alpha: float) -> go.Figure:
        return create_figure(view_mode, alpha, base_rgb, skeleton_comparison, ink_comparison)

    return app


def export_static_html(
    output_path: str | Path,
    expert_binary: np.ndarray,
    skeleton_comparison: SkeletonComparison,
    ink_comparison: InkComparison,
) -> Path:
    base_rgb = binary_to_grayscale_image(expert_binary)
    figure = create_figure("comparison", 0.7, base_rgb, skeleton_comparison, ink_comparison)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output), include_plotlyjs="cdn")
    return output
