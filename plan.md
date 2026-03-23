# 执行计划

## 1. 项目目标
- 构建一个可交互的 Python Web 应用，基于 HTML 展示 Expert 与 User 字形对比结果。
- 输入固定为 `sample/expert.jpg` 与 `sample/user.jpg`。
- 主入口为 `app.py`，统一调度全部模块并启动页面。
- 输出包括：交互式 HTML 页面，以及过程产物 `skeleton_overlay.png`、`heatmap_diff.png`。

## 2. 模块与职责

### `preprocess.py`
- 读取输入图片。
- 执行灰度化、二值化、去噪。
- 完成裁剪、缩放、居中、统一画布尺寸。
- 输出归一化后的二值字形图。

### `skeleton_module.py`
- 提取 Expert/User 的骨架。
- 计算骨架差异（Hausdorff、Chamfer）。
- 生成骨架叠加图供 Web 页面与过程图复用。

### `ink_module.py`
- 计算 Expert/User 的 EDT。
- 计算墨迹差异场 `|edt_user - edt_expert|`。
- 生成热力图数据与过程图。

### `metrics.py`
- 汇总结构差异、墨迹差异、重叠率等指标。
- 输出页面展示所需的统计结果。

### `visualization.py`
- 使用 Plotly + Dash 构建交互式 HTML 界面。
- 支持结构骨架视图、热力图视图、综合叠加视图。
- 提供透明度调节、缩放、平移、hover 数值查看。

### `app.py`
- 固定加载 `sample/expert.jpg` 与 `sample/user.jpg`。
- 依次调用预处理、骨架、墨迹、指标、可视化模块。
- 保存过程图并启动 Dash 应用。

## 3. 目录结构
```text
Character_Glyph_Contrast/
├── app.py
├── preprocess.py
├── skeleton_module.py
├── ink_module.py
├── metrics.py
├── visualization.py
├── requirements.txt
├── project.md
├── plan.md
├── sample/
│   ├── expert.jpg
│   └── user.jpg
└── outputs/
    ├── skeleton_overlay.png
    ├── heatmap_diff.png
    └── glyph_contrast_report.html
```

## 4. 执行流程
1. `app.py` 读取 `sample/expert.jpg` 与 `sample/user.jpg`。
2. `preprocess.py` 输出统一坐标系下的二值图。
3. `skeleton_module.py` 提取骨架并生成骨架叠加图。
4. `ink_module.py` 计算 EDT 并生成差异热力数据。
5. `metrics.py` 生成展示指标。
6. `visualization.py` 生成 HTML 交互页面并由 Dash 提供服务。

## 5. 必要依赖
- `opencv-python-headless`
- `numpy`
- `scipy`
- `scikit-image`
- `plotly`
- `dash`

## 6. 交付结果
- 运行 `app.py` 后可启动交互式字形对比页面。
- 页面内可切换骨架、热力图、综合视图。
- 过程图保存在 `outputs/`，用于调试与留档。
