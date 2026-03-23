# Character Glyph Contrast

一个用于比较 `expert` 与 `user` 字形差异的可视化项目，支持：

- 预处理与二值化归一
- 外轮廓（墨迹）差异对比
- 骨架提取与骨架差异对比
- Dash 交互式网页查看（透明度、降噪阈值可调）
- 导出静态 HTML 报告与中间图像


## 功能模块

- `preprocess.py`
  - 图像读取、灰度化、阈值分割、前景统一、裁剪、归一化到固定画布
- `ink_module.py`
  - 墨迹外轮廓提取与平滑（`find_contours` / `approximate_polygon` / `subdivide_polygon`）
  - 轮廓有序对齐后计算偏差，避免“只要靠近就判定低误差”的误配
- `skeleton_module.py`
  - 骨架提取与可调分支降噪
  - 计算 Hausdorff / Chamfer 等结构距离
- `metrics.py`
  - 汇总对比指标（结构距离、轮廓差异、重叠率、骨架像素数）
- `visualization.py`
  - Dash 界面与交互回调
  - 支持专家/用户底图独立透明度、骨架降噪阈值滑条
- `app.py`
  - 主流程入口：串联预处理、对比、导出、启动 Web 服务


## 快速运行

### 1) 准备环境（Windows PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2) 准备输入图像

将对比图像放在：

- `sample/expert.jpg`
- `sample/user.jpg`

### 3) 启动项目

```powershell
.\.venv\Scripts\python.exe .\app.py
```

浏览器打开：

- `http://127.0.0.1:8050`


## 交互说明

- 视图模式
  - 骨架偏差视图
  - 轮廓偏差视图
  - 综合偏差视图
- 专家图层透明度 / 用户图层透明度
  - 分别控制两张二值灰度底图的叠加强度
- 骨架降噪阈值
  - 当前范围：`15 ~ 25`，默认 `20`
  - 数值越高，短分支抑制越强


## 输出结果

运行后会在 `outputs/` 下生成：

- `skeleton_overlay.png`
- `heatmap_diff.png`
- `glyph_contrast_report.html`


## 常见问题

- PowerShell 脚本被禁用
  - 本项目可直接使用 `.\.venv\Scripts\python.exe` 运行，不依赖激活脚本
- 网页未更新
  - 重启服务后使用 `Ctrl + F5` 强刷，或打开 `http://127.0.0.1:8050/?v=1`
