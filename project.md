# 字形结果图像对比原型任务流程（最终版：Geometry Matching + Web 可视化）

## 一、问题定义（核心定位）

本项目仅基于**静态字形图像（offline handwriting）**进行对比分析，不涉及书写过程或时间序列数据。

因此问题被建模为：

> 字形对比 = 空间几何匹配问题（Geometry Matching）

通过对**结构骨架（Skeleton）**与**墨迹分布（EDT Distance Field）**的联合建模，实现笔画级差异定位与可解释分析。

---

## 二、输入与输出定义

### 输入

* Expert 字形图像（标准）
* User 字形图像（待评估）

### 输出

1. **骨架叠加图（Skeleton Overlay）**

   * 绿色：Expert
   * 红色：User
   * 表示结构偏移

2. **墨迹热力图（EDT Heatmap）**

   * 绿色：差异小
   * 红色：差异大
   * 表示粗细/形态差异

3. **交互式可视化网页（Plotly App）**

   * 支持图层切换（骨架 / 热力）
   * 支持透明度调节
   * 支持局部缩放与查看

---

## 三、系统模块划分

### 模块一：骨架提取与对比（Structure Module）

功能：

* 二值图 → skeletonize
* 提取骨架结构
* 计算骨架差异（Hausdorff / Chamfer）
* 生成骨架叠加图

输出：

* skeleton_expert
* skeleton_user
* skeleton_overlay

---

### 模块二：墨迹热力对比（Ink Module）

功能：

* 计算 EDT（欧氏距离变换）
* 构建墨迹厚度分布
* 计算差异场
* 生成热力图

输出：

* edt_expert
* edt_user
* heatmap_diff

---

### 模块三：可视化网页（Visualization Module）

基于 Plotly 构建交互式界面：

功能：

* 显示骨架叠加图
* 显示墨迹热力图
* 图层叠加（overlay）
* 透明度控制（alpha slider）
* 局部缩放（zoom / pan）

推荐实现：

* Plotly + Dash（Python Web App）

---

### 主程序：App（集成模块）

功能：

* 调用模块一（Skeleton）
* 调用模块二（EDT）
* 调用模块三（Plotly可视化）

流程：

```
输入图像
  ↓
Module 1（Skeleton）
  ↓
Module 2（EDT）
  ↓
Module 3（Visualization）
  ↓
输出 Web 页面
```

---

## 四、总体流程 Pipeline

```
输入图像（Expert / User）
        ↓
预处理（二值化 + 去噪）
        ↓
空间归一化（尺度 + 对齐）
        ↓
模块一：Skeleton 提取与对比
        ↓
模块二：EDT 计算与热力分析
        ↓
模块三：Plotly 可视化
        ↓
输出交互式网页
```

---

## 五、核心建模（统一表示）

### 5.1 骨架场（Structure Field）

```
S(x, y) ∈ {0,1}
```

来源：`skeletonize`

---

### 5.2 墨迹场（Distance Field）

```
D(x, y) = EDT(binary)
```

来源：`distance_transform_edt`

---

### 5.3 统一表达

```
Glyph = (S, D)
```

---

## 六、关键算法流程

### 6.1 双骨架提取

```python
from skimage.morphology import skeletonize, medial_axis

skel = skeletonize(binary)
skel_medial, dist = medial_axis(binary, return_distance=True)
```

---

### 6.2 骨架差异

```
D_skel = Hausdorff(skel_user, skel_expert)
```

---

### 6.3 EDT计算

```python
from scipy.ndimage import distance_transform_edt
edt = distance_transform_edt(binary)
```

---

### 6.4 墨迹差异

```
D_diff(x,y) = |EDT_user - EDT_expert|
```

---

## 七、可视化设计（Plotly）

### 7.1 骨架叠加图

* 红：User
* 绿：Expert
* 叠加显示

---

### 7.2 墨迹热力图

* 红：差异大
* 绿：差异小
* 使用连续 colormap

---

### 7.3 交互功能

* 图层开关（toggle）
* 透明度滑条
* hover 显示数值
* 局部放大查看细节

---

## 八、核心依赖库

### 图像处理

* OpenCV
* scikit-image

### 数值计算

* NumPy
* SciPy

### 匹配

* scipy.spatial

### Web可视化

* Plotly
* Dash

---

## 九、方法特点（论文贡献）

1. 将字形对比建模为**空间几何匹配问题**
2. 提出 **Skeleton + EDT 双场表示方法**
3. 构建 **可解释的可视化分析界面（Plotly）**

---

## 十、原型实现优先级

优先实现：

1. skeleton overlay
2. EDT heatmap
3. Plotly 可视化页面

后续扩展：

* 笔画级分割
* 错误类型映射（E401/E402/E403）

---

## 十一、总结

本系统通过模块化设计（Skeleton / EDT / Visualization），实现从字形输入到交互式分析输出的完整闭环，在无需时间序列信息的条件下，完成对书法字形的几何一致性分析与误差可视化表达。

---
