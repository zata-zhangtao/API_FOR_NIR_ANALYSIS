# 数据可视化模块 (draw)

数据可视化模块提供了丰富的图表绘制功能，专门为光谱数据分析和机器学习结果展示而设计。

## 模块导入

```python
from nirapi import draw
# 或者导入特定函数
from nirapi.draw import spectral_absorption, pred_plot, analyze_model_performance
```

## 光谱可视化

### spectral_absorption

绘制光谱吸收图。

```python
def spectral_absorption(X, wavelengths=None, title="光谱吸收图", 
                       sample_indices=None, figsize=(12, 6)):
    """
    绘制光谱吸收图
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        光谱数据
    wavelengths : array-like, optional
        波长信息
    title : str
        图表标题
    sample_indices : list, optional
        要显示的样本索引
    figsize : tuple
        图表大小
    """
```

**示例:**

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
wavelengths = np.linspace(1000, 2500, 1200)
X = np.random.randn(50, 1200) * 0.1 + 1.0

# 绘制光谱吸收图
draw.spectral_absorption(X, wavelengths, title="近红外光谱吸收图")
plt.show()

# 绘制特定样本
draw.spectral_absorption(X, wavelengths, sample_indices=[0, 1, 2, 3, 4])
plt.show()
```

### spectral_absorption_v2

增强版光谱吸收图，支持更多自定义选项。

```python
def spectral_absorption_v2(X, wavelengths=None, labels=None, 
                          colors=None, alpha=0.7, figsize=(12, 6)):
    """
    增强版光谱吸收图
    
    Parameters:
    -----------
    X : array-like
        光谱数据
    wavelengths : array-like, optional
        波长信息
    labels : list, optional
        样本标签
    colors : list, optional
        颜色列表
    alpha : float
        透明度
    figsize : tuple
        图表大小
    """
```

## 模型性能可视化

### pred_plot

绘制预测值与真实值的对比图。

```python
def pred_plot(y_true, y_pred, title="预测结果", figsize=(8, 6)):
    """
    绘制预测结果对比图
    
    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    title : str
        图表标题
    figsize : tuple
        图表大小
    """
```

**示例:**

```python
# 生成示例预测数据
y_true = np.random.uniform(0, 10, 100)
y_pred = y_true + np.random.normal(0, 0.5, 100)

# 绘制预测结果
draw.pred_plot(y_true, y_pred, title="血糖预测结果")
plt.show()
```

### train_val_and_test_pred_plot

绘制训练集、验证集和测试集的预测结果。

```python
def train_val_and_test_pred_plot(y_train_true, y_train_pred, 
                                y_val_true, y_val_pred,
                                y_test_true, y_test_pred,
                                title="模型性能对比"):
    """
    绘制训练、验证、测试集预测结果
    
    Parameters:
    -----------
    y_train_true, y_train_pred : array-like
        训练集真实值和预测值
    y_val_true, y_val_pred : array-like
        验证集真实值和预测值
    y_test_true, y_test_pred : array-like
        测试集真实值和预测值
    title : str
        图表标题
    """
```

### analyze_model_performance

自动分析和可视化模型性能。

```python
def analyze_model_performance(y_true, y_pred, model_name="Model", 
                            save_path=None, show_plots=True):
    """
    自动分析模型性能并生成可视化报告
    
    Parameters:
    -----------
    y_true : array-like
        真实值
    y_pred : array-like
        预测值
    model_name : str
        模型名称
    save_path : str, optional
        保存路径
    show_plots : bool
        是否显示图表
        
    Returns:
    --------
    dict
        性能指标字典
    """
```

## PCA 可视化

### plot_pca_with_class_distribution

绘制 PCA 分析结果和类别分布。

```python
def plot_pca_with_class_distribution(X, y=None, n_components=2, 
                                   title="PCA 分析结果"):
    """
    绘制 PCA 分析结果
    
    Parameters:
    -----------
    X : array-like
        特征数据
    y : array-like, optional
        标签数据
    n_components : int
        主成分数量
    title : str
        图表标题
    """
```

**示例:**

```python
from sklearn.decomposition import PCA

# 生成示例数据
X = np.random.randn(200, 1200)
y = np.random.randint(0, 3, 200)

# PCA 可视化
draw.plot_pca_with_class_distribution(X, y, title="光谱数据 PCA 分析")
plt.show()
```

### plot_3d_pca_scatter

绘制 3D PCA 散点图。

```python
def plot_3d_pca_scatter(X, y=None, title="3D PCA 散点图"):
    """
    绘制 3D PCA 散点图
    
    Parameters:
    -----------
    X : array-like
        特征数据
    y : array-like, optional
        标签数据
    title : str
        图表标题
    """
```

## 统计分析图表

### Numerical_distribution

绘制数值分布图。

```python
def Numerical_distribution(data, title="数值分布", bins=30):
    """
    绘制数值分布直方图
    
    Parameters:
    -----------
    data : array-like
        数据
    title : str
        图表标题
    bins : int
        直方图箱数
    """
```

### classification_report_plot

绘制分类报告图表。

```python
def classification_report_plot(y_true, y_pred, class_names=None):
    """
    绘制分类报告热力图
    
    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_pred : array-like
        预测标签
    class_names : list, optional
        类别名称
    """
```

## 交互式图表

### plotly_simple_chart

创建简单的 Plotly 交互式图表。

```python
def plotly_simple_chart(x, y, title="交互式图表", x_label="X", y_label="Y"):
    """
    创建 Plotly 交互式线图
    
    Parameters:
    -----------
    x : array-like
        X 轴数据
    y : array-like
        Y 轴数据
    title : str
        图表标题
    x_label : str
        X 轴标签
    y_label : str
        Y 轴标签
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly 图表对象
    """
```

**示例:**

```python
# 创建交互式光谱图
wavelengths = np.linspace(1000, 2500, 1200)
spectrum = np.random.randn(1200) * 0.1 + 1.0

fig = draw.plotly_simple_chart(
    wavelengths, spectrum, 
    title="交互式光谱图",
    x_label="波长 (nm)",
    y_label="吸光度"
)
fig.show()
```

## 专业图表

### clarke_error_grid

绘制 Clarke 误差网格图（用于血糖检测）。

```python
def clarke_error_grid(y_true, y_pred, title="Clarke 误差网格"):
    """
    绘制 Clarke 误差网格图
    
    Parameters:
    -----------
    y_true : array-like
        真实血糖值
    y_pred : array-like
        预测血糖值
    title : str
        图表标题
    """
```

### Sample_spectral_ranking

绘制样本光谱排序图。

```python
def Sample_spectral_ranking(X, y, n_samples=10):
    """
    绘制样本光谱排序图
    
    Parameters:
    -----------
    X : array-like
        光谱数据
    y : array-like
        目标变量
    n_samples : int
        显示的样本数量
    """
```

## 使用技巧

### 1. 批量绘图

```python
def batch_plotting(X, y, plot_functions, save_dir="plots"):
    """
    批量生成多种图表
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for func_name, func in plot_functions.items():
        plt.figure(figsize=(10, 6))
        func(X, y)
        plt.savefig(f"{save_dir}/{func_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

# 使用示例
plot_functions = {
    'spectral_absorption': draw.spectral_absorption,
    'pca_analysis': draw.plot_pca_with_class_distribution,
    'distribution': draw.Numerical_distribution
}

batch_plotting(X, y, plot_functions)
```

### 2. 自定义样式

```python
# 设置中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
plt.style.use('seaborn-v0_8')  # 或其他样式
```

### 3. 保存高质量图片

```python
def save_high_quality_plot(fig, filename, dpi=300):
    """
    保存高质量图片
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
```

## 相关模块

- [分析模块](analysis.md) - 数据分析工具
- [机器学习模块](ml_model.md) - 模型训练和评估
- [预处理模块](preprocessing.md) - 数据预处理方法
