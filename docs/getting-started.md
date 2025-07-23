# 快速开始

本指南将帮助您快速上手 NIR API，从安装到运行第一个光谱分析项目。

## 系统要求

- Python 3.8 或更高版本
- 推荐使用虚拟环境

## 安装

### 使用 pip 安装

```bash
pip install nirapi
```

### 从源码安装

```bash
git clone https://github.com/zata/nirapi.git
cd nirapi
pip install -e .
```

### 依赖包

NIR API 依赖以下主要包：

- `numpy` - 数值计算
- `pandas` - 数据处理
- `scikit-learn` - 机器学习
- `matplotlib` - 基础绘图
- `plotly` - 交互式可视化
- `optuna` - 超参数优化

完整的依赖列表请参见 `requirements.txt`。

## 验证安装

```python
import nirapi
print(f"NIR API 版本: {nirapi.__version__}")

# 测试主要模块导入
from nirapi import load_data, preprocessing, ML_model, draw, analysis
print("所有模块导入成功！")
```

## 第一个示例

让我们通过一个简单的示例来了解 NIR API 的基本使用流程。

### 1. 数据准备

```python
import numpy as np
import pandas as pd
from nirapi import load_data, preprocessing, ML_model, draw

# 生成示例数据（实际使用中您会从文件或数据库加载）
np.random.seed(42)
n_samples, n_features = 100, 1200
X = np.random.randn(n_samples, n_features)  # 光谱数据
y = np.random.uniform(0, 10, n_samples)     # 目标变量（如血糖浓度）

print(f"数据形状: X={X.shape}, y={y.shape}")
```

### 2. 数据预处理

```python
# 标准正态变量变换 (SNV)
X_snv = preprocessing.SNV(X)

# Savitzky-Golay 滤波
X_filtered = preprocessing.SG(X_snv, window_len=11, poly=2)

print("预处理完成")
```

### 3. 数据分割

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y, test_size=0.3, random_state=42
)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
```

### 4. 模型训练

```python
# 使用偏最小二乘回归 (PLSR)
y_train_pred, y_test_pred, y_train_actual, y_test_actual = ML_model.PLSR(
    X_train, X_test, y_train, y_test, n_components=10
)

print("模型训练完成")
```

### 5. 结果可视化

```python
# 绘制预测结果
draw.pred_plot(y_test_actual, y_test_pred, title="PLSR 预测结果")

# 计算性能指标
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
r2 = r2_score(y_test_actual, y_test_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
```

## 使用真实数据

### 从 Excel 文件加载数据

```python
# 加载光谱数据
data_df, info_df, X, y = analysis.load_spectral_data("your_data.xlsx")

# 基本数据信息
analysis.print_basic_data_info(data_df, X, y)

# 光谱概览图
analysis.plot_spectral_overview(X)
```

### 从数据库加载数据

```python
# 从 MySQL 数据库加载数据
dataset = load_data.get_dataset_from_mysql(
    database='光谱数据库',
    table_name="卷积式_v1",
    project_name="血糖检测",
    X_type=['光谱'],
    y_type=['血糖值']
)

X = dataset['光谱']
y = dataset['血糖值']
```

## 自动机器学习

NIR API 提供了自动机器学习功能，可以自动选择最佳的预处理方法和模型：

```python
from nirapi.utils import train_model_for_trick_game_v2
import datetime

# 定义搜索空间
kw = {
    "selected_outlier": ["不做异常值去除"],
    "selected_preprocess": ["SNV", "SG"],
    "selected_feat_sec": ["pca"],
    "selected_model": ["PLSR", "SVR"]
}

# 运行自动机器学习
result = train_model_for_trick_game_v2(
    splited_data=(X_train, X_test, y_train, y_test),
    max_attempts=10,
    n_trials=50,
    selected_metric="rmse",
    target_score=0.5,
    filename=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_automl",
    **kw
)
```

## 光谱重建

NIR API 还提供了强大的光谱重建功能：

```python
from nirapi.AnalysisClass.Create_rec_task import SpectralDictionaryMapper

# 创建重建任务
rec_task = SpectralDictionaryMapper()

# 训练（使用两种不同的光谱数据）
pd_samples = np.random.rand(100, 200)  # PD 光谱
ft_spectra = np.random.rand(100, 300)  # FT 光谱

rec_task.fit(pd_samples, ft_spectra)

# 重建
reconstructed = rec_task.transform(pd_samples[:5])
print(f"重建结果形状: {reconstructed.shape}")
```

## 常见问题

### Q: 如何处理中文字符显示问题？

A: NIR API 已经内置了中文字体配置，如果仍有问题，可以手动设置：

```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

### Q: 如何选择合适的预处理方法？

A: 建议使用自动机器学习功能来自动选择最佳的预处理组合，或者参考以下常用组合：

- 基础组合：SNV + SG 滤波
- 高级组合：MSC + 一阶导数
- 分类任务：标准化 + PCA

### Q: 模型性能不佳怎么办？

A: 可以尝试以下方法：

1. 检查数据质量，移除异常值
2. 尝试不同的预处理方法组合
3. 调整模型参数
4. 使用特征选择方法
5. 增加训练数据量

## 下一步

现在您已经掌握了 NIR API 的基本使用方法，可以继续学习：

- [API 参考文档](api/overview.md) - 了解所有可用的函数和类
- [教程](tutorials/basic-usage.md) - 深入学习各个模块的使用
- [示例](examples/complete-workflow.md) - 查看完整的应用案例

如果遇到问题，请查看 [GitHub Issues](https://github.com/zata/nirapi/issues) 或提交新的问题。
