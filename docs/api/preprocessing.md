# 数据预处理模块 (preprocessing)

数据预处理模块提供了各种光谱数据预处理方法，包括标准化、归一化、基线校正、滤波和异常值处理等功能。

## 模块导入

```python
from nirapi import preprocessing
# 或者导入特定函数
from nirapi.preprocessing import SNV, MSC, SG
```

## 标准化方法

### SNV (Standard Normal Variate)

标准正态变量变换，用于消除光谱中的散射效应。

::: nirapi.nirapi.preprocessing.SNV

**示例:**

```python
import numpy as np

# 生成示例光谱数据
X = np.random.randn(100, 1200)

# 应用 SNV 变换
X_snv = preprocessing.SNV(X)
print(f"原始数据形状: {X.shape}")
print(f"SNV 处理后形状: {X_snv.shape}")

# 验证 SNV 效果（每个样本的均值应接近0，标准差接近1）
print(f"处理后第一个样本均值: {np.mean(X_snv[0]):.6f}")
print(f"处理后第一个样本标准差: {np.std(X_snv[0]):.6f}")
```

### MSC (Multiplicative Scatter Correction)

多元散射校正，用于校正光谱中的散射效应。

::: nirapi.nirapi.preprocessing.MSC

**示例:**

```python
# 应用 MSC 校正
X_msc = preprocessing.MSC(X)
print(f"MSC 处理后数据形状: {X_msc.shape}")
```

### weighted_SNV

加权标准正态变量变换，基于变量排序的归一化方法。

::: nirapi.nirapi.preprocessing.weighted_SNV

**示例:**

```python
# 应用加权 SNV
X_weighted_snv = preprocessing.weighted_SNV(
    X,
    epsilon=1e-5,
    Ns_times=100,
    Nw_times=100,
    draw_weighed_martix=False
)
```

### RNV (Robust Normal Variate)

鲁棒正态变量变换，对异常值更加鲁棒的标准化方法。

::: nirapi.nirapi.preprocessing.RNV

**示例:**

```python
# 应用 RNV 变换
X_rnv = preprocessing.RNV(X)
print(f"RNV 处理后数据形状: {X_rnv.shape}")
```

## 归一化方法

### normalization

最小-最大归一化，将数据缩放到指定范围。

::: nirapi.nirapi.preprocessing.normalization

**示例:**

```python
# 归一化到 [0, 1] 范围
X_norm = preprocessing.normalization(X)
print(f"归一化后最小值: {np.min(X_norm)}")
print(f"归一化后最大值: {np.max(X_norm)}")

# 归一化到 [-1, 1] 范围
X_norm_centered = preprocessing.normalization(X, norm_range=(-1, 1))
```

### MC (Mean Centering)

均值中心化，从每个变量中减去其均值。

::: nirapi.nirapi.preprocessing.MC

**示例:**

```python
# 应用均值中心化
X_mc = preprocessing.MC(X)
print(f"中心化后各变量均值: {np.mean(X_mc, axis=0)[:5]}")
```

## 滤波和平滑

### SG (Savitzky-Golay Filter)

Savitzky-Golay 滤波器，用于平滑光谱数据并可计算导数。

::: nirapi.nirapi.preprocessing.SG

**示例:**

```python
# 基本平滑
X_sg = preprocessing.SG(X, window_len=11, poly=2)

# 计算一阶导数
X_sg_d1 = preprocessing.SG(X, window_len=11, poly=2, deriv=1)

# 计算二阶导数
X_sg_d2 = preprocessing.SG(X, window_len=11, poly=2, deriv=2)

print(f"原始数据: {X.shape}")
print(f"平滑后: {X_sg.shape}")
print(f"一阶导数: {X_sg_d1.shape}")
print(f"二阶导数: {X_sg_d2.shape}")
```

## 基线校正

### remove_baseline_drift

移除基线漂移。

::: nirapi.nirapi.preprocessing.remove_baseline_drift

**示例:**

```python
# 移除基线漂移
X_baseline_corrected = preprocessing.remove_baseline_drift(X)
print(f"基线校正后数据形状: {X_baseline_corrected.shape}")
```

### airPLS

自适应迭代重加权惩罚最小二乘基线校正。

::: nirapi.nirapi.preprocessing.airPLS

**示例:**

```python
# 应用 airPLS 基线校正
X_airpls = preprocessing.airPLS(X, lambda_=100, porder=1, itermax=15)
print(f"airPLS 校正后数据形状: {X_airpls.shape}")
```

## 异常值处理

### remove_outliers

基于 Z-score 的异常值移除。

::: nirapi.nirapi.preprocessing.remove_outliers

**示例:**

```python
# 移除异常值
X_clean, y_clean, outlier_indices = preprocessing.remove_outliers(
    X, y, threshold=3.0
)
print(f"原始样本数: {len(X)}")
print(f"清洗后样本数: {len(X_clean)}")
print(f"移除的异常值索引: {outlier_indices}")
```

### remove_top_20_percent_mahalanobis

基于马哈拉诺比斯距离移除异常值。

::: nirapi.nirapi.preprocessing.remove_top_20_percent_mahalanobis

**示例:**

```python
# 使用马哈拉诺比斯距离移除异常值
X_clean_maha, y_clean_maha = preprocessing.remove_top_20_percent_mahalanobis(X, y)
print(f"马哈拉诺比斯清洗后样本数: {len(X_clean_maha)}")
```

## 预处理流水线

### 组合多种预处理方法

```python
def preprocess_pipeline(X, y=None):
    """
    完整的预处理流水线示例
    """
    # 1. 移除异常值
    if y is not None:
        X, y, _ = preprocessing.remove_outliers(X, y, threshold=3.0)
    
    # 2. SNV 标准化
    X = preprocessing.SNV(X)
    
    # 3. Savitzky-Golay 滤波
    X = preprocessing.SG(X, window_len=11, poly=2)
    
    # 4. 均值中心化
    X = preprocessing.MC(X)
    
    if y is not None:
        return X, y
    return X

# 使用流水线
X_processed, y_processed = preprocess_pipeline(X, y)
```

### 自定义预处理函数

```python
def custom_preprocess(X, method_list=['SNV', 'SG', 'MC']):
    """
    根据方法列表进行预处理
    """
    X_processed = X.copy()
    
    for method in method_list:
        if method == 'SNV':
            X_processed = preprocessing.SNV(X_processed)
        elif method == 'MSC':
            X_processed = preprocessing.MSC(X_processed)
        elif method == 'SG':
            X_processed = preprocessing.SG(X_processed)
        elif method == 'MC':
            X_processed = preprocessing.MC(X_processed)
        elif method == 'normalization':
            X_processed = preprocessing.normalization(X_processed)
    
    return X_processed

# 使用自定义预处理
X_custom = custom_preprocess(X, ['SNV', 'SG', 'normalization'])
```

## 预处理效果可视化

```python
import matplotlib.pyplot as plt

def plot_preprocessing_comparison(X_original, X_processed, sample_idx=0):
    """
    比较预处理前后的光谱
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(X_original[sample_idx])
    plt.title('原始光谱')
    plt.xlabel('波长点')
    plt.ylabel('吸光度')
    
    plt.subplot(1, 2, 2)
    plt.plot(X_processed[sample_idx])
    plt.title('预处理后光谱')
    plt.xlabel('波长点')
    plt.ylabel('吸光度')
    
    plt.tight_layout()
    plt.show()

# 可视化预处理效果
plot_preprocessing_comparison(X, X_processed)
```

## 最佳实践

### 1. 预处理方法选择

```python
# 常用预处理组合
PREPROCESS_COMBINATIONS = {
    'basic': ['SNV', 'SG'],
    'advanced': ['MSC', 'SG', 'MC'],
    'robust': ['RNV', 'SG'],
    'derivative': ['SNV', 'SG_d1'],  # 一阶导数
    'classification': ['normalization', 'MC']
}
```

### 2. 参数优化

```python
# 使用网格搜索优化 SG 滤波参数
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

def optimize_sg_parameters(X, y):
    """
    优化 Savitzky-Golay 滤波参数
    """
    param_grid = {
        'window_len': [5, 7, 9, 11, 13, 15],
        'poly': [1, 2, 3, 4]
    }
    
    best_score = -np.inf
    best_params = None
    
    for window_len in param_grid['window_len']:
        for poly in param_grid['poly']:
            if poly < window_len:  # 确保多项式阶数小于窗口长度
                X_sg = preprocessing.SG(X, window_len=window_len, poly=poly)
                # 使用交叉验证评估
                # ... 评估代码
    
    return best_params
```

### 3. 数据质量检查

```python
def check_data_quality(X):
    """
    检查数据质量
    """
    # 检查 NaN 值
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"警告: 发现 {nan_count} 个 NaN 值")
    
    # 检查无穷大值
    inf_count = np.isinf(X).sum()
    if inf_count > 0:
        print(f"警告: 发现 {inf_count} 个无穷大值")
    
    # 检查数据范围
    print(f"数据范围: [{np.min(X):.4f}, {np.max(X):.4f}]")
    print(f"数据均值: {np.mean(X):.4f}")
    print(f"数据标准差: {np.std(X):.4f}")

# 预处理前后检查数据质量
check_data_quality(X)
check_data_quality(X_processed)
```

## 相关模块

- [机器学习模块](ml_model.md) - 特征选择和模型训练
- [分析模块](analysis.md) - 数据分析工具
- [可视化模块](draw.md) - 预处理效果可视化
