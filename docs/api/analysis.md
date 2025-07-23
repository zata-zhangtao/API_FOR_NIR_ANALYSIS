# 数据分析模块 (analysis)

数据分析模块提供了专门的光谱数据分析工具，包括数据加载、统计分析、异常值检测和相关性分析等功能。

## 模块导入

```python
from nirapi import analysis
# 或者导入特定函数
from nirapi.analysis import analyze_spectral_data, load_spectral_data, detect_outliers
```

## 数据加载和验证

### load_spectral_data

从 Excel 文件加载光谱数据。

```python
def load_spectral_data(file_path):
    """
    从 Excel 文件加载光谱数据
    
    Parameters:
    -----------
    file_path : str
        Excel 文件路径，需包含 'data' 和 'notes' 工作表
        
    Returns:
    --------
    tuple
        (data_df, info_df, X, y) 其中：
        - data_df: 主数据 DataFrame
        - info_df: 备注信息 DataFrame  
        - X: 特征矩阵
        - y: 目标变量
        
    Raises:
    -------
    FileNotFoundError
        文件不存在
    ValueError
        文件格式无效或缺少必需的工作表
    KeyError
        缺少必需的列
    """
```

**示例:**

```python
# 加载光谱数据
file_path = "spectral_data.xlsx"
data_df, info_df, X, y = analysis.load_spectral_data(file_path)

print(f"数据形状: {data_df.shape}")
print(f"特征矩阵: {X.shape}")
print(f"目标变量: {y.shape}")
```

### print_basic_data_info

打印数据的基本统计信息。

```python
def print_basic_data_info(data_df, X, y):
    """
    打印数据基本信息
    
    Parameters:
    -----------
    data_df : DataFrame
        原始数据
    X : array-like
        特征矩阵
    y : array-like
        目标变量
    """
```

**示例:**

```python
# 打印基本信息
analysis.print_basic_data_info(data_df, X, y)

# 输出示例:
# ==================================================
# 1. Basic Data Information
# ==================================================
# Data dimensions: (200, 1201)
# Number of features: 1200
# 
# First 5 rows:
#    wavelength_1000  wavelength_1001  ...  target
# 0         1.234567         1.245678  ...    5.67
# ...
```

## 光谱数据可视化

### plot_spectral_overview

绘制光谱数据概览图。

```python
def plot_spectral_overview(X, wavelengths=None, n_samples=5):
    """
    绘制光谱数据概览
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        光谱数据
    wavelengths : array-like, optional
        波长信息
    n_samples : int
        显示的样本数量
    """
```

**示例:**

```python
import numpy as np

# 生成示例数据
wavelengths = np.linspace(1000, 2500, 1200)
X = np.random.randn(100, 1200) * 0.1 + 1.0

# 绘制光谱概览
analysis.plot_spectral_overview(X, wavelengths, n_samples=10)
```

### plot_mean

绘制平均光谱及标准差。

```python
def plot_mean(X, wavelengths=None, title="平均光谱"):
    """
    绘制平均光谱
    
    Parameters:
    -----------
    X : array-like
        光谱数据
    wavelengths : array-like, optional
        波长信息
    title : str
        图表标题
    """
```

## 异常值检测

### detect_outliers

检测光谱数据中的异常值。

```python
def detect_outliers(X, method='mahalanobis', threshold=95):
    """
    检测异常值
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        输入数据
    method : str
        检测方法，支持 'mahalanobis', 'zscore', 'iqr'
    threshold : float
        阈值百分位数（对于 mahalanobis）或标准差倍数（对于 zscore）
        
    Returns:
    --------
    list
        异常值索引列表
    """
```

**示例:**

```python
# 使用马哈拉诺比斯距离检测异常值
outlier_indices = analysis.detect_outliers(X, method='mahalanobis', threshold=95)
print(f"检测到 {len(outlier_indices)} 个异常值")
print(f"异常值索引: {outlier_indices}")

# 使用 Z-score 方法
outlier_indices_z = analysis.detect_outliers(X, method='zscore', threshold=3.0)
print(f"Z-score 方法检测到 {len(outlier_indices_z)} 个异常值")

# 移除异常值
normal_indices = [i for i in range(len(X)) if i not in outlier_indices]
X_clean = X[normal_indices]
```

### plot_detailed_spectral_analysis

绘制详细的光谱分析图表。

```python
def plot_detailed_spectral_analysis(X, y=None, outlier_indices=None):
    """
    绘制详细的光谱分析图表
    
    Parameters:
    -----------
    X : array-like
        光谱数据
    y : array-like, optional
        目标变量
    outlier_indices : list, optional
        异常值索引
    """
```

## 相关性分析

### analyze_correlations

分析光谱数据与目标变量的相关性。

```python
def analyze_correlations(X, y, wavelengths=None, method='pearson'):
    """
    分析相关性
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        光谱数据
    y : array-like, shape (n_samples,)
        目标变量
    wavelengths : array-like, optional
        波长信息
    method : str
        相关性计算方法，'pearson' 或 'spearman'
        
    Returns:
    --------
    dict
        相关性分析结果
    """
```

**示例:**

```python
# 分析相关性
y = np.random.uniform(0, 10, len(X))
corr_results = analysis.analyze_correlations(X, y, wavelengths, method='pearson')

print(f"最高相关性: {corr_results['max_correlation']:.4f}")
print(f"最高相关性波长: {corr_results['max_corr_wavelength']:.1f} nm")
print(f"平均相关性: {corr_results['mean_correlation']:.4f}")
```

### plot_correlation_graph

绘制相关性图表。

```python
def plot_correlation_graph(correlations, wavelengths=None, threshold=0.3):
    """
    绘制相关性图表
    
    Parameters:
    -----------
    correlations : array-like
        相关系数
    wavelengths : array-like, optional
        波长信息
    threshold : float
        显著相关性阈值
    """
```

## 综合分析

### analyze_spectral_data

执行综合的光谱数据分析。

```python
def analyze_spectral_data(file_path):
    """
    综合光谱数据分析
    
    Parameters:
    -----------
    file_path : str
        Excel 文件路径
        
    Returns:
    --------
    tuple
        (data_df, info_df, outlier_indices, analysis_results)
        
    Example:
    --------
    file_path = "glucose_data.xlsx"
    data_df, info_df, outlier_indices, results = analyze_spectral_data(file_path)
    """
```

**示例:**

```python
# 执行综合分析
file_path = "glucose_spectral_data.xlsx"
data_df, info_df, outlier_indices, analysis_results = analysis.analyze_spectral_data(file_path)

print("=== 综合分析结果 ===")
print(f"数据维度: {data_df.shape}")
print(f"异常值数量: {len(outlier_indices)}")
print(f"数据质量评分: {analysis_results.get('quality_score', 'N/A')}")
```

## 数据质量评估

### assess_data_quality

评估光谱数据质量。

```python
def assess_data_quality(X, y=None):
    """
    评估数据质量
    
    Parameters:
    -----------
    X : array-like
        光谱数据
    y : array-like, optional
        目标变量
        
    Returns:
    --------
    dict
        数据质量报告
    """
```

**示例:**

```python
# 评估数据质量
quality_report = analysis.assess_data_quality(X, y)

print("=== 数据质量报告 ===")
print(f"缺失值比例: {quality_report['missing_ratio']:.2%}")
print(f"异常值比例: {quality_report['outlier_ratio']:.2%}")
print(f"信噪比: {quality_report['snr']:.2f}")
print(f"光谱质量评分: {quality_report['spectral_quality']:.2f}")
```

## 统计分析工具

### spectral_statistics

计算光谱统计信息。

```python
def spectral_statistics(X, wavelengths=None):
    """
    计算光谱统计信息
    
    Parameters:
    -----------
    X : array-like
        光谱数据
    wavelengths : array-like, optional
        波长信息
        
    Returns:
    --------
    dict
        统计信息字典
    """
```

### compare_spectral_groups

比较不同组别的光谱特征。

```python
def compare_spectral_groups(X, groups, wavelengths=None):
    """
    比较不同组别的光谱特征
    
    Parameters:
    -----------
    X : array-like
        光谱数据
    groups : array-like
        组别标签
    wavelengths : array-like, optional
        波长信息
        
    Returns:
    --------
    dict
        组别比较结果
    """
```

## 实用工具

### 完整分析流程

```python
def complete_spectral_analysis_workflow(file_path, output_dir="analysis_results"):
    """
    完整的光谱分析工作流程
    """
    import os
    import matplotlib.pyplot as plt
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载数据
    print("1. 加载数据...")
    data_df, info_df, X, y = analysis.load_spectral_data(file_path)
    
    # 2. 基本信息
    print("2. 数据基本信息...")
    analysis.print_basic_data_info(data_df, X, y)
    
    # 3. 光谱概览
    print("3. 绘制光谱概览...")
    plt.figure(figsize=(12, 6))
    analysis.plot_spectral_overview(X)
    plt.savefig(f"{output_dir}/spectral_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 异常值检测
    print("4. 异常值检测...")
    outlier_indices = analysis.detect_outliers(X)
    print(f"   检测到 {len(outlier_indices)} 个异常值")
    
    # 5. 相关性分析
    if y is not None:
        print("5. 相关性分析...")
        corr_results = analysis.analyze_correlations(X, y)
        
        plt.figure(figsize=(12, 6))
        analysis.plot_correlation_graph(corr_results['correlations'])
        plt.savefig(f"{output_dir}/correlation_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. 数据质量评估
    print("6. 数据质量评估...")
    quality_report = analysis.assess_data_quality(X, y)
    
    # 保存结果
    results = {
        'data_shape': X.shape,
        'outlier_count': len(outlier_indices),
        'outlier_indices': outlier_indices,
        'quality_report': quality_report
    }
    
    if y is not None:
        results['correlation_results'] = corr_results
    
    print(f"分析完成！结果保存在 {output_dir} 目录中")
    return results

# 使用示例
results = complete_spectral_analysis_workflow("data.xlsx")
```

## 最佳实践

### 1. 数据验证

```python
def validate_spectral_data(X, y=None):
    """
    验证光谱数据的有效性
    """
    # 检查数据类型
    assert isinstance(X, (np.ndarray, pd.DataFrame)), "X 必须是 numpy 数组或 DataFrame"
    
    # 检查数据形状
    assert len(X.shape) == 2, "X 必须是二维数组"
    
    # 检查缺失值
    if np.isnan(X).any():
        print("警告: 数据中包含缺失值")
    
    # 检查目标变量
    if y is not None:
        assert len(X) == len(y), "X 和 y 的样本数必须相同"
    
    print("✓ 数据验证通过")
```

### 2. 分析报告生成

```python
def generate_analysis_report(results, output_file="analysis_report.txt"):
    """
    生成分析报告
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== 光谱数据分析报告 ===\n\n")
        f.write(f"数据维度: {results['data_shape']}\n")
        f.write(f"异常值数量: {results['outlier_count']}\n")
        f.write(f"数据质量评分: {results['quality_report']['spectral_quality']:.2f}\n")
        # ... 更多报告内容
```

## 相关模块

- [数据加载模块](load_data.md) - 数据加载和管理
- [预处理模块](preprocessing.md) - 数据预处理方法
- [可视化模块](draw.md) - 数据可视化工具
