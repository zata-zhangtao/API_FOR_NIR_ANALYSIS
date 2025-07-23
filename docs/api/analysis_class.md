# 分析类模块 (AnalysisClass)

分析类模块提供了高级的光谱分析功能，特别是光谱重建算法和模型评估器。这些类封装了复杂的算法，提供了简单易用的接口。

## 模块导入

```python
from nirapi.AnalysisClass.Create_rec_task import (
    SpectrumModelEvaluator,
    SpectralDictionaryMapper,
    SpectrumTransformerByNN,
    conv_MatMul_recon,
    conv_DFT_recon
)
```

## 模型评估器

### SpectrumModelEvaluator

全面的光谱模型评估器，可以评估所有类型重建模型的效果。

```python
class SpectrumModelEvaluator:
    """
    光谱模型评估器
    
    用于评估各种光谱重建模型的性能，生成详细的评估报告和可视化结果。
    """
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def run_evaluation(self, X_train, y_train, X_test, y_test):
        """
        运行完整的模型评估
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples, n_features)
            训练集输入光谱
        y_train : array-like, shape (n_samples, n_features)
            训练集目标光谱
        X_test : array-like, shape (n_samples, n_features)
            测试集输入光谱
        y_test : array-like, shape (n_samples, n_features)
            测试集目标光谱
            
        Returns:
        --------
        dict
            评估结果，包含图表路径和文档路径
        """
```

**示例:**

```python
import numpy as np

# 生成示例数据
np.random.seed(42)
X_train = np.random.rand(100, 1200)  # 100个样本，1200个特征
y_train = np.random.rand(100, 1200)  # 对应的目标光谱
X_test = np.random.rand(20, 1200)
y_test = np.random.rand(20, 1200)

# 创建评估器
evaluator = SpectrumModelEvaluator()

# 运行评估
results = evaluator.run_evaluation(X_train, y_train, X_test, y_test)

print(f"评估图表保存路径: {results['plot_path']}")
print(f"评估报告保存路径: {results['doc_path']}")
```

## 光谱重建算法

### SpectralDictionaryMapper

基于字典学习的光谱重建方法，适用于任何光谱仪。

```python
class SpectralDictionaryMapper:
    """
    基于字典学习的光谱重建
    
    使用字典学习算法学习两种光谱之间的映射关系，
    可以将一种光谱仪的数据重建为另一种光谱仪的数据。
    """
    
    def __init__(self, n_components=50, alpha=1.0, max_iter=1000):
        """
        初始化字典学习重建器
        
        Parameters:
        -----------
        n_components : int
            字典原子数量
        alpha : float
            稀疏性参数
        max_iter : int
            最大迭代次数
        """
    
    def fit(self, X_source, X_target):
        """
        训练字典学习模型
        
        Parameters:
        -----------
        X_source : array-like, shape (n_samples, n_features_source)
            源光谱数据
        X_target : array-like, shape (n_samples, n_features_target)
            目标光谱数据
            
        Returns:
        --------
        self : object
            返回自身以支持链式调用
        """
    
    def transform(self, X_source):
        """
        重建光谱
        
        Parameters:
        -----------
        X_source : array-like, shape (n_samples, n_features_source)
            待重建的源光谱数据
            
        Returns:
        --------
        X_reconstructed : array-like, shape (n_samples, n_features_target)
            重建后的目标光谱数据
        """
```

**示例:**

```python
# 创建字典学习重建器
rec_task = SpectralDictionaryMapper(n_components=100, alpha=0.1)

# 准备数据（PD光谱 -> FT光谱）
pd_samples = np.random.rand(100, 200)  # PD光谱仪数据
ft_spectra = np.random.rand(100, 300)  # FT光谱仪数据

# 训练模型
rec_task.fit(pd_samples, ft_spectra)

# 重建新的光谱
new_pd_samples = np.random.rand(5, 200)
reconstructed_ft = rec_task.transform(new_pd_samples)

print(f"输入形状: {new_pd_samples.shape}")
print(f"重建结果形状: {reconstructed_ft.shape}")
```

### SpectrumTransformerByNN

基于神经网络的光谱重建方法，适用于任何光谱仪。

```python
class SpectrumTransformerByNN:
    """
    基于神经网络的光谱重建
    
    使用深度神经网络学习光谱之间的非线性映射关系。
    """
    
    def __init__(self, hidden_layers=[512, 256, 128], activation='relu', 
                 epochs=100, batch_size=32, learning_rate=0.001):
        """
        初始化神经网络重建器
        
        Parameters:
        -----------
        hidden_layers : list
            隐藏层神经元数量列表
        activation : str
            激活函数
        epochs : int
            训练轮数
        batch_size : int
            批次大小
        learning_rate : float
            学习率
        """
    
    def fit(self, X_source, X_target):
        """
        训练神经网络模型
        
        Parameters:
        -----------
        X_source : array-like
            源光谱数据
        X_target : array-like
            目标光谱数据
        """
    
    def transform(self, X_source):
        """
        使用训练好的神经网络重建光谱
        
        Parameters:
        -----------
        X_source : array-like
            待重建的源光谱数据
            
        Returns:
        --------
        X_reconstructed : array-like
            重建后的光谱数据
        """
```

**示例:**

```python
# 创建神经网络重建器
rec_task = SpectrumTransformerByNN(
    hidden_layers=[512, 256, 128],
    epochs=50,
    batch_size=16
)

# 训练模型
rec_task.fit(pd_samples, ft_spectra)

# 重建光谱
reconstructed = rec_task.transform(new_pd_samples)
print(f"神经网络重建结果形状: {reconstructed.shape}")
```

## 传统重建方法

### conv_MatMul_recon

基于矩阵乘法的卷积式重建方法，适用于1200维度的卷积式光谱仪。

```python
class conv_MatMul_recon:
    """
    矩阵乘法重建方法
    
    直接通过矩阵乘法求解传输矩阵的逆矩阵进行重建。
    专门用于卷积式光谱仪的重建任务。
    """
    
    def __init__(self):
        """初始化矩阵乘法重建器"""
        pass
    
    def transform(self, pd_samples, pd_source):
        """
        执行矩阵乘法重建
        
        Parameters:
        -----------
        pd_samples : array-like, shape (n_samples, 1200)
            PD样本数据，必须是1200维
        pd_source : array-like, shape (n_samples, 1200)
            PD源数据，必须是1200维
            
        Returns:
        --------
        tuple
            (rec_sample, rec_source) 重建后的样本和源数据
        """
```

**示例:**

```python
# 创建矩阵乘法重建器
rec_task = conv_MatMul_recon()

# 准备1200维数据
pd_samples = np.random.rand(30, 1200)
pd_source = np.random.rand(30, 1200)

# 执行重建（不需要训练）
rec_sample, rec_source = rec_task.transform(pd_samples, pd_source)

print(f"重建样本形状: {rec_sample.shape}")
print(f"重建源数据形状: {rec_source.shape}")
```

### conv_DFT_recon

基于离散傅里叶变换的卷积式重建方法。

```python
class conv_DFT_recon:
    """
    DFT重建方法
    
    使用离散傅里叶变换进行卷积式光谱重建。
    适用于1200维度的卷积式光谱仪。
    """
    
    def __init__(self):
        """初始化DFT重建器"""
        pass
    
    def transform(self, pd_samples, pd_source):
        """
        执行DFT重建
        
        Parameters:
        -----------
        pd_samples : array-like, shape (n_samples, 1200)
            PD样本数据
        pd_source : array-like, shape (n_samples, 1200)
            PD源数据
            
        Returns:
        --------
        tuple
            (rec_sample, rec_source) 重建结果
        """
```

**示例:**

```python
# 创建DFT重建器
rec_task = conv_DFT_recon()

# 执行DFT重建
rec_sample, rec_source = rec_task.transform(pd_samples, pd_source)

print(f"DFT重建样本形状: {rec_sample.shape}")
print(f"DFT重建源数据形状: {rec_source.shape}")
```

## 高级重建方法

### FermentPeelVectorReLU

结合字典学习和向量拟合的高级重建方法。

```python
class FermentPeelVectorReLU:
    """
    发酵剥离向量ReLU重建方法
    
    先使用字典学习学习基础映射，然后用向量拟合额外的变换。
    字典学习是可选的，适用于任何光谱仪和任意维度。
    """
    
    def __init__(self, enable_dic=True, n_components=50):
        """
        初始化重建器
        
        Parameters:
        -----------
        enable_dic : bool
            是否启用字典学习
        n_components : int
            字典原子数量
        """
        self.enable_dic = enable_dic
        self.n_components = n_components
    
    def fit(self, X_source, X_target):
        """
        训练模型
        
        Parameters:
        -----------
        X_source : array-like
            源光谱数据
        X_target : array-like
            目标光谱数据
        """
    
    def transform(self, X_source):
        """
        执行重建
        
        Parameters:
        -----------
        X_source : array-like
            待重建数据
            
        Returns:
        --------
        array-like
            重建结果
        """
```

**示例:**

```python
# 创建高级重建器
rec_task = FermentPeelVectorReLU(enable_dic=True, n_components=100)

# 准备任意维度数据
PD_Samples = np.random.rand(30, 800)
FT_spectra = np.random.rand(30, 800)

# 分阶段训练
# 1. 启用字典学习训练
rec_task.enable_dic = True
rec_task.fit(PD_Samples, FT_spectra)

# 2. 关闭字典学习，训练向量拟合
rec_task.enable_dic = False
rec_task.fit(PD_Samples, FT_spectra)

# 3. 重新启用字典学习进行预测
rec_task.enable_dic = True
pred = rec_task.transform(PD_Samples)
print(f"高级重建结果形状: {pred.shape}")
```

### conv_FullDictDotMat

字典学习结合可训练矩阵的重建方法。

```python
class conv_FullDictDotMat:
    """
    完整字典点乘矩阵重建方法
    
    使用字典学习结合可训练矩阵进行光谱重建。
    适用于任何光谱仪和任意维度。
    """
    
    def __init__(self, enable_dic=True):
        """
        初始化重建器
        
        Parameters:
        -----------
        enable_dic : bool
            是否启用字典学习
        """
        self.enable_dic = enable_dic
    
    def fit(self, X_source, X_target):
        """训练模型"""
        pass
    
    def transform(self, X_source):
        """执行重建"""
        pass
```

## 重建方法比较

### 选择指南

```python
RECONSTRUCTION_GUIDE = {
    "通用性": {
        "SpectralDictionaryMapper": "适用于任何光谱仪",
        "SpectrumTransformerByNN": "适用于任何光谱仪",
        "conv_MatMul_recon": "仅适用于1200维卷积式",
        "conv_DFT_recon": "仅适用于1200维卷积式"
    },
    "复杂度": {
        "SpectralDictionaryMapper": "中等",
        "SpectrumTransformerByNN": "高",
        "conv_MatMul_recon": "低",
        "conv_DFT_recon": "低"
    },
    "训练需求": {
        "SpectralDictionaryMapper": "需要训练",
        "SpectrumTransformerByNN": "需要训练",
        "conv_MatMul_recon": "无需训练",
        "conv_DFT_recon": "无需训练"
    }
}
```

### 性能比较示例

```python
def compare_reconstruction_methods(X_source, X_target, X_test):
    """
    比较不同重建方法的性能
    """
    methods = {
        'Dictionary Learning': SpectralDictionaryMapper(),
        'Neural Network': SpectrumTransformerByNN(epochs=50)
    }
    
    results = {}
    
    for name, method in methods.items():
        print(f"测试方法: {name}")
        
        # 训练
        method.fit(X_source, X_target)
        
        # 重建
        X_reconstructed = method.transform(X_test)
        
        # 计算重建误差
        mse = np.mean((X_target[:len(X_test)] - X_reconstructed) ** 2)
        
        results[name] = {
            'mse': mse,
            'reconstructed': X_reconstructed
        }
    
    return results

# 使用示例
comparison_results = compare_reconstruction_methods(
    pd_samples, ft_spectra, new_pd_samples
)

for method, result in comparison_results.items():
    print(f"{method}: MSE = {result['mse']:.6f}")
```

## 最佳实践

### 1. 数据预处理

```python
def preprocess_for_reconstruction(X_source, X_target):
    """
    重建任务的数据预处理
    """
    from nirapi.preprocessing import SNV, SG
    
    # 标准化
    X_source_processed = SNV(X_source)
    X_target_processed = SNV(X_target)
    
    # 平滑
    X_source_processed = SG(X_source_processed)
    X_target_processed = SG(X_target_processed)
    
    return X_source_processed, X_target_processed
```

### 2. 重建质量评估

```python
def evaluate_reconstruction_quality(X_true, X_reconstructed):
    """
    评估重建质量
    """
    from sklearn.metrics import mean_squared_error, r2_score
    
    metrics = {}
    
    for i in range(len(X_true)):
        mse = mean_squared_error(X_true[i], X_reconstructed[i])
        r2 = r2_score(X_true[i], X_reconstructed[i])
        
        metrics[f'sample_{i}'] = {'mse': mse, 'r2': r2}
    
    # 平均指标
    avg_mse = np.mean([m['mse'] for m in metrics.values()])
    avg_r2 = np.mean([m['r2'] for m in metrics.values()])
    
    return {
        'individual_metrics': metrics,
        'average_mse': avg_mse,
        'average_r2': avg_r2
    }
```

## 相关模块

- [机器学习模块](ml_model.md) - 基础机器学习算法
- [预处理模块](preprocessing.md) - 数据预处理方法
- [可视化模块](draw.md) - 结果可视化工具
