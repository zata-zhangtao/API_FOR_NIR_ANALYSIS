# 工具模块 (utils)

工具模块提供了自动机器学习、超参数优化、模型评估和其他实用工具函数，是 NIR API 的核心功能模块之一。

## 模块导入

```python
from nirapi import utils
# 或者导入特定函数
from nirapi.utils import train_model_for_trick_game_v2, run_optuna_v5, get_MZI_bands
```

## 自动机器学习

### train_model_for_trick_game_v2

自动机器学习主函数，根据输入数据自动选择最佳的预处理方法和模型。

```python
def train_model_for_trick_game_v2(splited_data, max_attempts=10, test_size=0.34, 
                                 n_trials=50, selected_metric="rmse", 
                                 target_score=0.5, filename=None, **kwargs):
    """
    自动机器学习训练和优化
    
    Parameters:
    -----------
    splited_data : tuple
        (X_train, X_test, y_train, y_test) 已分割的数据
    max_attempts : int
        最大尝试次数
    test_size : float
        测试集比例
    n_trials : int
        Optuna 优化试验次数
    selected_metric : str
        优化目标指标，支持 "rmse", "r2", "mae"
    target_score : float
        目标分数阈值
    filename : str, optional
        结果保存文件名
    **kwargs : dict
        其他配置参数
        
    Returns:
    --------
    dict
        训练结果和最佳配置
    """
```

**示例:**

```python
import numpy as np
from sklearn.model_selection import train_test_split
import datetime

# 准备数据
X = np.random.randn(200, 1200)
y = np.random.uniform(0, 10, 200)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义搜索空间
config = {
    "selected_outlier": ["不做异常值去除", "mahalanobis"],
    "selected_preprocess": ["SNV", "MSC", "SG"],
    "selected_feat_sec": ["cars", "pca", "spa"],
    "selected_model": ["PLSR", "SVR", "RFR"]
}

# 运行自动机器学习
result = utils.train_model_for_trick_game_v2(
    splited_data=(X_train, X_test, y_train, y_test),
    max_attempts=5,
    n_trials=30,
    selected_metric="rmse",
    target_score=0.5,
    filename=datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_automl",
    **config
)

print(f"最佳配置: {result['best_config']}")
print(f"最佳性能: {result['best_score']}")
```

### run_optuna_v5

使用 Optuna 进行超参数优化。

```python
def run_optuna_v5(X_train, X_test, y_train, y_test, n_trials=100, 
                 study_name="optuna_study", direction="minimize"):
    """
    Optuna 超参数优化
    
    Parameters:
    -----------
    X_train, X_test : array-like
        训练和测试特征数据
    y_train, y_test : array-like
        训练和测试目标数据
    n_trials : int
        优化试验次数
    study_name : str
        研究名称
    direction : str
        优化方向，"minimize" 或 "maximize"
        
    Returns:
    --------
    dict
        优化结果
    """
```

**示例:**

```python
# 运行 Optuna 优化
optuna_result = utils.run_optuna_v5(
    X_train, X_test, y_train, y_test,
    n_trials=50,
    study_name="glucose_prediction",
    direction="minimize"
)

print(f"最佳参数: {optuna_result['best_params']}")
print(f"最佳分数: {optuna_result['best_value']}")
```

### rebuild_model_v2

根据保存的配置重建模型。

```python
def rebuild_model_v2(config_file, X_train, X_test, y_train, y_test):
    """
    根据配置文件重建模型
    
    Parameters:
    -----------
    config_file : str
        配置文件路径
    X_train, X_test : array-like
        训练和测试数据
    y_train, y_test : array-like
        训练和测试标签
        
    Returns:
    --------
    dict
        重建结果
    """
```

## 模型评估工具

### PCA_LR_SVR_train_and_eval

使用 PCA + 线性回归/SVR 进行训练和评估。

```python
def PCA_LR_SVR_train_and_eval(X_train, X_test, y_train, y_test, 
                             n_components=50, model_type="LR"):
    """
    PCA + 回归模型训练评估
    
    Parameters:
    -----------
    X_train, X_test : array-like
        训练和测试特征
    y_train, y_test : array-like
        训练和测试标签
    n_components : int
        PCA 主成分数量
    model_type : str
        模型类型，"LR" 或 "SVR"
        
    Returns:
    --------
    dict
        评估结果
    """
```

### RF_LR_SVR_train_and_eval

随机森林特征选择 + 回归模型评估。

```python
def RF_LR_SVR_train_and_eval(X_train, X_test, y_train, y_test, 
                            n_features=100, model_type="LR"):
    """
    随机森林特征选择 + 回归模型
    
    Parameters:
    -----------
    X_train, X_test : array-like
        训练和测试数据
    y_train, y_test : array-like
        训练和测试标签
    n_features : int
        选择的特征数量
    model_type : str
        模型类型
        
    Returns:
    --------
    dict
        评估结果
    """
```

## 波长管理工具

### get_MZI_bands

获取 MZI（马赫-曾德尔干涉仪）光谱仪的波段信息。

```python
def get_MZI_bands():
    """
    获取 MZI 光谱仪波段信息
    
    Returns:
    --------
    list
        波段范围列表，包含四个波段
    """
```

**示例:**

```python
# 获取 MZI 波段
mzi_bands = utils.get_MZI_bands()
print(f"MZI 波段数量: {len(mzi_bands)}")
for i, band in enumerate(mzi_bands):
    print(f"波段 {i+1}: {band[0]:.1f} - {band[1]:.1f} nm")
```

### get_wavelength_ranges

获取指定光谱仪的波长范围。

```python
def get_wavelength_ranges(spectrometer_type):
    """
    获取光谱仪波长范围
    
    Parameters:
    -----------
    spectrometer_type : str
        光谱仪类型
        
    Returns:
    --------
    dict
        波长范围信息
    """
```

### validate_wavelength_range

验证波长范围的有效性。

```python
def validate_wavelength_range(wavelengths, expected_range=None):
    """
    验证波长范围
    
    Parameters:
    -----------
    wavelengths : array-like
        波长数组
    expected_range : tuple, optional
        期望的波长范围 (min, max)
        
    Returns:
    --------
    bool
        验证结果
    """
```

## 高级评估工具

### tpot_auto_tune

使用 TPOT 进行自动机器学习。

```python
def tpot_auto_tune(X_train, X_test, y_train, y_test, generations=5, 
                  population_size=20, cv=5):
    """
    TPOT 自动机器学习
    
    Parameters:
    -----------
    X_train, X_test : array-like
        训练和测试数据
    y_train, y_test : array-like
        训练和测试标签
    generations : int
        进化代数
    population_size : int
        种群大小
    cv : int
        交叉验证折数
        
    Returns:
    --------
    dict
        TPOT 优化结果
    """
```

### run_regression_optuna_v3

专门用于回归任务的 Optuna 优化。

```python
def run_regression_optuna_v3(X_train, X_test, y_train, y_test, 
                           model_types=None, n_trials=100):
    """
    回归任务 Optuna 优化
    
    Parameters:
    -----------
    X_train, X_test : array-like
        训练和测试数据
    y_train, y_test : array-like
        训练和测试标签
    model_types : list, optional
        要优化的模型类型列表
    n_trials : int
        优化试验次数
        
    Returns:
    --------
    dict
        优化结果
    """
```

## 实用工具函数

### train_pred_with_bands

使用指定波段进行训练和预测。

```python
def train_pred_with_bands(X_train, X_test, y_train, y_test, 
                         band_indices, model_type="PLSR"):
    """
    使用指定波段训练模型
    
    Parameters:
    -----------
    X_train, X_test : array-like
        训练和测试数据
    y_train, y_test : array-like
        训练和测试标签
    band_indices : list
        波段索引
    model_type : str
        模型类型
        
    Returns:
    --------
    dict
        训练和预测结果
    """
```

## 配置管理

### 默认配置

```python
DEFAULT_CONFIG = {
    "outlier_methods": ["不做异常值去除", "mahalanobis"],
    "preprocess_methods": ["SNV", "MSC", "SG", "normalization"],
    "feature_selection": ["cars", "spa", "pca", "random_select"],
    "models": ["PLSR", "SVR", "RFR", "XGBoost"],
    "metrics": ["rmse", "r2", "mae"]
}
```

### 自定义配置示例

```python
# 血糖检测专用配置
GLUCOSE_CONFIG = {
    "selected_outlier": ["mahalanobis"],
    "selected_preprocess": ["SNV", "SG"],
    "preprocess_number_input": 11,  # SG 窗口大小
    "selected_feat_sec": ["cars"],
    "selected_dim_red": ["pca"],
    "selected_model": ["PLSR", "SVR"]
}

# 分类任务配置
CLASSIFICATION_CONFIG = {
    "selected_outlier": ["不做异常值去除"],
    "selected_preprocess": ["normalization", "MC"],
    "selected_feat_sec": ["pca"],
    "selected_model": ["SVM", "RandomForest"]
}
```

## 批处理工具

### batch_model_evaluation

批量模型评估。

```python
def batch_model_evaluation(datasets, models, configs, output_dir="batch_results"):
    """
    批量模型评估
    
    Parameters:
    -----------
    datasets : list
        数据集列表
    models : list
        模型列表
    configs : list
        配置列表
    output_dir : str
        输出目录
        
    Returns:
    --------
    dict
        批量评估结果
    """
    import os
    import json
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for i, (dataset_name, (X_train, X_test, y_train, y_test)) in enumerate(datasets):
        print(f"处理数据集: {dataset_name}")
        
        for config in configs:
            result = train_model_for_trick_game_v2(
                splited_data=(X_train, X_test, y_train, y_test),
                filename=f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                **config
            )
            
            results[f"{dataset_name}_{i}"] = result
    
    # 保存结果
    with open(f"{output_dir}/batch_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results
```

## 性能监控

### monitor_training_progress

监控训练进度。

```python
def monitor_training_progress(study, trial_callback=None):
    """
    监控 Optuna 训练进度
    
    Parameters:
    -----------
    study : optuna.Study
        Optuna 研究对象
    trial_callback : callable, optional
        每次试验后的回调函数
    """
    def callback(study, trial):
        print(f"Trial {trial.number}: {trial.value}")
        if trial_callback:
            trial_callback(study, trial)
    
    return callback
```

## 最佳实践

### 1. 自动机器学习工作流程

```python
def complete_automl_workflow(X, y, test_size=0.3, config=None):
    """
    完整的自动机器学习工作流程
    """
    from sklearn.model_selection import train_test_split
    
    # 1. 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # 2. 使用默认配置（如果未提供）
    if config is None:
        config = {
            "selected_outlier": ["不做异常值去除"],
            "selected_preprocess": ["SNV", "SG"],
            "selected_feat_sec": ["cars"],
            "selected_model": ["PLSR", "SVR"]
        }
    
    # 3. 运行自动机器学习
    result = utils.train_model_for_trick_game_v2(
        splited_data=(X_train, X_test, y_train, y_test),
        max_attempts=10,
        n_trials=50,
        **config
    )
    
    return result
```

### 2. 模型比较

```python
def compare_multiple_configs(X, y, configs, n_runs=3):
    """
    比较多个配置的性能
    """
    results = {}
    
    for config_name, config in configs.items():
        print(f"测试配置: {config_name}")
        
        run_results = []
        for run in range(n_runs):
            result = complete_automl_workflow(X, y, config=config)
            run_results.append(result['best_score'])
        
        results[config_name] = {
            'mean_score': np.mean(run_results),
            'std_score': np.std(run_results),
            'all_scores': run_results
        }
    
    return results
```

## 相关模块

- [机器学习模块](ml_model.md) - 机器学习算法
- [预处理模块](preprocessing.md) - 数据预处理方法
- [分析模块](analysis.md) - 数据分析工具
