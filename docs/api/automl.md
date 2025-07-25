# 自动机器学习模块 (automl)

自动机器学习模块提供了基于 Optuna 的超参数优化、模型选择和训练流水线自动化功能，专门为近红外光谱数据分析设计。

## 模块导入

```python
from nirapi import automl
# 或者导入特定函数
from nirapi.automl import train_model_for_trick_game_v2, run_optuna_v5, run_regression_optuna_v3
```

## 核心功能

### train_model_for_trick_game_v2

具有迭代改进能力的自动机器学习函数，通过多次尝试达到目标性能。

::: nirapi.automl.train_model_for_trick_game_v2

**功能特点:**
- 自动超参数优化
- 迭代性能改进
- 多种评估指标支持
- 自动数据分割
- 结果报告生成

**示例:**

```python
import numpy as np
from sklearn.model_selection import train_test_split

# 准备数据
X = np.random.randn(200, 1200)  # 光谱数据
y = np.random.uniform(0, 10, 200)  # 目标变量

# 方法1: 直接传入数据，自动分割
results = automl.train_model_for_trick_game_v2(
    X=X,
    y=y,
    max_attempts=10,
    test_size=0.3,
    n_trials=200,
    selected_metric="rmse",
    target_score=0.5,
    filename="glucose_model"
)

# 方法2: 使用预分割的数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
splited_data = (X_train, X_test, y_train, y_test)

results = automl.train_model_for_trick_game_v2(
    splited_data=splited_data,
    max_attempts=5,
    n_trials=100,
    selected_metric="r2",
    target_score=0.9,
    filename="optimized_model"
)

print(f"训练状态: {results['status']}")
print(f"保存文件名: {results['filename']}")
```

**参数说明:**

- `max_attempts`: 最大优化尝试次数，当模型性能未达到目标时会重新训练
- `splited_data`: 预分割的数据元组 (X_train, X_test, y_train, y_test)
- `X`, `y`: 原始数据，如果未提供 splited_data 则使用这些数据
- `test_size`: 测试集比例，仅在使用 X, y 时有效
- `n_trials`: Optuna 优化试验次数
- `selected_metric`: 优化目标 ('rmse', 'mae', 'r2')
- `target_score`: 目标性能分数
- `filename`: 保存文件名前缀

### run_optuna_v5

基于 Optuna 的高级超参数优化函数。

```python
def run_optuna_v5(data_dict, train_key, isReg, chose_n_trails, 
                  selected_metric='rmse', save=None, save_name="", **kw):
    """
    Optuna-based hyperparameter optimization.
    
    Parameters:
    -----------
    data_dict : dict
        包含训练数据的字典
    train_key : str
        数据字典中训练数据的键名
    isReg : bool
        是否为回归任务 (True: 回归, False: 分类)
    chose_n_trails : int
        优化试验次数
    selected_metric : str
        优化指标 ('rmse', 'mae', 'r2', 'accuracy')
    save : str, optional
        保存路径
    save_name : str
        保存文件名
    **kw : dict
        其他参数选项
        
    Returns:
    --------
    dict
        优化结果和最佳参数
    """
```

**示例:**

```python
# 准备数据字典
data_dict = {
    'train_data': (X_train, X_test, y_train, y_test),
    'validation_data': (X_val, y_val)
}

# 回归任务优化
regression_results = automl.run_optuna_v5(
    data_dict=data_dict,
    train_key='train_data',
    isReg=True,
    chose_n_trails=200,
    selected_metric='rmse',
    save_name="regression_optimization"
)

# 分类任务优化
classification_results = automl.run_optuna_v5(
    data_dict=data_dict,
    train_key='train_data',
    isReg=False,
    chose_n_trails=150,
    selected_metric='accuracy',
    save_name="classification_optimization"
)
```

### run_regression_optuna_v3

专门针对回归任务的 Optuna 优化函数。

```python
def run_regression_optuna_v3(
    data_name, X=None, y=None, data_splited=None, model='PLS', 
    split='SPXY', test_size=0.3, n_trials=200, object="R2", 
    cv=None, save_dir=None, each_class_mae=False, only_train_and_val_set=False
):
    """
    Regression model optimization with Optuna.
    
    Parameters:
    -----------
    data_name : str
        数据集名称
    X, y : array-like, optional
        原始数据
    data_splited : tuple, optional
        预分割数据
    model : str
        模型类型 ('PLS', 'SVR', 'RFR', 'XGBoost')
    split : str
        数据分割方法 ('SPXY', 'random', 'stratified')
    test_size : float
        测试集比例
    n_trials : int
        优化试验次数
    object : str
        优化目标 ('R2', 'RMSE', 'MAE')
    cv : int, optional
        交叉验证折数
    save_dir : str, optional
        保存目录
    each_class_mae : bool
        是否计算每类的 MAE
    only_train_and_val_set : bool
        是否只使用训练和验证集
        
    Returns:
    --------
    dict
        回归优化结果
    """
```

**示例:**

```python
# PLS 回归优化
pls_results = automl.run_regression_optuna_v3(
    data_name="glucose_spectral_data",
    X=X,
    y=y,
    model='PLS',
    split='SPXY',
    test_size=0.3,
    n_trials=300,
    object="R2",
    cv=5,
    save_dir="optimization_results"
)

# SVR 优化
svr_results = automl.run_regression_optuna_v3(
    data_name="skin_moisture_data",
    data_splited=(X_train, X_test, y_train, y_test),
    model='SVR',
    n_trials=200,
    object="RMSE"
)

# 随机森林优化
rf_results = automl.run_regression_optuna_v3(
    data_name="protein_content",
    X=X,
    y=y,
    model='RFR',
    split='random',
    test_size=0.25,
    n_trials=150,
    object="MAE"
)
```

### tpot_auto_tune

基于 TPOT 的自动机器学习函数。

```python
def tpot_auto_tune(X, y, generations=5, population_size=20, cv=5):
    """
    TPOT-based automated machine learning.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        特征矩阵
    y : array-like, shape (n_samples,)
        目标变量
    generations : int
        遗传算法代数
    population_size : int
        种群大小
    cv : int
        交叉验证折数
        
    Returns:
    --------
    dict
        TPOT 优化结果和最佳流水线
    """
```

**示例:**

```python
# TPOT 自动优化
tpot_results = automl.tpot_auto_tune(
    X=X_train,
    y=y_train,
    generations=10,
    population_size=50,
    cv=5
)

print(f"最佳流水线: {tpot_results['best_pipeline']}")
print(f"最佳分数: {tpot_results['best_score']:.4f}")
```

### rebuild_model_v2

使用指定参数重建和重新训练模型。

```python
def rebuild_model_v2(splited_data=None, params_dict: dict = None):
    """
    Rebuild and retrain models with specified parameters.
    
    Parameters:
    -----------
    splited_data : tuple, optional
        预分割的数据
    params_dict : dict
        模型参数字典
        
    Returns:
    --------
    dict
        重建模型的结果
    """
```

**示例:**

```python
# 定义最佳参数
best_params = {
    'model_type': 'PLSR',
    'n_components': 15,
    'preprocessing': ['SNV', 'SG'],
    'feature_selection': 'CARS'
}

# 重建模型
rebuild_results = automl.rebuild_model_v2(
    splited_data=(X_train, X_test, y_train, y_test),
    params_dict=best_params
)

print(f"重建模型性能: R² = {rebuild_results['r2']:.4f}")
```

## 完整工作流程示例

### 自动化光谱分析流水线

```python
def automated_spectral_analysis_pipeline(X, y, data_name="spectral_analysis"):
    """
    完整的自动化光谱分析流水线
    """
    from sklearn.model_selection import train_test_split
    import os
    
    # 1. 数据准备和分割
    print("1. 数据准备...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 2. 自动模型优化
    print("2. 自动模型优化...")
    auto_results = automl.train_model_for_trick_game_v2(
        X=X,
        y=y,
        max_attempts=8,
        test_size=0.3,
        n_trials=200,
        selected_metric="rmse",
        target_score=0.1,
        filename=f"{data_name}_auto_model"
    )
    
    # 3. 多模型对比优化
    print("3. 多模型对比...")
    models_to_test = ['PLS', 'SVR', 'RFR', 'XGBoost']
    comparison_results = {}
    
    for model in models_to_test:
        try:
            result = automl.run_regression_optuna_v3(
                data_name=f"{data_name}_{model}",
                X=X,
                y=y,
                model=model,
                n_trials=100,
                object="R2"
            )
            comparison_results[model] = result
            print(f"   {model}: R² = {result.get('best_r2', 'N/A'):.4f}")
        except Exception as e:
            print(f"   {model}: 优化失败 - {e}")
    
    # 4. 选择最佳模型
    print("4. 选择最佳模型...")
    best_model = max(
        comparison_results.keys(),
        key=lambda k: comparison_results[k].get('best_r2', -1)
    )
    best_params = comparison_results[best_model]['best_params']
    
    # 5. 重建最佳模型
    print("5. 重建最佳模型...")
    final_model = automl.rebuild_model_v2(
        splited_data=(X_train, X_test, y_train, y_test),
        params_dict=best_params
    )
    
    # 6. 结果汇总
    results_summary = {
        'auto_optimization': auto_results,
        'model_comparison': comparison_results,
        'best_model': best_model,
        'final_model': final_model,
        'recommendations': {
            'best_algorithm': best_model,
            'performance_r2': final_model.get('r2', 'N/A'),
            'feature_importance': final_model.get('feature_importance', [])
        }
    }
    
    print(f"分析完成！最佳模型: {best_model}")
    return results_summary

# 使用示例
results = automated_spectral_analysis_pipeline(X, y, "glucose_detection")
```

### 超参数优化策略

```python
def advanced_hyperparameter_optimization(X, y, task_type='regression'):
    """
    高级超参数优化策略
    """
    
    optimization_configs = {
        'regression': {
            'models': ['PLS', 'SVR', 'RFR'],
            'metrics': ['R2', 'RMSE', 'MAE'],
            'trials': [100, 200, 150]
        },
        'classification': {
            'models': ['SVM', 'RandomForest', 'XGBoost'],
            'metrics': ['accuracy', 'f1', 'auc'],
            'trials': [150, 200, 180]
        }
    }
    
    config = optimization_configs[task_type]
    all_results = {}
    
    for model, metric, n_trials in zip(config['models'], config['metrics'], config['trials']):
        print(f"优化 {model} 模型...")
        
        if task_type == 'regression':
            result = automl.run_regression_optuna_v3(
                data_name=f"advanced_{model}",
                X=X,
                y=y,
                model=model,
                n_trials=n_trials,
                object=metric
            )
        else:
            # 分类任务的优化逻辑
            data_dict = {'data': (X, y)}
            result = automl.run_optuna_v5(
                data_dict=data_dict,
                train_key='data',
                isReg=False,
                chose_n_trails=n_trials,
                selected_metric=metric
            )
        
        all_results[f"{model}_{metric}"] = result
    
    return all_results
```

## AutoML 最佳实践

### 1. 数据准备策略

```python
def prepare_data_for_automl(X, y, validation_split=0.2):
    """
    为 AutoML 准备数据的最佳实践
    """
    from sklearn.model_selection import train_test_split
    
    # 1. 数据验证
    assert len(X) == len(y), "特征和目标变量长度不匹配"
    assert not np.isnan(X).any(), "特征数据包含缺失值"
    assert not np.isnan(y).any(), "目标变量包含缺失值"
    
    # 2. 数据分割 (训练集、验证集、测试集)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=validation_split, random_state=42
    )
    
    return {
        'train': (X_train, y_train),
        'validation': (X_val, y_val),
        'test': (X_test, y_test),
        'full_train': (X_temp, y_temp)
    }
```

### 2. 优化策略配置

```python
AUTOML_STRATEGIES = {
    'quick': {
        'n_trials': 50,
        'max_attempts': 3,
        'models': ['PLS', 'SVR']
    },
    'standard': {
        'n_trials': 200,
        'max_attempts': 5,
        'models': ['PLS', 'SVR', 'RFR']
    },
    'comprehensive': {
        'n_trials': 500,
        'max_attempts': 10,
        'models': ['PLS', 'SVR', 'RFR', 'XGBoost']
    }
}
```

### 3. 性能监控

```python
def monitor_automl_performance(results_dict, target_metrics):
    """
    监控 AutoML 性能
    """
    performance_summary = {}
    
    for model_name, results in results_dict.items():
        performance_summary[model_name] = {
            'meets_target': all(
                results.get(metric, 0) >= target_metrics[metric]
                for metric in target_metrics
            ),
            'best_score': max(results.get('scores', [0])),
            'convergence': len(results.get('optimization_history', []))
        }
    
    return performance_summary
```

## 故障排除

### 常见问题及解决方案

1. **内存不足**
   ```python
   # 减少试验次数或使用批处理
   result = automl.train_model_for_trick_game_v2(
       n_trials=50,  # 降低试验次数
       max_attempts=3  # 降低尝试次数
   )
   ```

2. **收敛慢**
   ```python
   # 调整优化策略
   result = automl.run_optuna_v5(
       chose_n_trails=100,
       selected_metric='rmse',  # 使用更敏感的指标
       early_stopping_rounds=20
   )
   ```

3. **模型性能不佳**
   ```python
   # 增加数据预处理步骤
   from nirapi.preprocessing import SNV, SG
   X_processed = SG(SNV(X))
   
   result = automl.train_model_for_trick_game_v2(
       X=X_processed,
       y=y,
       target_score=0.8  # 设置合理的目标分数
   )
   ```

## 相关模块

- [机器学习模块](ml_model.md) - 基础机器学习算法
- [预处理模块](preprocessing.md) - 数据预处理方法
- [分析模块](analysis.md) - 数据分析工具
- [工具模块](utils.md) - 辅助工具函数