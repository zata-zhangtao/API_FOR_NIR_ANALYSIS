# 机器学习模块 (ML_model)

机器学习模块提供了各种回归和分类算法、特征选择方法以及数据分割工具，专门针对光谱数据分析进行了优化。

## 模块导入

```python
from nirapi import ML_model
# 或者导入特定函数
from nirapi.ML_model import PLSR, SVR, cars, pca
```

## 回归模型

### PLSR (Partial Least Squares Regression)

偏最小二乘回归，光谱分析中最常用的回归方法。

::: nirapi.nirapi.ML_model.PLSR

**示例:**

```python
import numpy as np
from sklearn.model_selection import train_test_split

# 准备数据
X = np.random.randn(100, 1200)  # 光谱数据
y = np.random.uniform(0, 10, 100)  # 目标变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 使用 PLSR
y_train_actual, y_test_actual, y_train_pred, y_test_pred = ML_model.PLSR(
    X_train, X_test, y_train, y_test, n_components=10
)

# 计算性能指标
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
r2 = r2_score(y_test_actual, y_test_pred)
print(f"PLSR - RMSE: {rmse:.4f}, R²: {r2:.4f}")
```

### SVR (Support Vector Regression)

支持向量回归，适用于非线性关系建模。

::: nirapi.nirapi.ML_model.SVR

**示例:**

```python
# 使用 SVR
y_train_actual, y_test_actual, y_train_pred, y_test_pred = ML_model.SVR(
    X_train, X_test, y_train, y_test,
    kernel='rbf', C=1.0, gamma='scale'
)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
r2 = r2_score(y_test_actual, y_test_pred)
print(f"SVR - RMSE: {rmse:.4f}, R²: {r2:.4f}")
```

### 其他回归模型

```python
# 随机森林回归
y_train_actual, y_test_actual, y_train_pred, y_test_pred = ML_model.RFR(
    X_train, X_test, y_train, y_test, n_estimators=100
)

# XGBoost 回归
y_train_actual, y_test_actual, y_train_pred, y_test_pred = ML_model.XGBoostRegression(
    X_train, X_test, y_train, y_test
)

# 线性回归
y_train_actual, y_test_actual, y_train_pred, y_test_pred = ML_model.LR(
    X_train, X_test, y_train, y_test
)

# 贝叶斯岭回归
y_train_actual, y_test_actual, y_train_pred, y_test_pred = ML_model.BayesianRidge(
    X_train, X_test, y_train, y_test
)
```

## 分类模型

### SVM (Support Vector Machine)

支持向量机分类器。

::: nirapi.nirapi.ML_model.SVM

**示例:**

```python
# 准备分类数据
y_class = np.random.randint(0, 3, 100)  # 3类分类问题
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.3)

# 使用 SVM
y_train_actual, y_test_actual, y_train_pred, y_test_pred = ML_model.SVM(
    X_train, X_test, y_train, y_test,
    kernel='rbf', C=1.0
)

# 计算分类准确率
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test_actual, y_test_pred)
print(f"SVM 准确率: {accuracy:.4f}")
print(classification_report(y_test_actual, y_test_pred))
```

### 其他分类模型

```python
# 随机森林分类
y_train_actual, y_test_actual, y_train_pred, y_test_pred = ML_model.RandomForest(
    X_train, X_test, y_train, y_test, n_estimators=100
)

# K近邻分类
y_train_actual, y_test_actual, y_train_pred, y_test_pred = ML_model.KNN(
    X_train, X_test, y_train, y_test, n_neighbors=5
)

# XGBoost 分类
y_train_actual, y_test_actual, y_train_pred, y_test_pred = ML_model.XGBoost(
    X_train, X_test, y_train, y_test
)
```

## 特征选择

### CARS (Competitive Adaptive Reweighted Sampling)

竞争性自适应重加权采样，光谱特征选择的经典方法。

::: nirapi.nirapi.ML_model.cars

**示例:**

```python
# 使用 CARS 进行特征选择
X_selected, selected_indices = ML_model.cars(
    X_train, y_train, 
    N=50,  # 采样次数
    f=0.8  # 每次采样保留的特征比例
)

print(f"原始特征数: {X_train.shape[1]}")
print(f"选择的特征数: {X_selected.shape[1]}")
print(f"选择的特征索引: {selected_indices[:10]}...")  # 显示前10个

# 在测试集上应用相同的特征选择
X_test_selected = X_test[:, selected_indices]
```

### SPA (Successive Projections Algorithm)

连续投影算法，另一种有效的特征选择方法。

::: nirapi.nirapi.ML_model.spa

**示例:**

```python
# 使用 SPA 进行特征选择
X_selected, selected_indices = ML_model.spa(
    X_train, y_train,
    m_max=50  # 最大选择特征数
)

print(f"SPA 选择的特征数: {len(selected_indices)}")
```

### 其他特征选择方法

```python
# 相关系数特征选择
X_selected, selected_indices = ML_model.corr_coefficient(
    X_train, y_train, threshold=0.5
)

# ANOVA F检验特征选择
X_selected, selected_indices = ML_model.anova(
    X_train, y_train, k=100  # 选择前100个特征
)

# PCA 降维
X_pca, pca_model = ML_model.pca(
    X_train, n_components=50
)
X_test_pca = pca_model.transform(X_test)
```

## 数据分割和预处理

### 异常值检测

```python
# 马哈拉诺比斯距离异常值检测
X_clean, y_clean = ML_model.mahalanobis(
    X_train, y_train, threshold=95
)

print(f"原始样本数: {len(X_train)}")
print(f"清洗后样本数: {len(X_clean)}")
```

### 数据分割

```python
# 自定义训练测试分割
X_train_new, X_test_new, y_train_new, y_test_new = ML_model.custom_train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 随机分割
X_train_rand, X_test_rand, y_train_rand, y_test_rand = ML_model.random_split(
    X, y, test_ratio=0.3
)
```

## 模型组合和集成

### 模型比较

```python
def compare_models(X_train, X_test, y_train, y_test):
    """
    比较多个模型的性能
    """
    models = {
        'PLSR': ML_model.PLSR,
        'SVR': ML_model.SVR,
        'RFR': ML_model.RFR,
        'XGBoost': ML_model.XGBoostRegression
    }
    
    results = {}
    
    for name, model_func in models.items():
        try:
            y_train_actual, y_test_actual, y_train_pred, y_test_pred = model_func(
                X_train, X_test, y_train, y_test
            )
            
            rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
            r2 = r2_score(y_test_actual, y_test_pred)
            
            results[name] = {'RMSE': rmse, 'R²': r2}
            
        except Exception as e:
            print(f"{name} 模型训练失败: {e}")
    
    return results

# 运行模型比较
model_results = compare_models(X_train, X_test, y_train, y_test)
for model, metrics in model_results.items():
    print(f"{model}: RMSE={metrics['RMSE']:.4f}, R²={metrics['R²']:.4f}")
```

### 特征选择 + 模型训练流水线

```python
def feature_selection_pipeline(X_train, X_test, y_train, y_test):
    """
    特征选择 + 模型训练的完整流水线
    """
    # 1. CARS 特征选择
    X_train_cars, cars_indices = ML_model.cars(X_train, y_train, N=50)
    X_test_cars = X_test[:, cars_indices]
    
    # 2. 在选择的特征上训练 PLSR
    y_train_actual, y_test_actual, y_train_pred, y_test_pred = ML_model.PLSR(
        X_train_cars, X_test_cars, y_train, y_test, n_components=10
    )
    
    # 3. 评估性能
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    r2 = r2_score(y_test_actual, y_test_pred)
    
    return {
        'selected_features': cars_indices,
        'n_features': len(cars_indices),
        'rmse': rmse,
        'r2': r2,
        'predictions': (y_test_actual, y_test_pred)
    }

# 运行流水线
pipeline_result = feature_selection_pipeline(X_train, X_test, y_train, y_test)
print(f"特征选择后性能: RMSE={pipeline_result['rmse']:.4f}, R²={pipeline_result['r2']:.4f}")
```

## 超参数优化

### 网格搜索示例

```python
def optimize_plsr_components(X_train, X_test, y_train, y_test):
    """
    优化 PLSR 的主成分数量
    """
    best_score = np.inf
    best_n_components = None
    
    for n_components in range(1, min(50, X_train.shape[1])):
        try:
            y_train_actual, y_test_actual, y_train_pred, y_test_pred = ML_model.PLSR(
                X_train, X_test, y_train, y_test, n_components=n_components
            )
            
            rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
            
            if rmse < best_score:
                best_score = rmse
                best_n_components = n_components
                
        except Exception as e:
            continue
    
    return best_n_components, best_score

# 优化 PLSR 参数
best_components, best_rmse = optimize_plsr_components(X_train, X_test, y_train, y_test)
print(f"最佳主成分数: {best_components}, 最佳 RMSE: {best_rmse:.4f}")
```

## 模型评估工具

### 交叉验证

```python
from sklearn.model_selection import cross_val_score
from sklearn.cross_decomposition import PLSRegression

def cross_validate_plsr(X, y, n_components=10, cv=5):
    """
    PLSR 交叉验证
    """
    model = PLSRegression(n_components=n_components)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    
    return {
        'mean_rmse': np.mean(rmse_scores),
        'std_rmse': np.std(rmse_scores),
        'scores': rmse_scores
    }

# 运行交叉验证
cv_results = cross_validate_plsr(X_train, y_train)
print(f"交叉验证 RMSE: {cv_results['mean_rmse']:.4f} ± {cv_results['std_rmse']:.4f}")
```

## 最佳实践

### 1. 模型选择指南

```python
MODEL_SELECTION_GUIDE = {
    '线性关系': ['PLSR', 'LR', 'BayesianRidge'],
    '非线性关系': ['SVR', 'RFR', 'XGBoost'],
    '高维数据': ['PLSR', 'SVR'],
    '小样本': ['PLSR', 'BayesianRidge'],
    '大样本': ['XGBoost', 'RFR'],
    '分类任务': ['SVM', 'RandomForest', 'XGBoost']
}
```

### 2. 特征选择策略

```python
FEATURE_SELECTION_STRATEGY = {
    '光谱数据': ['CARS', 'SPA'],
    '高维数据': ['PCA', 'CARS'],
    '相关性分析': ['corr_coefficient'],
    '统计检验': ['anova'],
    '降维': ['PCA']
}
```

### 3. 性能评估

```python
def comprehensive_evaluation(y_true, y_pred):
    """
    综合性能评估
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    
    return metrics
```

## 相关模块

- [预处理模块](preprocessing.md) - 数据预处理方法
- [可视化模块](draw.md) - 模型结果可视化
- [工具模块](utils.md) - 自动机器学习工具
