# 基础使用教程

本教程将引导您学习 NIR API 的基础使用方法，从数据加载到模型训练的完整流程。

## 学习目标

完成本教程后，您将能够：

- 加载和处理光谱数据
- 应用基本的预处理方法
- 训练和评估机器学习模型
- 可视化分析结果

## 准备工作

### 环境设置

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 导入 NIR API 模块
from nirapi import load_data, preprocessing, ML_model, draw, analysis
```

### 数据准备

我们将使用模拟的光谱数据来演示基本功能：

```python
# 生成模拟光谱数据
np.random.seed(42)
n_samples, n_wavelengths = 200, 1200

# 模拟光谱数据（近红外波段）
wavelengths = np.linspace(1000, 2500, n_wavelengths)  # 1000-2500 nm
X = np.random.randn(n_samples, n_wavelengths) * 0.1 + 1.0

# 添加一些光谱特征
for i in range(n_samples):
    # 添加吸收峰
    X[i] += 0.5 * np.exp(-((wavelengths - 1450) / 50) ** 2)  # 水的吸收峰
    X[i] += 0.3 * np.exp(-((wavelengths - 1940) / 30) ** 2)  # 另一个吸收峰
    
# 模拟目标变量（如血糖浓度）
y = 5 + 2 * np.mean(X[:, 400:500], axis=1) + np.random.normal(0, 0.5, n_samples)

print(f"光谱数据形状: {X.shape}")
print(f"目标变量形状: {y.shape}")
print(f"波长范围: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
```

## 第一步：数据探索

### 基本统计信息

```python
# 查看数据基本信息
print("=== 数据基本信息 ===")
print(f"样本数量: {X.shape[0]}")
print(f"波长点数: {X.shape[1]}")
print(f"目标变量范围: {y.min():.2f} - {y.max():.2f}")
print(f"目标变量均值: {y.mean():.2f} ± {y.std():.2f}")

# 检查缺失值
print(f"光谱数据缺失值: {np.isnan(X).sum()}")
print(f"目标变量缺失值: {np.isnan(y).sum()}")
```

### 数据可视化

```python
# 绘制平均光谱
plt.figure(figsize=(12, 8))

# 子图1：平均光谱
plt.subplot(2, 2, 1)
mean_spectrum = np.mean(X, axis=0)
std_spectrum = np.std(X, axis=0)
plt.plot(wavelengths, mean_spectrum, 'b-', label='平均光谱')
plt.fill_between(wavelengths, 
                 mean_spectrum - std_spectrum,
                 mean_spectrum + std_spectrum,
                 alpha=0.3, label='±1σ')
plt.xlabel('波长 (nm)')
plt.ylabel('吸光度')
plt.title('平均光谱及标准差')
plt.legend()

# 子图2：几个样本的光谱
plt.subplot(2, 2, 2)
for i in range(5):
    plt.plot(wavelengths, X[i], alpha=0.7, label=f'样本 {i+1}')
plt.xlabel('波长 (nm)')
plt.ylabel('吸光度')
plt.title('样本光谱')
plt.legend()

# 子图3：目标变量分布
plt.subplot(2, 2, 3)
plt.hist(y, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('目标变量值')
plt.ylabel('频次')
plt.title('目标变量分布')

# 子图4：光谱与目标变量的关系
plt.subplot(2, 2, 4)
# 选择一个波长点与目标变量的关系
selected_wavelength_idx = 450
plt.scatter(X[:, selected_wavelength_idx], y, alpha=0.6)
plt.xlabel(f'波长 {wavelengths[selected_wavelength_idx]:.1f} nm 处的吸光度')
plt.ylabel('目标变量值')
plt.title('光谱-目标变量关系')

plt.tight_layout()
plt.show()
```

## 第二步：数据预处理

### 异常值检测

```python
# 使用马哈拉诺比斯距离检测异常值
print("=== 异常值检测 ===")
X_clean, y_clean = preprocessing.remove_top_20_percent_mahalanobis(X, y)
print(f"原始样本数: {len(X)}")
print(f"清洗后样本数: {len(X_clean)}")
print(f"移除样本数: {len(X) - len(X_clean)}")
```

### 光谱预处理

```python
# 应用多种预处理方法
print("=== 光谱预处理 ===")

# 1. 标准正态变量变换 (SNV)
X_snv = preprocessing.SNV(X_clean)
print("✓ SNV 变换完成")

# 2. Savitzky-Golay 滤波
X_sg = preprocessing.SG(X_snv, window_len=11, poly=2)
print("✓ Savitzky-Golay 滤波完成")

# 3. 均值中心化
X_mc = preprocessing.MC(X_sg)
print("✓ 均值中心化完成")

# 可视化预处理效果
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(wavelengths, X_clean[0], label='原始')
plt.plot(wavelengths, X_snv[0], label='SNV')
plt.xlabel('波长 (nm)')
plt.ylabel('吸光度')
plt.title('SNV 预处理效果')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(wavelengths, X_snv[0], label='SNV')
plt.plot(wavelengths, X_sg[0], label='SNV + SG')
plt.xlabel('波长 (nm)')
plt.ylabel('吸光度')
plt.title('Savitzky-Golay 滤波效果')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(wavelengths, X_sg[0], label='SNV + SG')
plt.plot(wavelengths, X_mc[0], label='SNV + SG + MC')
plt.xlabel('波长 (nm)')
plt.ylabel('吸光度')
plt.title('均值中心化效果')
plt.legend()

plt.tight_layout()
plt.show()

# 使用预处理后的数据
X_processed = X_mc
y_processed = y_clean
```

## 第三步：数据分割

```python
# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_processed, 
    test_size=0.3, 
    random_state=42
)

print("=== 数据分割 ===")
print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")
print(f"特征数: {X_train.shape[1]}")
```

## 第四步：特征选择

```python
# 使用 CARS 算法进行特征选择
print("=== 特征选择 ===")
X_train_selected, selected_indices = ML_model.cars(
    X_train, y_train, 
    N=50,  # 采样次数
    f=0.8  # 每次保留的特征比例
)

# 在测试集上应用相同的特征选择
X_test_selected = X_test[:, selected_indices]

print(f"原始特征数: {X_train.shape[1]}")
print(f"选择的特征数: {len(selected_indices)}")
print(f"特征选择比例: {len(selected_indices)/X_train.shape[1]*100:.1f}%")

# 可视化选择的特征
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(wavelengths, np.mean(X_train, axis=0), 'b-', alpha=0.5, label='所有波长')
plt.scatter(wavelengths[selected_indices], 
           np.mean(X_train, axis=0)[selected_indices], 
           c='red', s=20, label='选择的波长')
plt.xlabel('波长 (nm)')
plt.ylabel('平均吸光度')
plt.title('CARS 特征选择结果')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(selected_indices, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('波长索引')
plt.ylabel('频次')
plt.title('选择特征的分布')

plt.tight_layout()
plt.show()
```

## 第五步：模型训练

### 训练多个模型

```python
print("=== 模型训练 ===")

# 定义要比较的模型
models = {
    'PLSR': ML_model.PLSR,
    'SVR': ML_model.SVR,
    'RFR': ML_model.RFR
}

model_results = {}

for model_name, model_func in models.items():
    print(f"\n训练 {model_name} 模型...")
    
    try:
        # 训练模型
        if model_name == 'PLSR':
            y_train_actual, y_test_actual, y_train_pred, y_test_pred = model_func(
                X_train_selected, X_test_selected, y_train, y_test,
                n_components=min(10, X_train_selected.shape[1])
            )
        else:
            y_train_actual, y_test_actual, y_train_pred, y_test_pred = model_func(
                X_train_selected, X_test_selected, y_train, y_test
            )
        
        # 计算性能指标
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
        train_r2 = r2_score(y_train_actual, y_train_pred)
        test_r2 = r2_score(y_test_actual, y_test_pred)
        
        model_results[model_name] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': (y_test_actual, y_test_pred)
        }
        
        print(f"  训练集 - RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"  测试集 - RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
        
    except Exception as e:
        print(f"  {model_name} 训练失败: {e}")

# 选择最佳模型
best_model = min(model_results.keys(), 
                key=lambda x: model_results[x]['test_rmse'])
print(f"\n最佳模型: {best_model}")
print(f"测试集 RMSE: {model_results[best_model]['test_rmse']:.4f}")
print(f"测试集 R²: {model_results[best_model]['test_r2']:.4f}")
```

## 第六步：结果可视化

```python
# 可视化模型性能比较
plt.figure(figsize=(15, 10))

# 子图1：模型性能比较
plt.subplot(2, 3, 1)
model_names = list(model_results.keys())
test_rmse_values = [model_results[name]['test_rmse'] for name in model_names]
test_r2_values = [model_results[name]['test_r2'] for name in model_names]

x_pos = np.arange(len(model_names))
plt.bar(x_pos, test_rmse_values, alpha=0.7)
plt.xlabel('模型')
plt.ylabel('测试集 RMSE')
plt.title('模型 RMSE 比较')
plt.xticks(x_pos, model_names)

plt.subplot(2, 3, 2)
plt.bar(x_pos, test_r2_values, alpha=0.7, color='orange')
plt.xlabel('模型')
plt.ylabel('测试集 R²')
plt.title('模型 R² 比较')
plt.xticks(x_pos, model_names)

# 子图3-5：各模型的预测结果
for i, (model_name, results) in enumerate(model_results.items()):
    plt.subplot(2, 3, i+3)
    y_true, y_pred = results['predictions']
    
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    # 绘制理想预测线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'{model_name} 预测结果\nR² = {results["test_r2"]:.3f}')
    
    # 添加统计信息
    plt.text(0.05, 0.95, f'RMSE = {results["test_rmse"]:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.show()
```

## 第七步：模型解释

```python
# 分析最佳模型的特征重要性（以选择的波长为例）
print("=== 模型解释 ===")

# 计算每个选择波长的重要性（简单的相关性分析）
feature_importance = []
for idx in selected_indices:
    correlation = np.corrcoef(X_train[:, idx], y_train)[0, 1]
    feature_importance.append(abs(correlation))

# 找出最重要的波长
top_n = 10
top_indices = np.argsort(feature_importance)[-top_n:]
top_wavelengths = wavelengths[selected_indices[top_indices]]
top_importance = np.array(feature_importance)[top_indices]

print(f"最重要的 {top_n} 个波长:")
for i, (wl, imp) in enumerate(zip(top_wavelengths, top_importance)):
    print(f"  {i+1}. {wl:.1f} nm (重要性: {imp:.3f})")

# 可视化特征重要性
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(len(feature_importance)), feature_importance, alpha=0.7)
plt.xlabel('特征索引')
plt.ylabel('重要性 (|相关系数|)')
plt.title('所有选择特征的重要性')

plt.subplot(1, 2, 2)
plt.barh(range(top_n), top_importance)
plt.yticks(range(top_n), [f'{wl:.1f} nm' for wl in top_wavelengths])
plt.xlabel('重要性 (|相关系数|)')
plt.title(f'前 {top_n} 个重要波长')

plt.tight_layout()
plt.show()
```

## 总结

在本教程中，我们学习了：

1. **数据探索**: 了解光谱数据的基本特征和分布
2. **数据预处理**: 应用 SNV、SG 滤波和均值中心化
3. **异常值处理**: 使用马哈拉诺比斯距离检测异常值
4. **特征选择**: 使用 CARS 算法选择重要波长
5. **模型训练**: 比较 PLSR、SVR 和随机森林模型
6. **结果评估**: 使用 RMSE 和 R² 评估模型性能
7. **模型解释**: 分析重要特征和模型可解释性

### 关键要点

- **预处理很重要**: 适当的预处理可以显著提高模型性能
- **特征选择有效**: CARS 算法能有效减少特征维度并提高模型性能
- **模型选择**: 不同模型适用于不同类型的数据和问题
- **交叉验证**: 在实际应用中应该使用交叉验证来更可靠地评估模型

### 下一步

- 学习更高级的预处理方法
- 探索自动机器学习功能
- 了解光谱重建技术
- 学习更复杂的可视化方法

继续学习：
- [光谱重建教程](spectrum-reconstruction.md)
- [机器学习建模教程](machine-learning.md)
- [数据可视化教程](visualization.md)
