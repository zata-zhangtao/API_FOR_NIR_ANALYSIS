# 完整工作流程示例

本示例展示了使用 NIR API 进行近红外光谱分析的完整工作流程，从数据加载到模型部署的全过程。

## 项目背景

假设我们要开发一个血糖检测系统，使用近红外光谱技术预测血糖浓度。我们有一批从不同志愿者收集的光谱数据和对应的血糖值。

## 完整代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 导入 NIR API
from nirapi import load_data, preprocessing, ML_model, draw, analysis, utils
from nirapi.AnalysisClass.Create_rec_task import SpectrumModelEvaluator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=== NIR API 血糖检测完整工作流程 ===")
print(f"开始时间: {datetime.datetime.now()}")
```

## 步骤1: 数据加载

```python
print("\n1. 数据加载")
print("-" * 50)

# 方法1: 从数据库加载（如果有数据库）
try:
    dataset = load_data.get_dataset_from_mysql(
        database='光谱数据库',
        table_name="卷积式_v1",
        project_name="血糖检测项目",
        X_type=['光谱', '采集日期', '志愿者'],
        y_type=['血糖值'],
        start_time="2024-01-01 00:00:00",
        end_time="2024-12-31 23:59:59"
    )
    
    X = dataset['光谱']
    y = dataset['血糖值']
    dates = dataset['采集日期']
    volunteers = dataset['志愿者']
    
    print(f"✓ 从数据库加载数据成功")
    print(f"  样本数: {len(X)}")
    print(f"  志愿者数: {len(np.unique(volunteers))}")
    
except Exception as e:
    print(f"✗ 数据库加载失败: {e}")
    print("  使用模拟数据...")
    
    # 方法2: 生成模拟数据
    np.random.seed(42)
    n_samples, n_wavelengths = 300, 1200
    
    # 模拟光谱数据
    wavelengths = np.linspace(1000, 2500, n_wavelengths)
    X = np.random.randn(n_samples, n_wavelengths) * 0.05 + 1.0
    
    # 添加血糖相关的光谱特征
    glucose_absorption_bands = [1450, 1940, 2100, 2270]  # 血糖相关吸收带
    for i, band in enumerate(glucose_absorption_bands):
        band_idx = np.argmin(np.abs(wavelengths - band))
        for j in range(n_samples):
            X[j] += 0.1 * (i + 1) * np.exp(-((wavelengths - band) / 30) ** 2)
    
    # 模拟血糖值 (4-15 mmol/L)
    y = 4 + 8 * np.random.random(n_samples) + \
        2 * np.mean(X[:, 400:500], axis=1) + \
        np.random.normal(0, 0.3, n_samples)
    
    # 模拟其他信息
    volunteers = [f"志愿者_{i//10 + 1}" for i in range(n_samples)]
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='H')
    
    print(f"✓ 生成模拟数据")
    print(f"  样本数: {n_samples}")
    print(f"  波长点数: {n_wavelengths}")
    print(f"  血糖范围: {y.min():.1f} - {y.max():.1f} mmol/L")

# 保存波长信息
wavelength_info = {
    'wavelengths': wavelengths,
    'n_wavelengths': len(wavelengths),
    'range': f"{wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm"
}
```

## 步骤2: 数据探索分析

```python
print("\n2. 数据探索分析")
print("-" * 50)

# 基本统计信息
print(f"数据维度: {X.shape}")
print(f"血糖值统计: {y.mean():.2f} ± {y.std():.2f} mmol/L")
print(f"志愿者数量: {len(np.unique(volunteers))}")

# 使用 analysis 模块进行综合分析
try:
    # 创建临时 DataFrame 用于分析
    temp_df = pd.DataFrame(X)
    temp_df['血糖值'] = y
    
    # 基本数据信息
    analysis.print_basic_data_info(temp_df, X, y)
    
    # 绘制光谱概览
    analysis.plot_spectral_overview(X)
    
    print("✓ 数据探索分析完成")
    
except Exception as e:
    print(f"✗ 数据分析失败: {e}")

# 检测异常值
outlier_indices = analysis.detect_outliers(X, method='mahalanobis', threshold=95)
print(f"检测到 {len(outlier_indices)} 个异常值")
```

## 步骤3: 数据预处理

```python
print("\n3. 数据预处理")
print("-" * 50)

# 移除异常值
if len(outlier_indices) > 0:
    normal_indices = [i for i in range(len(X)) if i not in outlier_indices]
    X_clean = X[normal_indices]
    y_clean = y[normal_indices]
    volunteers_clean = [volunteers[i] for i in normal_indices]
    print(f"✓ 移除 {len(outlier_indices)} 个异常值")
else:
    X_clean, y_clean, volunteers_clean = X, y, volunteers

# 预处理流水线
preprocessing_steps = [
    ('SNV', preprocessing.SNV),
    ('SG滤波', lambda x: preprocessing.SG(x, window_len=11, poly=2)),
    ('均值中心化', preprocessing.MC)
]

X_processed = X_clean.copy()
for step_name, step_func in preprocessing_steps:
    X_processed = step_func(X_processed)
    print(f"✓ {step_name} 完成")

# 可视化预处理效果
plt.figure(figsize=(15, 5))
sample_idx = 0

plt.subplot(1, 3, 1)
plt.plot(wavelengths, X_clean[sample_idx], label='原始光谱')
plt.xlabel('波长 (nm)')
plt.ylabel('吸光度')
plt.title('原始光谱')
plt.legend()

plt.subplot(1, 3, 2)
X_temp = preprocessing.SNV(X_clean)
plt.plot(wavelengths, X_temp[sample_idx], label='SNV处理')
plt.xlabel('波长 (nm)')
plt.ylabel('吸光度')
plt.title('SNV标准化')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(wavelengths, X_processed[sample_idx], label='完整预处理')
plt.xlabel('波长 (nm)')
plt.ylabel('吸光度')
plt.title('完整预处理流水线')
plt.legend()

plt.tight_layout()
plt.show()

print(f"预处理后数据形状: {X_processed.shape}")
```

## 步骤4: 特征选择

```python
print("\n4. 特征选择")
print("-" * 50)

# 数据分割（用于特征选择）
X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
    X_processed, y_clean, test_size=0.3, random_state=42
)

# CARS 特征选择
print("执行 CARS 特征选择...")
X_train_cars, cars_indices = ML_model.cars(
    X_train_temp, y_train_temp, 
    N=50, f=0.8
)

print(f"✓ CARS 选择了 {len(cars_indices)} 个特征")
print(f"  特征选择比例: {len(cars_indices)/X_processed.shape[1]*100:.1f}%")

# 应用特征选择到所有数据
X_selected = X_processed[:, cars_indices]
selected_wavelengths = wavelengths[cars_indices]

# 可视化选择的波长
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(wavelengths, np.mean(X_processed, axis=0), 'b-', alpha=0.5, label='所有波长')
plt.scatter(selected_wavelengths, 
           np.mean(X_processed, axis=0)[cars_indices], 
           c='red', s=30, label='选择的波长', zorder=5)
plt.xlabel('波长 (nm)')
plt.ylabel('平均吸光度')
plt.title('CARS 特征选择结果')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(selected_wavelengths, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('波长 (nm)')
plt.ylabel('频次')
plt.title('选择波长的分布')

plt.tight_layout()
plt.show()

# 保存特征选择信息
feature_info = {
    'selected_indices': cars_indices,
    'selected_wavelengths': selected_wavelengths,
    'n_selected': len(cars_indices),
    'selection_ratio': len(cars_indices)/X_processed.shape[1]
}
```

## 步骤5: 模型训练和比较

```python
print("\n5. 模型训练和比较")
print("-" * 50)

# 最终数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_clean, test_size=0.3, random_state=42, stratify=None
)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# 定义模型
models_to_test = {
    'PLSR': lambda: ML_model.PLSR(X_train, X_test, y_train, y_test, n_components=15),
    'SVR': lambda: ML_model.SVR(X_train, X_test, y_train, y_test, kernel='rbf'),
    'RFR': lambda: ML_model.RFR(X_train, X_test, y_train, y_test, n_estimators=100),
    'XGBoost': lambda: ML_model.XGBoostRegression(X_train, X_test, y_train, y_test)
}

# 训练和评估模型
model_results = {}
for model_name, model_func in models_to_test.items():
    print(f"\n训练 {model_name}...")
    try:
        y_train_actual, y_test_actual, y_train_pred, y_test_pred = model_func()
        
        # 计算性能指标
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train_actual, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_actual, y_test_pred)),
            'train_r2': r2_score(y_train_actual, y_train_pred),
            'test_r2': r2_score(y_test_actual, y_test_pred),
            'test_mae': mean_absolute_error(y_test_actual, y_test_pred),
            'predictions': (y_test_actual, y_test_pred)
        }
        
        model_results[model_name] = metrics
        
        print(f"  训练集 - RMSE: {metrics['train_rmse']:.3f}, R²: {metrics['train_r2']:.3f}")
        print(f"  测试集 - RMSE: {metrics['test_rmse']:.3f}, R²: {metrics['test_r2']:.3f}")
        print(f"  测试集 - MAE: {metrics['test_mae']:.3f}")
        
    except Exception as e:
        print(f"  ✗ {model_name} 训练失败: {e}")

# 选择最佳模型
if model_results:
    best_model_name = min(model_results.keys(), 
                         key=lambda x: model_results[x]['test_rmse'])
    best_metrics = model_results[best_model_name]
    
    print(f"\n🏆 最佳模型: {best_model_name}")
    print(f"   测试集 RMSE: {best_metrics['test_rmse']:.3f} mmol/L")
    print(f"   测试集 R²: {best_metrics['test_r2']:.3f}")
    print(f"   测试集 MAE: {best_metrics['test_mae']:.3f} mmol/L")
```

## 步骤6: 自动机器学习优化

```python
print("\n6. 自动机器学习优化")
print("-" * 50)

# 使用 NIR API 的自动机器学习功能
try:
    # 定义搜索空间
    automl_config = {
        "selected_outlier": ["不做异常值去除"],
        "selected_preprocess": ["SNV", "SG"],
        "selected_feat_sec": ["cars"],
        "selected_model": ["PLSR", "SVR", "RFR"]
    }
    
    print("启动自动机器学习优化...")
    automl_result = utils.train_model_for_trick_game_v2(
        splited_data=(X_train, X_test, y_train, y_test),
        max_attempts=5,
        n_trials=20,
        selected_metric="rmse",
        target_score=0.3,
        filename=datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_glucose_automl",
        **automl_config
    )
    
    print(f"✓ 自动机器学习完成")
    print(f"  最佳配置已保存到文件")
    
except Exception as e:
    print(f"✗ 自动机器学习失败: {e}")
```

## 步骤7: 结果可视化和分析

```python
print("\n7. 结果可视化和分析")
print("-" * 50)

# 创建综合可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 模型性能比较
ax1 = axes[0, 0]
model_names = list(model_results.keys())
test_rmse_values = [model_results[name]['test_rmse'] for name in model_names]
test_r2_values = [model_results[name]['test_r2'] for name in model_names]

x_pos = np.arange(len(model_names))
bars = ax1.bar(x_pos, test_rmse_values, alpha=0.7, color='skyblue')
ax1.set_xlabel('模型')
ax1.set_ylabel('测试集 RMSE (mmol/L)')
ax1.set_title('模型性能比较 (RMSE)')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_names, rotation=45)

# 添加数值标签
for bar, value in zip(bars, test_rmse_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

# 2. R² 比较
ax2 = axes[0, 1]
bars = ax2.bar(x_pos, test_r2_values, alpha=0.7, color='lightcoral')
ax2.set_xlabel('模型')
ax2.set_ylabel('测试集 R²')
ax2.set_title('模型性能比较 (R²)')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(model_names, rotation=45)

for bar, value in zip(bars, test_r2_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

# 3. 最佳模型预测结果
ax3 = axes[0, 2]
if model_results:
    y_true, y_pred = model_results[best_model_name]['predictions']
    ax3.scatter(y_true, y_pred, alpha=0.6, color='green')
    
    # 理想预测线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    
    ax3.set_xlabel('真实血糖值 (mmol/L)')
    ax3.set_ylabel('预测血糖值 (mmol/L)')
    ax3.set_title(f'{best_model_name} 预测结果')
    
    # 添加性能指标
    ax3.text(0.05, 0.95, f'R² = {best_metrics["test_r2"]:.3f}\nRMSE = {best_metrics["test_rmse"]:.3f}',
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 4. 残差分析
ax4 = axes[1, 0]
if model_results:
    residuals = y_true - y_pred
    ax4.scatter(y_pred, residuals, alpha=0.6)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('预测值 (mmol/L)')
    ax4.set_ylabel('残差 (mmol/L)')
    ax4.set_title('残差分析')

# 5. 特征重要性（选择的波长）
ax5 = axes[1, 1]
# 计算简单的特征重要性（相关系数）
feature_importance = []
for idx in cars_indices:
    corr = np.corrcoef(X_processed[:, idx], y_clean)[0, 1]
    feature_importance.append(abs(corr))

top_n = 10
top_indices = np.argsort(feature_importance)[-top_n:]
top_wavelengths = selected_wavelengths[top_indices]
top_importance = np.array(feature_importance)[top_indices]

ax5.barh(range(top_n), top_importance, alpha=0.7)
ax5.set_yticks(range(top_n))
ax5.set_yticklabels([f'{wl:.0f} nm' for wl in top_wavelengths])
ax5.set_xlabel('重要性 (|相关系数|)')
ax5.set_title(f'前{top_n}个重要波长')

# 6. 血糖值分布
ax6 = axes[1, 2]
ax6.hist(y_clean, bins=20, alpha=0.7, edgecolor='black', color='gold')
ax6.axvline(y_clean.mean(), color='red', linestyle='--', label=f'均值: {y_clean.mean():.1f}')
ax6.set_xlabel('血糖值 (mmol/L)')
ax6.set_ylabel('频次')
ax6.set_title('血糖值分布')
ax6.legend()

plt.tight_layout()
plt.show()
```

## 步骤8: 模型保存和部署准备

```python
print("\n8. 模型保存和部署准备")
print("-" * 50)

# 保存最佳模型（这里我们需要重新训练以获得模型对象）
if model_results:
    print(f"准备保存最佳模型: {best_model_name}")
    
    # 创建模型配置
    model_config = {
        'model_name': best_model_name,
        'preprocessing_steps': ['SNV', 'SG', 'MC'],
        'feature_selection': 'CARS',
        'selected_features': cars_indices.tolist(),
        'selected_wavelengths': selected_wavelengths.tolist(),
        'performance_metrics': best_metrics,
        'training_date': datetime.datetime.now().isoformat(),
        'data_info': {
            'n_samples': len(X_clean),
            'n_features_original': X_processed.shape[1],
            'n_features_selected': len(cars_indices),
            'target_range': [float(y_clean.min()), float(y_clean.max())],
            'wavelength_range': wavelength_info['range']
        }
    }
    
    # 保存配置
    import json
    config_filename = f"glucose_model_config_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_filename, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 模型配置已保存: {config_filename}")
    
    # 保存预处理后的数据样本
    sample_data = {
        'X_sample': X_selected[:5].tolist(),  # 保存5个样本作为示例
        'y_sample': y_clean[:5].tolist(),
        'wavelengths': selected_wavelengths.tolist()
    }
    
    sample_filename = f"glucose_data_sample_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(sample_filename, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"✓ 数据样本已保存: {sample_filename}")

print(f"\n=== 工作流程完成 ===")
print(f"结束时间: {datetime.datetime.now()}")
print(f"最佳模型: {best_model_name if model_results else 'N/A'}")
print(f"最终性能: RMSE = {best_metrics['test_rmse']:.3f} mmol/L, R² = {best_metrics['test_r2']:.3f}" if model_results else "N/A")
```

## 总结

这个完整的工作流程展示了：

1. **数据管理**: 从数据库加载或生成模拟数据
2. **质量控制**: 异常值检测和数据清洗
3. **预处理**: 多步骤光谱预处理流水线
4. **特征工程**: CARS 算法进行特征选择
5. **模型开发**: 多模型比较和选择
6. **自动优化**: 使用自动机器学习进行超参数优化
7. **结果分析**: 综合性能评估和可视化
8. **模型部署**: 模型配置保存和部署准备

### 关键优势

- **完整性**: 覆盖了从数据到部署的全流程
- **自动化**: 集成了自动机器学习功能
- **可视化**: 提供了丰富的分析图表
- **可重现**: 所有步骤都有详细记录和配置保存
- **可扩展**: 易于添加新的预处理方法和模型

### 实际应用建议

1. **数据质量**: 确保光谱数据的质量和一致性
2. **样本代表性**: 收集足够多样化的样本
3. **交叉验证**: 使用更严格的验证策略
4. **模型监控**: 在生产环境中监控模型性能
5. **持续改进**: 定期更新模型和重新训练
