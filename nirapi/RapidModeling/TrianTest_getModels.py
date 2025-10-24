import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
import warnings
import joblib

# 尝试导入 XGBoost（可选）
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

def train_and_evaluate_models(X_train, y_train, X_test, y_test, output_dir="./train_test_results_and_models"):
    """
    训练多个回归模型，评估性能，并保存结果图和模型文件。

    Parameters:
    - X_train, y_train: 训练集
    - X_test, y_test: 测试集
    - output_dir: 输出目录（用于保存图片和 .pkl 模型）
    """
    # 忽略收敛警告（如 SVR 可能出现）
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义模型字典
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        # 训练
        model.fit(X_train, y_train)

        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 计算指标
        def calc_metrics(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            return mae, rmse, r2

        train_mae, train_rmse, train_r2 = calc_metrics(y_train, y_train_pred)
        test_mae, test_rmse, test_r2 = calc_metrics(y_test, y_test_pred)

        results.append({
            "Model": name,
            "Train_MAE": train_mae,
            "Test_MAE": test_mae,
            "Train_RMSE": train_rmse,
            "Test_RMSE": test_rmse,
            "Train_R2": train_r2,
            "Test_R2": test_r2,
        })

        # 保存模型
        model_path = os.path.join(output_dir, f"{name.replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

    # 转换为 DataFrame
    df_results = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(df_results.to_string(index=False))

    # 可视化
    metrics = ["MAE", "RMSE", "R2"]
    n_metrics = len(metrics)
    n_models = len(models)

    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_models)
    width = 0.35

    # ---------- 柱状图 ----------
    fig_bar, axes_bar = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes_bar = [axes_bar]
    for i, metric in enumerate(metrics):
        train_vals = df_results[f"Train_{metric}"].values
        test_vals = df_results[f"Test_{metric}"].values

        axes_bar[i].bar(x - width/2, train_vals, width, label='Train', alpha=0.8)
        axes_bar[i].bar(x + width/2, test_vals, width, label='Test', alpha=0.8)
        axes_bar[i].set_title(f'{metric} Comparison')
        axes_bar[i].set_xticks(x)
        axes_bar[i].set_xticklabels([name[:15] for name in df_results["Model"]], rotation=45, ha='right')
        axes_bar[i].legend()
        axes_bar[i].grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path_bar = os.path.join(output_dir, "model_comparison_bar.png")
    plt.savefig(plot_path_bar, dpi=300, bbox_inches='tight')
    print(f"\nBar plot saved to {plot_path_bar}")
    plt.show()
    
    # ---------- 每个模型的散点图 ----------
    for name, model in models.items():
        # 模型名用于保存
        sanitized_name = name.replace(' ', '_')
        # 预测值
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        plt.figure(figsize=(7, 6))
        plt.scatter(y_train, y_train_pred, c='deepskyblue', label='Train', alpha=0.7)
        plt.scatter(y_test, y_test_pred, c='orangered', label='Test', alpha=0.7, marker='^')
        # 画对角线
        all_y = np.concatenate([y_train, y_test])
        y_min, y_max = np.min(all_y), np.max(all_y)
        plt.plot([y_min, y_max], [y_min, y_max], 'k--')
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title(f'{name} Predicted vs True')
        plt.legend()
        plt.grid(alpha=0.4)
        scatter_path = os.path.join(output_dir, f"{sanitized_name}_scatter.png")
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Scatter plot for {name} saved to {scatter_path}")

    return df_results