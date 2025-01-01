from typing import Union
'''绘图模块包含了绘制光谱吸收图的函数
---------
Functions:
---------  
    - plotly_simple_chart(data, x_axis=None, x_tick_interval=100, x_title="Time", y_title="PD Sample Value", title="Sample Data Over Time", template="plotly_white", tick_angle=90) 
    - analyze_model_performance(y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred) 
    - plot_actual_vs_predicted(X_axis_train, y_train, y_train_pred, X_axis_val, y_val, y_val_pred, X_axis_test, y_test, y_test_pred)
    - plot_pca_with_class_distribution(X_train, X_val, X_test, y_train, y_val, y_test, n_components=3) 
    - plot_mean(data_name,X_name,y_name, data_X, data_y, scale=True,draw_y=True, save_dir=None)  
    - spectral_absorption(X,y,category="all_samples",wave = None,plt_show = False)
    - Sample_spectral_ranking(X,y,class="all_samples")
    - Numerical_distribution(ndarr,class="all_samples",feat_ = None)
    - RankingMAE_and_Spearman_between_X_and_Y(X,y,class = "all_samples")
    - Comparison_of_data_before_and_after(before_X,before_y,after_X,after_y,class = "all_samples")
    - train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,class = "all_samples")
    - classification_report_plot(y_true,y_pred)
    - train_and_test_pred_plot(y_train,y_test,y_pred_train,y_pred_test,data_name)
    - train_val_and_test_pred_plot(y_train,y_val,y_test,y_pred_train,y_pred_val,y_pred_test, data_name="",save_dir=None,each_class_mae=True,content = "") 
    - pred_plot(y_test,y_pred_test, data_name="",save_dir=None,each_class_mae=True,content = "") 
    - plot_multiclass_line(data_name,X,y,save_dir=None)

---------
Example:
---------
    - 使用plotly画简单折线图  plotly_simple_chart
    - 自动判断任务类型并输出分析报告  analyze_model_performance
    - 画训练集验证集测试集的图，横坐标我一般设置成时间顺序，也可以设置其他，可以画散点图或者是折线图    plot_actual_vs_predicted
    - 画PCA降维图,传入训练集验证集和验证集  plot_pca_with_class_distribution
    - 画数据集的均值变化图  plot_mean
    - 画光谱吸收图  spectral_absorption
    - 画光谱吸收排序图  Sample_spectral_ranking
    - 画数值分布图  Numerical_distribution
    - 画X和Y之间的排名MAE和spearman相关系数  RankingMAE_and_Spearman_between_X_and_Y
    - 画数据处理前后的spearman图  Comparison_of_data_before_and_after
    - 画训练集和测试集的散点图  train_test_scatter
    - 画分类任务中的分类报告图  classification_report_plot
    - 画回归任务中的散点图用plotly画  train_and_test_pred_plot
    - 画训练验证测试集的散点图最好用在回归任务中  train_val_and_test_pred_plot
    - 画真实值和预测值的散点图-只有一组数据集时候用  pred_plot
    - 根据Y的值设置不同颜色画折线图  plot_multiclass_line


    
    

---------

'''


import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 解决中文显示问题
plt.rcParams['font.family'] = 'SimHei'  # 替换为你选择的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
# wavelengths = pd.read_csv(r"C:\Users\zata\AppData\Local\Programs\Python\Python310\Lib\site-packages\nirapi\Alcohol.csv").columns[:1899].values.astype("float")
# wavelengths = pd.read_csv(r"C:\Users\zata\A ppData\Local\Programs\Python\Python310\Lib\site-packages\nirapi\Alcohol.csv").columns[:1899].values.astype("float")
from scipy.stats import pearsonr, spearmanr

# 血糖的克拉克图
def clarke_error_grid(reference, prediction):
    """
    绘制Clarke Error Grid分析图
    参数:
    reference: 参考血糖值 (mmol/L)
    prediction: 预测血糖值 (mmol/L)
    """
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 设置坐标轴范围
    max_val = 22.2  # 400 mg/dL in mmol/L
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # 绘制对角线（虚线）
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1)

    # 关键血糖值点（mg/dL转换为mmol/L）
    x_70 = 3.89   # 70 mg/dL
    x_150 = 8.33  # 150 mg/dL
    x_180 = 10    # 180 mg/dL
    y_70 = 3.89   # 70 mg/dL
    y_180 = 10    # 180 mg/dL

    # 垂直分界线
    ax.axvline(x=x_70, ymin=0, ymax=x_70*0.8/max_val, color='k', linewidth=1)
    ax.axvline(x=x_70, ymin=x_70*1.2/max_val, ymax=max_val, color='k', linewidth=1)
    ax.axvline(x=x_180, ymin=0, ymax=x_70/max_val, color='k', linewidth=1)
    ax.axvline(x=240/18, ymin=y_70/max_val, ymax=y_180/max_val, color='k', linewidth=1)

    # 水平分界线
    ax.axhline(y=y_70, xmin=0, xmax=x_70/max_val/1.2, color='k', linewidth=1)
    ax.axhline(y=y_70, xmin=x_180/max_val, xmax=max_val, color='k', linewidth=1)
    ax.axhline(y=y_180, xmin=0, xmax=x_70/max_val, color='k', linewidth=1)
    ax.axhline(y=y_180, xmin=240/18/max_val, xmax=max_val, color='k', linewidth=1)

    # A区边界线（±20%）
    ax.plot(np.array([x_70/1.2, max_val]), 1.2 * np.array([x_70/1.2, max_val]), 'k-', linewidth=1)
    ax.plot(np.array([x_70, max_val]), 0.8 * np.array([x_70, max_val]), 'k-', linewidth=1)

    # # B区边界线（±50%）
    ax.plot([x_70, (max_val-110/18)], [y_180, max_val], 'k-', linewidth=1)
    ax.plot([130/18,x_180], [0, y_70], 'k-', linewidth=1)

    # 添加区域标签
    ax.text(2, 3, 'A', fontsize=15)   # 左下B区
    ax.text(6, 15, 'C', fontsize=15)  # A区中心
    ax.text(18, 12, 'B', fontsize=15)  # 右上B区
    ax.text(10, 15, 'B', fontsize=15)  # 右上B区
    ax.text(9, 1, 'C', fontsize=15)  # C区
    ax.text(2, 8, 'D', fontsize=15)  # 左上D区
    ax.text(18, 8, 'D', fontsize=15) # 右上D区
    ax.text(2, 15, 'E', fontsize=15)  # 左上E区
    ax.text(18, 2, 'E', fontsize=15)  # 右下E区

    # 绘制数据点
    plt.scatter(reference, prediction, c='blue', s=30)

    # 设置标题和标签
    plt.title("Clarke Error Grid Analysis")
    plt.xlabel("Reference Glucose (mmol/L)")
    plt.ylabel("Predicted Glucose (mmol/L)")

    def get_zone(ref, pred):
        """
        确定测量点所属的区域
        ref, pred: 单位为mmol/L
        """
        # 转换为mg/dL
        ref_mgdl = ref * 18
        pred_mgdl = pred * 18
        
        # A区判定
        if ref_mgdl < 70:
            if abs(pred_mgdl - ref_mgdl) <= 20:
                return 'A'
        else:
            if pred_mgdl >= ref_mgdl * 0.8 and pred_mgdl <= ref_mgdl * 1.2:
                return 'A'
        
        # B区判定
        if ref_mgdl < 70:
            if pred_mgdl <= 180 and pred_mgdl > 70:
                return 'B'
        elif ref_mgdl >= 180:
            if pred_mgdl >= 70:
                return 'B'
        else:
            if ((pred_mgdl >= ref_mgdl * 0.7 and pred_mgdl <= ref_mgdl * 1.5) or
                (pred_mgdl >= ref_mgdl - 30 and pred_mgdl <= ref_mgdl + 30)):
                return 'B'
        
        # C区判定
        if (ref_mgdl >= 70 and ref_mgdl <= 290 and
            pred_mgdl >= ref_mgdl * 0.5 and pred_mgdl <= ref_mgdl * 0.8):
            return 'C'
        
        # E区判定
        if ((ref_mgdl <= 70 and pred_mgdl >= 180) or
            (ref_mgdl >= 180 and pred_mgdl <= 70)):
            return 'E'
        
        # D区判定（其他所有情况）
        return 'D'

    # 计算各区域的点数
    zones = [get_zone(r, p) for r, p in zip(reference, prediction)]
    zone_counts = {zone: zones.count(zone) for zone in 'ABCDE'}
    total = len(zones)
    
    # 显示统计结果
    # for i, zone in enumerate('ABCDE'):
    #     percentage = zone_counts[zone] / total * 100
    #     plt.text(0.02, 0.98 - i*0.05, 
    #             f'Zone {zone}: {percentage:.1f}%',
    #             transform=plt.gca().transAxes)

    return zone_counts


def plotly_simple_chart(data, x_axis=None, x_tick_interval=100, x_title="", y_title="Intensity", title="", template="plotly_white", tick_angle=90):
    
    """
    绘制数据图表，支持可选参数调整。

    参数:
    - data: 必选，二维numpy数组，每列为一个样本
    - x_axis: 可选，x轴坐标的numpy数组，默认为None
    - x_tick_interval: 可选，x轴显示间隔，默认为100
    - x_title: 可选，x轴标题，默认为"Time"
    - y_title: 可选，y轴标题，默认为"PD Sample Value"
    - title: 可选，表的标题，默认为"Sample Data Over Time"
    - template: 可选，Plotly图表模板，默认为"plotly_white"
    - tick_angle: 可选，x轴刻度角度，默认为90度
    """
    import warnings
    warnings.warn("这里的画图和matplotlib一样,每一列才是一个样本。", UserWarning)
    fig = go.Figure()

    # 绘制每列数据
    for i in range(data.shape[1]):
        fig.add_trace(go.Scatter(y=data[:, i], mode='lines', name=f'{i+1}'))

    # 如果没有提供x_axis，使用默认的索引
    if x_axis is None:
        x_axis = np.arange(data.shape[0])

    # 设置x轴刻度值
    x_ticks = np.arange(0, len(x_axis), x_tick_interval)

    # 更新图表布局
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis=dict(
            tickmode='array',
            tickvals=x_ticks,
            ticktext=x_axis[x_ticks],
            tickangle=tick_angle
        ),
        template=template
    )

    fig.show()
    return fig

def analyze_model_performance(y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred):
    """
    自动判断任务类型（分类或回归），并输出相应的分析报告。

    参数:
    y_train: 训练集真实值
    y_train_pred: 训练集预测值
    y_val: 验证集真实值
    y_val_pred: 验证集预测值
    y_test: 测试集真实值
    y_test_pred: 测试集预测值
    """
    # 判断任务类型：分类（离散）还是回归（连续）
    if np.issubdtype(y_train.dtype, np.integer) or len(np.unique(y_train)) <= 20:
        task_type = 'classification'
    else:
        task_type = 'regression'
    
    if task_type == 'classification':
        # 分类任务
        print("Classification Task Detected")
        
        # 计算分类指标
        accuracy_train = accuracy_score(y_train, y_train_pred)
        precision_train = precision_score(y_train, y_train_pred)
        recall_train = recall_score(y_train, y_train_pred)
        f1_train = f1_score(y_train, y_train_pred)

        accuracy_val = accuracy_score(y_val, y_val_pred)
        precision_val = precision_score(y_val, y_val_pred)
        recall_val = recall_score(y_val, y_val_pred)
        f1_val = f1_score(y_val, y_val_pred)

        accuracy_test = accuracy_score(y_test, y_test_pred)
        precision_test = precision_score(y_test, y_test_pred)
        recall_test = recall_score(y_test, y_test_pred)
        f1_test = f1_score(y_test, y_test_pred)


        # 打印分类报告
        print("\nTraining Metrics:")
        print(f"Accuracy: {accuracy_train:.4f}")
        print(f"Precision: {precision_train:.4f}")
        print(f"Recall: {recall_train:.4f}")
        print(f"F1 Score: {f1_train:.4f}")

        print("\nValidation Metrics:")
        print(f"Accuracy: {accuracy_val:.4f}")
        print(f"Precision: {precision_val:.4f}")
        print(f"Recall: {recall_val:.4f}")
        print(f"F1 Score: {f1_val:.4f}")


        print("\nTesting Metrics:")
        print(f"Accuracy: {accuracy_test:.4f}")
        print(f"Precision: {precision_test:.4f}")
        print(f"Recall: {recall_test:.4f}")
        print(f"F1 Score: {f1_test:.4f}")
        
  

    elif task_type == 'regression':
        # 回归任务
        print("Regression Task Detected")
        
        # 计算回归指标
        mse_train = mean_squared_error(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)

        mse_val = mean_squared_error(y_val, y_val_pred)
        mae_val = mean_absolute_error(y_val, y_val_pred)
        r2_val = r2_score(y_val, y_val_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # 打印回归报告
        print("\nTraining Metrics:")
        print(f"MSE: {mse_train:.4f}")
        print(f"MAE: {mae_train:.4f}")
        print(f"R2: {r2_train:.4f}")

        print("\nValidation Metrics:")
        print(f"MSE: {mse_val:.4f}")
        print(f"MAE: {mae_val:.4f}")
        print(f"R2: {r2_val:.4f}")

        print("Regression Task Detected")
        print("\nTest Metrics:")
        print(f"MSE: {mse_test:.4f}")
        print(f"MAE: {mae_test:.4f}")
        print(f"R2: {r2_test:.4f}")

        # 绘制实际值与预测值的散点图
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(y_train, y_train_pred, color='blue', alpha=0.5)
        plt.title('Training: Actual vs. Predicted')
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')

        plt.subplot(1, 2, 2)
        plt.scatter(y_val, y_val_pred, color='green', alpha=0.5)
        plt.title('Validation: Actual vs. Predicted')
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')

        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_test_pred, color='red', alpha=0.5)
        plt.title('Test: Actual vs. Predicted')
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        # plt.show()

        plt.tight_layout()
        plt.show()


def plot_actual_vs_predicted(X_axis_train, y_train, y_train_pred, X_axis_val, y_val, y_val_pred, X_axis_test, y_test, y_test_pred,title = "",X_axis_step=30,draw_type = 'line'):
    """
    绘制训练集、验证集和测试集的实际值和预测概率。

    参数：
    - date_time_train: 训练集的时间序列
    - df_train_results: 训练集的DataFrame，包含'Actual'和'Probability'列
    - date_time_val: 验证集的时间序列
    - df_val_results: 验证集的DataFrame，包含'Actual'和'Probability'列
    - date_time_test: 测试集的时间序列
    - df_test_results: 测试集的DataFrame，包含'Actual'和'Probability'列
    """
    if draw_type == 'scatter':
        plt.figure(figsize=(24, 16))
        # 1. 绘制训练集的实际值和预测值散点图
        plt.scatter(X_axis_train, y_train, label="Train Actual", alpha=0.7, color='blue', marker='o')
        plt.scatter(X_axis_train, y_train_pred, label="Train Predict", alpha=0.7, color='cyan', marker='x')
        # 2. 绘制验证集的实际值和预测值散点图
        plt.scatter(X_axis_val, y_val, label="Val Actual", alpha=0.7, color='green', marker='o')
        plt.scatter(X_axis_val, y_val_pred, label="Val Predict", alpha=0.7, color='lightgreen', marker='x')
        # 3. 绘制测试集的实际值和预测值散点图
        plt.scatter(X_axis_test, y_test, label="Test Actual", color='red', marker='o')
        plt.scatter(X_axis_test, y_test_pred, label="Test Predict", color='orange', marker='x')
        # 设置图表标题、标签和图例
        plt.title('Actual vs Predicted (Train, Validation, and Test)' + title)
        plt.xlabel('DateTime')
        plt.ylabel('Class Label / Probability')
        # 稀疏化 X 轴：显示时间轴并旋转
        plt.xticks(np.concatenate([X_axis_train, X_axis_val, X_axis_test])[::X_axis_step], rotation=90)
        # 添加图例
        plt.legend()
        # 显示网格
        plt.grid(True)
        # 调整布局
        plt.tight_layout()
        # 显示图表
        plt.show()
        return
    plt.figure(figsize=(12, 8))
    # 1. 绘制训练集的实际值和预测概率
    plt.plot(X_axis_train, y_train, label="Train Actual", marker='o', linestyle='--', alpha=0.7, color='blue')
    plt.plot(X_axis_train, y_train_pred, label="Train Predict", marker='x', linestyle='--', alpha=0.7, color='cyan')
    # 2. 绘制验证集的实际值和预测概率
    plt.plot(X_axis_val, y_val, label="Val Actual", marker='o', linestyle='--', alpha=0.7, color='green')
    plt.plot(X_axis_val, y_val_pred, label="Val Predict", marker='x', linestyle='--', alpha=0.7, color='lightgreen')
    # 3. 绘制测试集的实际值和预测概率
    plt.plot(X_axis_test, y_test, label="Test Actual", marker='o', color='red')
    plt.plot(X_axis_test, y_test_pred, label="Test Predict", marker='x', color='orange')
    # 设置图表标题、标签和图例
    plt.title('Actual vs Predicted Probabilities (Train, Validation, and Test)'+title)
    plt.xlabel('DateTime')
    plt.ylabel('Class Label / Probability')
    # 稀疏化 X 轴：显示时间轴并旋转
    plt.xticks(np.concatenate([X_axis_train, X_axis_val, X_axis_test])[::X_axis_step], rotation=90)
    # 添加图例
    plt.legend()
    # 显示网格
    plt.grid(True)
    # 调整布局
    plt.tight_layout()
    # 显示图表
    plt.show()


def plot_pca_with_class_distribution(X_train, X_val, X_test, y_train, y_val, y_test, n_components=3):
    """传入划分好的三组数据，进行PCA降维，并画3D图,总共是有6个类别，每个类别用不同的颜色表示，会打印每个类别的数量
    Perform PCA on the combined train, validation, and test datasets, and plot them in a 3D scatter plot.
    Different classes (from the labels) are colored differently.
    
    Parameters:
    ----------
    X_train : pd.DataFrame or np.array
        Training dataset features.
    X_val : pd.DataFrame or np.array
        Validation dataset features.
    X_test : pd.DataFrame or np.array
        Test dataset features.
    y_train : pd.Series or np.array
        Training dataset labels.
    y_val : pd.Series or np.array
        Validation dataset labels.
    y_test : pd.Series or np.array
        Test dataset labels.
    n_components : int, optional
        The number of PCA components to reduce to. Default is 3.
        
    Returns:
    -------
    None
        Displays the 3D scatter plot.
    """
    plt.rcParams['font.family'] = 'SimHei'  # 替换为你选择的字体
    plt.rcParams['axes.unicode_minus'] = False 
    # Combine the datasets (features and labels)
    X_combined = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_val), pd.DataFrame(X_test)], axis=0)
    y_combined = pd.concat([pd.Series(y_train), pd.Series(y_val), pd.Series(y_test)], axis=0)
    
    # Perform PCA to reduce the combined data to the specified number of components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_combined)
    
    # Get the sizes of the train, validation, and test datasets
    train_size = len(X_train)
    val_size = len(X_val)
    
    # Split the PCA results back into the respective datasets
    X_train_pca = X_pca[:train_size]
    X_val_pca = X_pca[train_size:train_size + val_size]
    X_test_pca = X_pca[train_size + val_size:]
    
    # Split the labels back into the respective datasets
    y_train_pca = y_combined[:train_size]
    y_val_pca = y_combined[train_size:train_size + val_size]
    y_test_pca = y_combined[train_size + val_size:]

    # Count the number of instances of each class in train, validation, and test sets
    print("Class distribution:")
    unique_labels = y_combined.unique()
    for label in unique_labels:
        train_count = sum(y_train == label)
        val_count = sum(y_val == label)
        test_count = sum(y_test == label)
        total_count = train_count + val_count + test_count
        print(f"Class {label}: Train = {train_count}, Validation = {val_count}, Test = {test_count}, Total = {total_count}")
    
    
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot train data by class
    unique_labels = y_combined.unique()
    for label in unique_labels:
        ax.scatter(X_train_pca[y_train_pca == label, 0], X_train_pca[y_train_pca == label, 1], X_train_pca[y_train_pca == label, 2], 
                   label=f'Train - Class {label}', alpha=0.6)

    # Plot validation data by class
    for label in unique_labels:
        ax.scatter(X_val_pca[y_val_pca == label, 0], X_val_pca[y_val_pca == label, 1], X_val_pca[y_val_pca == label, 2], 
                   label=f'Validation - Class {label}', alpha=0.6)

    # Plot test data by class
    for label in unique_labels:
        ax.scatter(X_test_pca[y_test_pca == label, 0], X_test_pca[y_test_pca == label, 1], X_test_pca[y_test_pca == label, 2], 
                   label=f'Test - Class {label}', alpha=0.6)

    # Set axis labels and title
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title('PCA 3D Visualization of Train, Validation, and Test Data by Class')

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

def matlplotlib_chinese_display_fix():
    # 查看系统存在的字体
    a=sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
    for i in a:
        print(i)
    # 解决中文显示问题
    plt.rcParams['font.family'] = 'SimHei'  # 替换为你选择的字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_mean( data_X, data_y = None,data_name = '',X_name = '',y_name='',scale=True,draw_y=True, save_dir=None,width_and_height=None,X_ticks=None):
# 2024-10-10
    '''
    example:
        # 数据名
        data_name = 'Simple Data Example'
        X_name = 'Feature Mean'
        y_name = 'Target'

        # 随机生成10个样本，每个样本有5个特征
        data_X = np.random.rand(10, 5)

        # 随机生成10个标签
        data_y = np.random.rand(10)
        width_and_height = (1800, 1200)

        # 调用plot_mean函数进行绘图
        plot_mean(data_name, X_name, y_name, data_X, data_y, scale=True, draw_y=True)

    ---------
    Parameters:
    ---------
        - data_name : str 数据名
        - X_name : str 特征名
        - y_name : str 标签名
        - data_X : ndarray 特征数据
        - data_y : ndarray 标签数据
        - scale : bool 是否进行数据缩放
        - draw_y : bool 是否画y轴
        - save_dir : str 保存路径
        ---------
    Returns:
    ---------
        - 绘制均值变化图
    '''
    
    data_X_mean = np.mean(data_X, axis=1)


    if data_y is None:
        draw_y = False
    if scale:
        data_X_mean = (data_X_mean-np.min(data_X_mean))/(np.max(data_X_mean)-np.min(data_X_mean))
        if data_y is not None:
            data_y = (data_y-np.min(data_y))/(np.max(data_y)-np.min(data_y))
    

    if X_ticks is None:
        X_ticks = np.arange(len(data_X_mean))

    fig = go.Figure()

    # 设置不同颜色
    fig.add_trace(go.Scatter(x=X_ticks,
                                y=data_X_mean,
                                mode='lines+markers',
                                marker=dict(color='blue'),
                                line=dict(color='blue'),
                                text=data_y, 
                                name=X_name))
    if draw_y:
        fig.add_trace(go.Scatter(x=X_ticks,
                                y=data_y,
                                mode='lines',
                                name=y_name))
        # 设置图表布局，包括标题
    fig.update_layout(
        title=f"Mean of {data_name}",
        xaxis_title="X Axis Title",
        yaxis_title="Y Axis Title"
    )
    
    if width_and_height is None:
        width, height = 1800, 1200
    else:
        width, height = width_and_height


    if save_dir is None:
        fig.show()
    else:
        fig.write_image( save_dir+ f'/{data_name}.png',width=1800, height=1200)

def spectral_absorption_v2(X,y = None,category="all_samples",wave = None,plt_show = False):
    '''画光谱吸收图
    -------
    Parameters:
    ---------
        - X : ndarray 光谱数据 shape of (n_samples,n_features)
        - y : ndarray 物质含量
        - class :表示数据是属于谁的，默认是所有人的，如果是某个人的，就填写志愿者的名字
        - wave : ndarray 波长
        - plt_show : bool 是否用matplotlib画图
    ---------
    Returns:
    ---------
    Modify:
    -------
        - 2023-11-28 : 增加了Wavelengths参数，可以传入波长数据，如果不传入，就默认加载1899维的波长数据
        - 2023-12-14 : 增加了plt_show参数，可以选择是用matlibplot画图还是用plotly画图
        - 2024-01-09 : 增加了相同y值使用相同颜色的功能

    '''
    import numpy as np
    import pandas as pd 
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    
    # 读取数据
    absorbance = X
###### 2023-11-28 增加了Wavelengths参数，可以传入波长数据，如果不传入，就默认加载1899维的波长数据 begin
    # if wave is None: 
    #     wavelengths = [i for i in range(X.shape[1])]
    # else:
    #     wavelengths = wave
###############################3 end
        
    ####begin 是否用matplotlib画图 modify 2023-12-14
    if plt_show:
        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # matplotlib plotting
        plt.figure(figsize=(15,5))
        
        if y is not None:
            # 获取唯一的y值
            unique_y = np.unique(y)
            # 为每个唯一的y值分配一个颜色
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_y)))
            color_dict = dict(zip(unique_y, colors))
            
            # 按y值分组画图
            for y_val in unique_y:
                mask = y == y_val
                for i in range(len(absorbance[mask])):
                    plt.plot(absorbance[mask][i,:], color=color_dict[y_val], label=f'y={y_val}')
        else:
            for i in range(len(absorbance)):
                plt.plot(absorbance[i,:])
                
        plt.xlabel('Wavelength')
        plt.ylabel('Absorbance')
        plt.legend()
        plt.title(category+' Spectra Absorbance Curve')
        plt.show()
        return
    ####end 是否用matplotlib画图 modify 2023-12-14

    if y is None:
        y = np.zeros(X.shape[0])
        print("Warning: y is None, set y to zeros")

    fig = go.Figure()
    
    # 获取唯一的y值并为每个值分配颜色
    unique_y = np.unique(y)
    colors = [f'hsl({h},50%,50%)' for h in np.linspace(0, 360, len(unique_y))]
    color_dict = dict(zip(unique_y, colors))
    
    # 按y值分组添加曲线
    for y_val in unique_y:
        mask = y == y_val
        for i in range(len(absorbance[mask])):
            fig.add_trace(go.Scatter(
                x=wave if wave is not None else None, 
                y=absorbance[mask][i,:],
                name=f'y={y_val}',
                line=dict(color=color_dict[y_val])
            ))
            
    fig.update_layout(
        title=category+" Spectra Absorbance Curve",
        xaxis_title="Wavelength(nm)",
        yaxis_title="Absorbance"
    )
    fig.show()

def spectral_absorption(X,y = None,category="all_samples",wave = None,plt_show = False):
    '''画光谱吸收图
    -------
    Parameters:
    ---------
        - X : ndarray 光谱数据 shape of (n_samples,n_features)
        - y : ndarray 物质含量
        - class :表示数据是属于谁的，默认是所有人的，如果是某个人的，就填写志愿者的名字
        - wave : ndarray 波长
        - plt_show : bool 是否用matplotlib画图
    ---------
    Returns:
    ---------
    Modify:
    -------
        - 2023-11-28 : 增加了Wavelengths参数，可以传入波长数据，如果不传入，就默认加载1899维的波长数据
        - 2023-12-14 : 增加了plt_show参数，可以选择是用matlibplot画图还是用plotly画图

    '''
    import numpy as np
    import pandas as pd 
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    
    # 读取数据
    absorbance = X
###### 2023-11-28 增加了Wavelengths参数，可以传入波长数据，如果不传入，就默认加载1899维的波长数据 begin
    # if wave is None: 
    #     wavelengths = [i for i in range(X.shape[1])]
    # else:
    #     wavelengths = wave
###############################3 end
        

    ####begin 是否用matplotlib画图 modify 2023-12-14
    if plt_show:
        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # matplotlib plotting
        plt.figure(figsize=(15,5))
        for i in range(len(absorbance)):
            plt.plot(absorbance[i,:])
        plt.xlabel('Wavelength')
        plt.ylabel('Absorbance')
        # plt.legend()
        plt.title(category+' Spectra Absorbance Curve')
        plt.show()
        return
    ####end 是否用matplotlib画图 modify 2023-12-14

    
    if y is None:
        y = np.zeros(X.shape[0])
        print("Warning: y is None, set y to zeros")


    fig = go.Figure()
    for i in range(len(absorbance)):
        fig.add_trace(go.Scatter(x=None, y=absorbance[i,:],name=str(y[i])))
    fig.update_layout(title=category+" Spectra Absorbance Curve",
                        xaxis_title="Wavelength(nm)",
                        yaxis_title="Absorbance")
    fig.show()

def Sample_spectral_ranking(X,y,category="all_samples" ,wave = None):
    '''画光谱吸收排序图
    -------
    Parameters:
    ---------
        - X : ndarray 光谱数据
        - y : ndarray 物质含量
        - class :表示数据是属于谁的，默认是所有人的，如果是某个人的，就填写志愿者的名字
    ---------
    Returns:
    
        '''
    # all samples drwa with plotly
    import pandas as pd
    import plotly.graph_objects as go
    
    if wave is None:
        wavelengths = [i for i in range(X.shape[1])]
    else:
        wavelengths = wave

    absorbance = X
    
    ranked = X.argsort(axis=0).argsort(axis=0)

    
    # Plotly plotting
    fig = go.Figure()
    for i in range(ranked.shape[0]):
        fig.add_trace(go.Scatter(x=None, y=ranked[i,:], name=str(y[i])))
        
    # 设置标题
    # fig.update_layout(title=str(ca+" Spectra Absorbance Ranking Curve")
    fig.update_xaxes(title_text="Wavelength")
    fig.update_yaxes(title_text="Rank")
    fig.update_layout(title=str(category)+" Spectra Absorbance Ranking Curve" )
    fig.show()

def Numerical_distribution(ndarr,category:Union[str,list]="all_samples",feat_ = None):
    '''画数值分布图
    -------
    Parameters:
    ---------
        - ndarr : ndarray 数据,每一列是一个特征或者Y值，每一行是一个样本
        - class :表示数据是属于谁的，默认是所有人的，如果是某个人的，就填写志愿者的名字
        - feat_ : 特征名称或Y值名称
    ---------
    Returns:
    ---------
    Example:
    ---------

    '''
        
    import numpy as np
    import pandas as pd
    import plotly.express as px

    m = ndarr.shape[0]
    n = ndarr.shape[1]
    if feat_ is None:
        feat_ = []
        for i in range(n):
            feat_.append("feat_"+str(i))
    
    vals = [] 
    counts = []
    feats = []
    for i in range(n):
        unique, counts_i = np.unique(ndarr[:,i], return_counts=True)
        feats += [feat_[i]]*len(unique) # 重复特征名称,根据unique长度给出每个值的特征名称
        vals += list(unique)# 每个值
        counts += list(counts_i)# 每个值出现的次数
        
    df = pd.DataFrame({'feats': feats, 'vals': vals, 'counts': counts})
    fig1 = px.bar(df, x='vals', y='counts', color='feats', barmode='group',title=category+" Numerical Distribution Bar")

    fig1.show()

def RankingMAE_and_Spearman_between_X_and_Y(X,
                                            y,
                                            category = "all_samples",
                                            vertical_line:list = None,
                                            RankingMAE = True,
                                            Spearman = True,
                                            Spearman_abs = True,
                                            return_spearman = False,
                                            draw = True,
                                            wave = None
                                            ):
    """画X和Y之间的排名MAE和spearman相关系数
    -------
    Parameters:
    ---------
        - X : ndarray 光谱数据
        - y : ndarray 物质含量
        - class :表示数据是属于谁的，默认是所有人的，如果是某个人的，就填写志愿者的名字
        - vertical_line : list 垂直线的位置
        - RankingMAE : bool 是否画MAE
        - Spearman : bool 是否画spearman
    ---------
    Returns:
    ---------
    Modify:
        - 2023-11-28 : 增加了vertical_line参数，可以在spearman图中画出垂直线
                       增加了RankingMAE和Spearman两个参数，可以选择是否画MAE和spearman
        - 2023-11-29 : 增加了Spearman_abs参数，可以选择是否画spearman的绝对值
        - 2023-12-06 : 增加了return_spearman参数，可以选择是否返回spearman的最大值和平均值
        - 2023-12-06 : 增加了draw参数，可以选择是否画图
    """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


    if wave is None:
        wavelengths = [i for i in range(X.shape[1])]
    
    from sklearn.metrics import mean_absolute_error
    import matplotlib.pyplot as plt

    X_ranks = X.argsort(axis=0).argsort(axis=0)
    y_ranks = y.argsort(axis=0).argsort(axis=0)
    
    if RankingMAE: # 画MAE
        maes = []
        for i in range(len(wavelengths)):
            maes.append(mean_absolute_error(X_ranks[:,i], y_ranks))
        plt.figure(figsize=(15,5))
        if Spearman:
            plt.subplot(1,2,1)
        plt.plot(wavelengths, maes)
        plt.xlabel('Wavelength')
        plt.ylabel('MAE')
        plt.title(category+'MAE Between Absorbance Ranking and y_value Ranking')


    if Spearman and draw: # 画spearman
        coffs_spearman = []
        from scipy.stats import spearmanr
        for i in range(len(wavelengths)):
            coeff, p_value  = spearmanr(X[:,i], y)
            
            coffs_spearman.append(coeff)
            # #### modify 2023-11-29
            # if p_value<0.01:
            #     coffs_spearman.append(coeff)
            # else:
            #     coffs_spearman.append(0)
            # ###### modify 2023-11-29
        if Spearman_abs:
            coffs_spearman = np.abs(coffs_spearman)
        
        if RankingMAE and draw:
            plt.subplot(1,2,2)
        plt.plot(wavelengths, coffs_spearman)
        plt.xlabel("Wavelengths")
        plt.ylabel("spearman"+"(avg:{},abs_max:{},index:{})".format(round(np.mean(np.abs(coffs_spearman)),3),round(max(np.abs(coffs_spearman)),3),np.argmax(np.abs(coffs_spearman))))
    # 11-28 增加了画垂直线的功能 begin
        if vertical_line is not None:  
            for i in vertical_line:
                from nirapi.load_data import get_wave_accroding_feat_index
                plt.vlines(get_wave_accroding_feat_index([i]),min(coffs_spearman),max(coffs_spearman),colors = "r",linestyles = "dashed")
            # plt.legend()
    # 11-28 增加了画垂直线的功能 end
        plt.title(category+"spearman Between Absorbance and y_value,samples_num:"+str(len(y_ranks)))
    plt.show()
    if return_spearman:
        coffs_spearman = []
        from scipy.stats import spearmanr
        for i in range(len(wavelengths)):
            coeff, p_value  = spearmanr(X[:,i], y)
            coffs_spearman.append(coeff)
        coffs_spearman = np.abs(coffs_spearman)
        return (round(max(coffs_spearman),3),round(np.mean(coffs_spearman),3))

def train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,category = "all_samples",save = None):
    """画训练集和测试集的散点图
    -------
    Parameters:
    ---------
        - y_train : ndarray 训练集的y值
        - y_train_pred : ndarray 训练集的预测y值
        - y_test : ndarray 测试集的y值
        - y_test_pred : ndarray 测试集的预测y值
        - class :表示数据是属于谁的，默认是所有人的，如果是某个人的，就填写志愿者的名字
    ---------
    Returns:
    """
    import matplotlib.pyplot as plt
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure()
    plt.scatter(y_train,y_train_pred,label='Training set')
    plt.scatter(y_test,y_test_pred,label='Testing set')
    from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
    MAE = mean_absolute_error(y_test,y_test_pred)
    R2 = r2_score(y_test,y_test_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.plot([0.1,0.5],[0.1,0.5],'r-')
    plt.text(0.1,0.4,"MAE:{:.5},R2:{:.5}".format(MAE,R2))
    plt.legend()
    plt.title(category+" train_test_Scatter")
    from datetime import datetime

    # 获取当前时间
    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    if save is not None:
        plt.savefig(save + category +current_time_str+".png")
    plt.show()
    return

def Comparison_of_data_before_and_after(before_X,
                                        before_y,
                                        after_X,
                                        after_y,
                                        category = "all_samples",
                                        Spearman = True,
                                        Spearman_abs = True,
                                        vertical_line = True, # 在spearman图中画垂直线
                                        Residual = True, # 画残差线
                                        ):
    """画数据处理前后的spearman图
    -------
    Parameters:
    ---------

        - before_X : ndarray 处理前的X
        - before_y : ndarray 处理前的y
        - after_X : ndarray 处理后的X
        - after_y : ndarray 处理后的y
        - category : str 表示数据是属于谁的，默认是所有人的，如果是某个人的，就填写志愿者的名字
        - Spearman : bool 是否画spearman
        - Spearman_abs : bool spearman是否是绝对值
        - vertical_line : list 垂直线的位置
        - Residual : bool 是否画残差图

    ---------
    Returns:
    ---------
    Modify:
        - 2023-12-13 : 第一次生成了这个函数，用来画数据处理前后的spearman图
        - 2023-12-14 : 增加了画残差图的功能
    """

    import numpy as np
    import matplotlib.pyplot as plt
    #解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False








    # 画spearman
    if True: # 画spearman  modify 2023-12-14  原本是if Spearman: ，现在改成了if True: 因为我想要画残差图，该功能由下面代码实现
            P=True   # 是否画p_value大于0.01的spearman ,如果为True,就画所有

            # 处理前的数据画spearmen  
            before_coffs_spearman = []
            from scipy.stats import spearmanr
            for i in range(len(wavelengths)):
                coeff, p_value  = spearmanr(before_X[:,i], before_y)
                if P:
                    before_coffs_spearman.append(coeff)
                else:
                    if p_value<0.01:
                        before_coffs_spearman.append(coeff)
                    else:
                        before_coffs_spearman.append(0)
            if Spearman_abs:
                before_coffs_spearman = np.abs(before_coffs_spearman)

            if Spearman: # 画spearman  add 2023-12-14
                plt.plot(wavelengths, before_coffs_spearman,label = "before")







            # 处理后的数据画spearman
            after_coffs_spearman = []
            from scipy.stats import spearmanr
            for i in range(len(wavelengths)):
                coeff, p_value  = spearmanr(after_X[:,i], after_y)
                if P:
                    after_coffs_spearman.append(coeff)
                else:
                    if p_value<0.01:
                        after_coffs_spearman.append(coeff)
                    else:
                        after_coffs_spearman.append(0)
            if Spearman_abs:
                after_coffs_spearman = np.abs(after_coffs_spearman)
            if Spearman: # 画spearman  add 2023-12-14
                plt.plot(wavelengths, after_coffs_spearman  ,label = "after")

            




            ##begin 画残差图 modify 2023-12-14
            residual = after_coffs_spearman - before_coffs_spearman  ## 计算残差
            plt.plot(wavelengths, residual,label = "residual")
            ##end 画残差图








        # 11-28 增加了画垂直线的功能 begin
            # if vertical_line is not None:  
            #     for i in vertical_line:
            #         from nirapi.load_data import get_wave_accroding_feat_index
                    # plt.vlines(get_wave_accroding_feat_index([i]),min(coffs_spearman),max(coffs_spearman),colors = "r",linestyles = "dashed")
                # plt.legend()
        # 11-28 增加了画垂直线的功能 end







            ## 显示图
            plt.xlabel("Wavelengths") # 设置x轴标签
            plt.ylabel("spearman") # 设置y轴标签
            plt.title(category+"spearman before and after preprocessing, samples:"+str(len(before_y))) # 设置图表标题
            plt.legend() # 显示图例
            plt.savefig( r"D:/Desktop/NIR spectroscopy/main/Features_Selection_Analysis/every_day_try/file/"+category+".png")
            plt.close()
            # plt.show()
            return

def Numerical_distribution_V2(ndarr,category:Union[str,list]="all_samples",feat_ = None):
    '''画数值分布图,这个版本是为了解决画图的时候，要同时画好多个特征， 或者好多个Y值的情况，并且主要是Y的数量不一致的情况
    在12-13号的时候，我想要画10个人随机特征选择得到的特征的分布图，但是我不想要画10个图，我想要画一个图，这个函数就是为了解决这个问题
    -------
    Parameters:
    ---------
        - ndarr : list of ndarray 数据,list中每个元素是一个ndarray，shape of (-1,1) 
        - class : 如果是一个字符串，表示就只有一个人，如果是一个list，表示有多个人
        - feat_ : 特征名称或Y值名称
    ---------
    Returns:
    ---------
    Example:
    ---------

    '''
    if isinstance(category,list):
    
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go

    
        n = len(category)
        feat_ = category
        # if feat_ is None:
        #     feat_ = []
        #     for i in range(n):
        #         feat_.append("feat_"+str(i))
        
        vals = [] 
        counts = []
        feats = []
        for i in range(n):
            unique, counts_i = np.unique(ndarr[i], return_counts=True)
            feats += [feat_[i]]*len(unique) # 重复特征名称,根据unique长度给出每个值的特征名称
            vals += list(unique)# 每个值
            counts += list(counts_i)# 每个值出现的次数
            
        df = pd.DataFrame({'feats': feats, 'vals': vals, 'counts': counts})

        # fig1.add_bar(df, x='vals', y='counts', color='feats', barmode='group',title=name+" Numerical Distribution Bar")
        colors = ['red','blue','green','yellow','black','pink','orange','purple','gray','brown']
        
        fig1 = px.bar(df, x='vals', y='counts', color='feats',barmode='group',title="Numerical Distribution Bar")
        fig1.show()
        return
    import numpy as np
    import pandas as pd
    import plotly.express as px

    m = ndarr.shape[0]
    n = ndarr.shape[1]
    if feat_ is None:
        feat_ = []
        for i in range(n):
            feat_.append("feat_"+str(i))
    
    # 画数值分布图
    vals = [] 
    counts = []
    feats = []
    for i in range(n):
        unique, counts_i = np.unique(ndarr[:,i], return_counts=True)
        feats += [feat_[i]]*len(unique) # 重复特征名称,根据unique长度给出每个值的特征名称
        vals += list(unique)# 每个值
        counts += list(counts_i)# 每个值出现的次数
        
    df = pd.DataFrame({'feats': feats, 'vals': vals, 'counts': counts})
    fig1 = px.bar(df, x='vals', y='counts', color='feats', barmode='group',title=category+" Numerical Distribution Bar")

    fig1.show()

def classification_report_plot(y_test, y_pred,data_name = ""):

    # 画分类的图
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    acc = accuracy_score(y_test, y_pred)
    stats_text = "\n\nAccuracy: {:0.3f}".format(acc)
    fig = plt.figure(figsize=(10, 7))
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="Blues", fmt='g')
    plt.title( data_name + 'Validation Set Confusion Matrix' + stats_text, y=1.1)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def train_and_test_pred_plot(y_train,y_test,y_pred_train,y_pred_test,data_name="",save_dir=None,each_class_mae=True,content = ""):
    import plotly.graph_objects as go
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr

    # 计算指标
    pearson_train, _ = pearsonr(y_train, y_pred_train)
    pearson_test, _ = pearsonr(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    

    y_train_mae_dict = {}
    y_test_mae_dict = {}

    if each_class_mae:
        for i in np.unique(y_test):
            
            indes_train = np.where(y_train == i)[0]
            indes_test = np.where(y_test == i)[0]
            y_train_i = y_train[indes_train]
            y_pred_train_i = y_pred_train[indes_train]
            y_test_i = y_test[indes_test]
            y_pred_test_i = y_pred_test[indes_test]
            

            mae_train_i = mean_absolute_error(y_train_i, y_pred_train_i)
            mae_test_i = mean_absolute_error(y_test_i, y_pred_test_i)
            y_train_mae_dict[i] = mae_train_i.round(3)
            y_test_mae_dict[i] = mae_test_i.round(3) 
        
        




    # 使用plotly画图
    title = f'{data_name}<br>'\
            f'Train set: Pearson r={pearson_train:.3f}, MAE={mae_train:.3f}, RMSE={rmse_train:.3f}, R2={r2_train:.3f}<br>' \
            f'Test set: Pearson r={pearson_test:.3f}, MAE={mae_test:.3f}, RMSE={rmse_test:.3f}, R2={r2_test:.3f}<br>'
    if each_class_mae:
        title += f'Train set: Each class MAE: {y_test_mae_dict}<br>' 
    if content!= "":
        title += f'{content}<br>'
    

    fig = go.Figure()
    # train data
    fig.add_trace(go.Scatter(x=y_test, y=y_pred_test, mode='markers', name='test set Predicted', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=y_train, y=y_pred_train, mode='markers', name='train set Predicted', marker=dict(color='green')))

    fig.update_layout(xaxis_title='Actual', yaxis_title='Predicted', title=title)
    fig.add_shape(type="line", x0=min(y_test), y0=min(y_test), x1=max(y_test), y1=max(y_test), line=dict(color="green", width=1))
    if save_dir is not None:
        fig.write_image(f'{save_dir}/{data_name}_train_test_pred.png',width=1800, height=1200)
    else:
        return 0
    
def train_val_and_test_pred_plot(y_train,y_val,y_test,y_pred_train,y_pred_val,y_pred_test, data_name="",save_dir=None,each_class_mae=True,content = ""):
    import plotly.graph_objects as go
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    
    
    # 计算指标
    pearson_train, _ = pearsonr(y_train, y_pred_train)
    pearson_val, _ = pearsonr(y_val, y_pred_val)
    pearson_test, _ = pearsonr(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    r2_test = r2_score(y_test, y_pred_test)
    

    y_train_mae_dict = {}
    y_val_mae_dict = {}
    y_test_mae_dict = {}

    if each_class_mae:
        for i in np.unique(y_test):
            
            # indes_train = np.where(y_train == i)[0]
            # indes_val = np.where(y_val == i)[0]
            indes_test = np.where(y_test == i)[0]
            # y_train_i = y_train[indes_train]
            # y_pred_train_i = y_pred_train[indes_train]
            # y_val_i = y_val[indes_val]
            # y_pred_val_i = y_pred_val[indes_val]
            y_test_i = y_test[indes_test]
            y_pred_test_i = y_pred_test[indes_test]
            

            # mae_train_i = mean_absolute_error(y_train_i, y_pred_train_i)
            # mae_val_i = mean_absolute_error(y_val_i, y_pred_val_i)
            mae_test_i = mean_absolute_error(y_test_i, y_pred_test_i)
            # y_train_mae_dict[i] = mae_train_i.round(3)
            # y_val_mae_dict[i] = mae_val_i.round(3)
            y_test_mae_dict[i] = mae_test_i.round(3) 
        
        




    # 使用plotly画图
    title = f'{data_name}<br>'\
            f'Train set: Pearson r={pearson_train:.3f}, MAE={mae_train:.3f}, RMSE={rmse_train:.3f}, R2={r2_train:.3f}<br>' \
            f'Val set: Pearson r={pearson_val:.3f}, MAE={mae_val:.3f}, RMSE={rmse_val:.3f}, R2={r2_val:.3f}<br>'\
            f'Test set: Pearson r={pearson_test:.3f}, MAE={mae_test:.3f}, RMSE={rmse_test:.3f}, R2={r2_test:.3f}<br>'
    if each_class_mae:
        title += f'Test set: Each class MAE: {y_test_mae_dict}<br>' 
    if content!= "":
        title += f'{content}<br>'
    

    fig = go.Figure()
    # train data
    fig.add_trace(go.Scatter(x=y_test, y=y_pred_test, mode='markers', name='test set Predicted', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=y_val, y=y_pred_val, mode='markers', name='val set Predicted', marker=dict(color='orange')))
    fig.add_trace(go.Scatter(x=y_train, y=y_pred_train, mode='markers', name='train set Predicted', marker=dict(color='green')))

    fig.update_layout(xaxis_title='Actual', yaxis_title='Predicted', title=title)
    fig.add_shape(type="line", x0=min(y_test), y0=min(y_test), x1=max(y_test), y1=max(y_test), line=dict(color="green", width=1))
    if save_dir is not None:
        fig.write_image(f'{save_dir}/{data_name}_train_test_pred.png',width=1800, height=1200)
    else:
        fig.show()
        return 0
    
def pred_plot(y_test,y_pred_test, data_name="",save_dir=None,each_class_mae=True,content = ""):
    import plotly.graph_objects as go
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    
    
    # 计算指标

    pearson_test, _ = pearsonr(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    


    y_test_mae_dict = {}

    if each_class_mae:
        for i in np.unique(y_test):
            

            indes_test = np.where(y_test == i)[0]
            y_test_i = y_test[indes_test]
            y_pred_test_i = y_pred_test[indes_test]
            mae_test_i = mean_absolute_error(y_test_i, y_pred_test_i)
            y_test_mae_dict[i] = mae_test_i.round(3) 
        
        




    # 使用plotly画图
    title = f'{data_name}<br>'\
            f'Test set: Pearson r={pearson_test:.3f}, MAE={mae_test:.3f}, RMSE={rmse_test:.3f}, R2={r2_test:.3f}<br>'
    if each_class_mae:
        title += f'Test set: Each class MAE: {y_test_mae_dict}<br>' 
    if content!= "":
        title += f'{content}<br>'
    

    fig = go.Figure()
    # train data
    fig.add_trace(go.Scatter(x=y_test, y=y_pred_test, mode='markers', name='test set Predicted', marker=dict(color='blue')))

    fig.update_layout(xaxis_title='Actual', yaxis_title='Predicted', title=title)
    fig.add_shape(type="line", x0=min(y_test), y0=min(y_test), x1=max(y_test), y1=max(y_test), line=dict(color="green", width=1))
    if save_dir is not None:
        fig.write_image(f'{save_dir}/{data_name}_train_test_pred.png',width=1800, height=1200)
    else:
        return 0

def plot_multiclass_line(data_name,X,y,save_dir=None):
    """
    data_name : str, 数据集名称
    X : numpy array, shape (n_samples, n_features)
    y : numpy array, shape (n_samples,)
    """
    import plotly.graph_objs as go
    import numpy as np    
    import random

    def generate_random_colors(num_colors):
        colors = []
        for _ in range(num_colors):
            # Generate random RGB values
            red = random.randint(0, 255)
            green = random.randint(0, 255)
            blue = random.randint(0, 255)
            # Format as hexadecimal color code
            color = "#{:02x}{:02x}{:02x}".format(red, green, blue)
            colors.append(color)
        return colors

    # Example usage: generate 10 random colors
    colors_len = np.unique(y).shape[0]
    colors = generate_random_colors(colors_len)
    y_unique = np.unique(y)

    fig = go.Figure()
    for value in y_unique:
        indes = np.where(y == value)[0]
        X_axis = [list(range(X.shape[1]))+[None] for i in indes]
        
        all_data=[]
        for index in indes:
            all_data.append([ list(X[index, :]) + [None]])

        fig.add_trace(go.Scatter(
            x=np.array(X_axis).flatten(),
            y=np.array(all_data).flatten(),
            mode='lines',
            name=f'y = {value}',
            line=dict(color=colors[np.where(y_unique == value)[0][0]])  # 可以自定义颜色
        ))

    # 更新布局
    fig.update_layout(
        title='Line plot of '+ str(data_name) ,
        xaxis_title='Data Point Index',
        yaxis_title='Feature Value',
        legend_title='Samples',
        showlegend=True  # 添加这一行以显示图例
    )
    import datetime
    now = datetime.datetime.now()
    if save_dir is not None:
        # fig.write_image(save_dir+data_name+'_multiclass_line.png',width=1800, height=1200)
        fig.write_image(f'{save_dir}/{data_name}_{now.strftime("%Y-%m-%d_%H-%M-%S")}_multiclass_line.png',width=1800, height=1200)

    else:
        fig.show()

if __name__ == '__main__':


    # Assume each dataset has 100 observations
    num_samples = 100

    # Generating random data for training, validation, and testing
    np.random.seed(0)
    y_train = np.random.rand(num_samples)
    y_train_pred = np.random.rand(num_samples)
    y_val = np.random.rand(num_samples)
    y_val_pred = np.random.rand(num_samples)
    y_test = np.random.rand(num_samples)
    y_test_pred = np.random.rand(num_samples)


    
    ####### example:

    ###### analyze_model_performance
    analyze_model_performance(y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred)



