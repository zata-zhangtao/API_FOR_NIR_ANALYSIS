"""
Spectral data analysis functions for NIR spectroscopy.

This module provides functions for analyzing spectral data including
outlier detection, PCA analysis, correlation analysis, and reporting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

# Set font configuration for Chinese characters (optional)
try:
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    # Fallback to default fonts if SimHei is not available
    pass


__all__ = [
    'analyze_spectral_data',
    'load_spectral_data',
    'print_basic_data_info',
    'plot_spectral_overview', 
    'detect_outliers',
    'plot_detailed_spectral_analysis',
    'analyze_correlations',
    'plot_mean',
    'plot_correlation_graph'
]


# === Comprehensive Spectral Data Analysis Functions ===
# This module has been refactored to break down the monolithic analyze_spectral_data
# function into smaller, focused components for better maintainability.


def load_spectral_data(file_path: str) -> tuple:
    """
    Load spectral data from Excel file.
    
    Args:
        file_path: Path to Excel file with 'data' and 'notes' sheets
        
    Returns:
        Tuple of (data_df, info_df, X, y) where:
        - data_df: Main data DataFrame
        - info_df: Notes/remarks DataFrame  
        - X: Feature matrix
        - y: Target variable
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the file format is invalid or required sheets are missing
        KeyError: If required columns are missing
    """
    import os
    
    # Input validation
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")
    
    if not file_path.strip():
        raise ValueError("file_path cannot be empty")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.lower().endswith(('.xlsx', '.xls')):
        raise ValueError("file_path must be an Excel file (.xlsx or .xls)")
    
    try:
        # Try to load the required sheets
        data_df = pd.read_excel(file_path, sheet_name='数据')
        info_df = pd.read_excel(file_path, sheet_name='备注')
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {str(e)}. Make sure the file contains '数据' and '备注' sheets.")
    
    # Validate data structure
    if data_df.empty:
        raise ValueError("Data sheet is empty")
    
    if data_df.shape[1] < 2:
        raise ValueError("Data sheet must have at least 2 columns (features and target)")
    
    # Separate features (X) and target variable (y)
    X = data_df.iloc[:, :-1]
    y = data_df.iloc[:, -1]
    
    # Check for completely missing target values
    if y.isnull().all():
        raise ValueError("Target variable contains only missing values")
    
    return data_df, info_df, X, y


def print_basic_data_info(data_df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> None:
    """
    Print basic information about the dataset.
    
    Args:
        data_df: Complete dataset
        X: Feature matrix
        y: Target variable
        
    Raises:
        TypeError: If inputs are not the expected types
        ValueError: If inputs are empty or invalid
    """
    # Input validation
    if not isinstance(data_df, pd.DataFrame):
        raise TypeError("data_df must be a pandas DataFrame")
    
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("y must be a pandas Series or DataFrame")
    
    if data_df.empty:
        raise ValueError("data_df cannot be empty")
    
    if X.empty:
        raise ValueError("X cannot be empty")
    
    if len(y) == 0:
        raise ValueError("y cannot be empty")
    
    try:
        print("="*50)
        print("1. Basic Data Information")
        print("="*50)
        print(f"Data dimensions: {data_df.shape}")
        print(f"\nNumber of features: {X.shape[1]}")
        print("\nFirst 5 rows:")
        print(data_df.head())
        
        # Data statistical description
        print("\nData statistical description:")
        print(data_df.describe())
        
        # Check missing values
        print("\nMissing values count:")
        print(data_df.isnull().sum())
        
    except Exception as e:
        raise RuntimeError(f"Error printing data information: {str(e)}")


def plot_spectral_overview(X: pd.DataFrame) -> None:
    """
    Plot spectral data overview including individual spectra and mean ± std.
    
    Args:
        X: Feature matrix (spectral data)
    """
    # Plot spectral data
    plt.figure(figsize=(12, 6))
    for i in range(len(X)):
        plt.plot(X.columns, X.iloc[i, :], alpha=0.1)
    plt.title('Spectral Data Distribution')
    plt.xlabel('Wavelength/Frequency')
    plt.ylabel('Spectral Value')
    plt.show()
    
    # Calculate and plot mean spectrum and standard deviation
    mean_spectrum = X.mean()
    std_spectrum = X.std()
    
    plt.figure(figsize=(12, 6))
    plt.plot(X.columns.astype(float), mean_spectrum, 'b-', label='Mean Spectrum')
    plt.fill_between(X.columns.astype(float), 
                     mean_spectrum - 2*std_spectrum,
                     mean_spectrum + 2*std_spectrum,
                     alpha=0.3, label='±2σ Range')
    plt.title('Average Spectrum with Standard Deviation')
    plt.xlabel('Wavelength/Frequency')
    plt.ylabel('Spectral Value')
    plt.legend()
    plt.show()


def analyze_spectral_data(file_path):
    '''
    
    example:
         file_path = r"C:\BaiduSyncdisk\0A-ZATA\data\光谱数据\王兵血糖\郑雷-18.xlsx"
         data_df, info_df, outlier_indices, pca = analyze_spectral_data(file_path)
    
    '''

    data_df = pd.read_excel(file_path, sheet_name='数据')
    info_df = pd.read_excel(file_path, sheet_name='备注')

    # 分离特征(X)和目标变量(y)
    X = data_df.iloc[:, :-1]
    y = data_df.iloc[:, -1]

    print("="*50)
    print("1. 基础数据信息")
    print("="*50)
    print(f"数据维度: {data_df.shape}")
    print("\n特征数量:", X.shape[1])
    print("\n前5行数据:")
    print(data_df.head())

    # 数据统计描述
    print("\n数据统计描述:")
    print(data_df.describe())

    # 检查缺失值
    print("\n缺失值统计:")
    print(data_df.isnull().sum())

    # 绘制光谱图
    plt.figure(figsize=(12, 6))
    for i in range(len(X)):
        plt.plot(X.columns, X.iloc[i, :], alpha=0.1)
    plt.title('光谱数据分布')
    plt.xlabel('波长/频率')
    plt.ylabel('光谱值')
    plt.show()

    # 计算并绘制平均光谱和标准差
    mean_spectrum = X.mean()
    std_spectrum = X.std()

    plt.figure(figsize=(12, 6))
    plt.plot(X.columns.astype(float), mean_spectrum, 'b-', label='平均光谱')
    plt.fill_between(X.columns.astype(float), 
                     mean_spectrum - 2*std_spectrum,
                     mean_spectrum + 2*std_spectrum,
                     alpha=0.2,
                     label='±2倍标准差区间')
    plt.title('平均光谱和变异范围')
    plt.xlabel('波长/频率')
    plt.ylabel('光谱值')
    plt.legend()
    plt.show()

    # 异常值检测
    print("\n="*50)
    print("2. 异常值检测")
    print("="*50)

    # 使用椭圆包络法检测异常值
    outlier_detector = EllipticEnvelope(contamination=0.1, random_state=42)
    outliers = outlier_detector.fit_predict(X)
    outlier_indices = np.where(outliers == -1)[0]

    if len(outlier_indices) > 0:
        print(f"\n检测到 {len(outlier_indices)} 个可能的异常样本")
        print("异常样本索引:", outlier_indices)

        # 绘制异常值光谱
        plt.figure(figsize=(12, 6))
        # 绘制正常样本
        for i in range(len(X)):
            if i not in outlier_indices:
                plt.plot(X.columns, X.iloc[i, :], 'b-', alpha=0.1)
        # 绘制异常样本
        for i in outlier_indices:
            plt.plot(X.columns, X.iloc[i, :], 'r-', alpha=0.5, label='异常样本' if i == outlier_indices[0] else "")
        plt.title('光谱数据分布 (红色为可能的异常样本)')
        plt.xlabel('波长/频率')
        plt.ylabel('光谱值')
        plt.legend()
        plt.show()

        # 去除异常值
        X = X.drop(outlier_indices)
        y = y.drop(outlier_indices)
        print(f"\n已去除异常值,剩余样本数: {len(X)}")

    # 重新绘制光谱图,展示更多细节
    print("\n="*50)
    print("2.5 光谱详细分析")
    print("="*50)
    
    plt.figure(figsize=(15, 10))
    
    # 子图1:原始光谱
    plt.subplot(2, 2, 1)
    for i in range(len(X)):
        plt.plot(X.columns.astype(float), X.iloc[i,:], alpha=0.1)
    plt.plot(X.columns.astype(float), X.mean(), 'r-', linewidth=2, label='平均光谱')
    plt.title('原始光谱数据分布')
    plt.xlabel('波长/频率')
    plt.ylabel('光谱值')
    plt.legend()
    
    # 子图2:标准化后的光谱
    plt.subplot(2, 2, 2)
    X_norm = (X - X.mean()) / X.std()
    for i in range(len(X_norm)):
        plt.plot(X_norm.columns.astype(float), X_norm.iloc[i,:], alpha=0.1)
    plt.plot(X_norm.columns.astype(float), X_norm.mean(), 'r-', linewidth=2, label='平均光谱')
    plt.title('标准化后的光谱分布')
    plt.xlabel('波长/频率')
    plt.ylabel('标准化光谱值')
    plt.legend()
    
    # 子图3:一阶导数光谱
    plt.subplot(2, 2, 3)
    X_diff = X.diff(axis=1)
    for i in range(len(X_diff)):
        plt.plot(X.columns.astype(float)[1:], X_diff.iloc[i,1:], alpha=0.1)
    plt.plot(X.columns.astype(float)[1:], X_diff.mean()[1:], 'r-', linewidth=2, label='平均一阶导')
    plt.title('一阶导数光谱')
    plt.xlabel('波长/频率')
    plt.ylabel('一阶导数值')
    plt.legend()
    
    # 子图4:光谱统计特征
    plt.subplot(2, 2, 4)
    plt.plot(X.columns.astype(float), X.mean(), 'b-', label='平均值')
    plt.plot(X.columns.astype(float), X.median(), 'g-', label='中位数')
    plt.fill_between(X.columns.astype(float),
                     X.quantile(0.25),
                     X.quantile(0.75),
                     alpha=0.2,
                     label='四分位数范围')
    plt.title('光谱统计特征')
    plt.xlabel('波长/频率')
    plt.ylabel('光谱值')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # 相关性分析
    print("\n="*50)
    print("3. 相关性分析")
    print("="*50)

    # 计算与目标变量的相关性
    correlation_values = []
    for col in X.columns:
        corr_coef, _ = stats.pearsonr(X[col], y)
        correlation_values.append(abs(corr_coef))
    
    correlations = pd.DataFrame({
        '特征': X.columns,
        '与目标变量的相关性': correlation_values
    })
    correlations = correlations.sort_values('与目标变量的相关性', ascending=False)

    print("\n与目标变量相关性最强的前10个特征:")
    print(correlations.head(10))

    # 绘制所有特征的相关性折线图
    plt.figure(figsize=(12, 6))
    plt.plot(X.columns.astype(float), correlation_values, 'b-', label='相关性')
    
    # 找出极大值点
    from scipy.signal import find_peaks
    # 找出所有特征的极大值点
    peaks, _ = find_peaks(correlation_values)
    peak_x = X.columns.astype(float).values[peaks]
    peak_y = np.array(correlation_values)[peaks]
    
    # 绘制极大值点
    # plt.scatter(peak_x, peak_y, 'ro', label='极大值点')
    plt.scatter(peak_x, peak_y, c='red', marker='*', s=100, label='极大值点')
    
    plt.title('所有特征与目标变量的相关性分布')
    plt.xlabel('波长/频率')
    plt.ylabel('相关性系数')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 打印备注信息
    print("\n="*50)
    print("4. 备注信息")
    print("="*50)
    print(info_df)
    
    # 添加新的分析部分
    print("\n="*50)
    print("5. 数据分布特征分析")
    print("="*50)
    
    # 计算偏度和峰度
    skewness = X.skew()
    kurtosis = X.kurtosis()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(X.columns.astype(float), skewness, 'g-')
    plt.title('光谱数据偏度分布')
    plt.xlabel('波长/频率')
    plt.ylabel('偏度')
    
    plt.subplot(1, 2, 2)
    plt.plot(X.columns.astype(float), kurtosis, 'r-')
    plt.title('光谱数据峰度分布')
    plt.xlabel('波长/频率')
    plt.ylabel('峰度')
    plt.tight_layout()
    plt.show()
    
    # PCA分析
    print("\n="*50)
    print("6. PCA主成分分析")
    print("="*50)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # 计算解释方差比
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # 绘制碎石图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
    plt.title('PCA碎石图')
    plt.xlabel('主成分数量')
    plt.ylabel('解释方差比')
    plt.show()
    
    # 绘制累积方差图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'ro-')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95%阈值')
    plt.title('PCA累积解释方差比')
    plt.xlabel('主成分数量')
    plt.ylabel('累积解释方差比')
    plt.legend()
    plt.show()
    
    # 目标变量分析
    print("\n="*50)
    print("7. 目标变量分析")
    print("="*50)
    plt.figure(figsize=(8, 6))
    
    # 分布直方图
    sns.histplot(y, bins=30, kde=True)
    plt.title('目标变量分布')
    plt.xlabel('值')
    plt.ylabel('频次')
    
    # 添加偏度峰度文本
    plt.text(0.05, 0.95, f'偏度: {y.skew():.3f}\n峰度: {y.kurtosis():.3f}',
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    print("\n目标变量统计描述：")
    print(y.describe())
    print("\n目标变量偏度和峰度：")
    print(f"偏度: {y.skew():.3f}")
    print(f"峰度: {y.kurtosis():.3f}")
    # 生成分析报告
    print("\n="*50)
    print("8. 数据分析结论")
    print("="*50)
    
    print("\n1. 数据基本特征：")
    print(f"- 样本数量：{len(data_df)}个")
    print(f"- 特征数量：{X.shape[1]}个")
    print(f"- 缺失值情况：{'存在' if data_df.isnull().sum().sum() > 0 else '无'}缺失值")
    
    print("\n2. 异常值分析：")
    print(f"- 检测到{len(outlier_indices)}个异常样本")
    print("- 异常样本占比：{:.2f}%".format(len(outlier_indices)/len(data_df)*100))
    
    print("\n3. 相关性分析结论：")
    top_corr = correlations.iloc[0]
    print(f"- 与目标变量相关性最强的波长为：{top_corr['特征']}")
    print(f"- 最强相关系数为：{top_corr['与目标变量的相关性']:.3f}")
    
    print("\n4. PCA分析结论：")
    n_components_95 = np.where(cumulative_variance_ratio >= 0.95)[0][0] + 1
    print(f"- 保留95%信息需要的主成分数量：{n_components_95}")
    print(f"- 第一主成分解释方差比：{explained_variance_ratio[0]:.3f}")
    print(f"- 前3个主成分累积解释方差比：{cumulative_variance_ratio[2]:.3f}")
    
    print("\n5. 光谱特征分布：")
    print("- 偏度范围：{:.2f} 到 {:.2f}".format(skewness.min(), skewness.max()))
    print("- 峰度范围：{:.2f} 到 {:.2f}".format(kurtosis.min(), kurtosis.max()))
    
    print("\n6. 目标变量特征：")
    print(f"- 范围：{y.min():.2f} 到 {y.max():.2f}")
    print(f"- 平均值：{y.mean():.2f}")
    print(f"- 标准差：{y.std():.2f}")
    print(f"- 变异系数：{y.std()/y.mean():.3f}")


    print("\n="*50)
    print("9. 建模分析")
    print("="*50)




    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义模型列表
    models = {
        'PLS回归': PLSRegression(n_components=10),
        '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf'),
        '线性回归': LinearRegression()
    }
    # 评估结果存储
    results = {}
    
    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()  # 将axes转换为一维数组，便于索引
    
    print("\n各模型性能评估:")
    for i, (name, model) in enumerate(models.items()):
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算评估指标
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results[name] = {
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        }
        
        print(f"\n{name}模型评估结果:")
        print(f"- R2得分: {r2:.3f}")
        print(f"- RMSE: {rmse:.3f}")
        print(f"- MAE: {mae:.3f}")
        
        # 在子图中绘制散点图
        axes[i].scatter(y_test, y_pred, alpha=0.5)
        y_test_array = np.array(y_test)
        axes[i].plot([y_test_array.min(), y_test_array.max()], [y_test_array.min(), y_test_array.max()], 'r--', lw=2)
        axes[i].set_xlabel('实际值')
        axes[i].set_ylabel('预测值')
        axes[i].set_title(f'{name}预测结果散点图')
        
        # 在子图中添加评估指标文本
        axes[i].text(0.05, 0.95, f'R2 = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}', 
                    transform=axes[i].transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')
    
    plt.tight_layout()  # 调整子图布局
    plt.show()

    # 找出最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['R2'])
    print(f"\n最佳模型为: {best_model[0]}")
    print(f"- R2得分: {best_model[1]['R2']:.3f}")
    print(f"- RMSE: {best_model[1]['RMSE']:.3f}")
    print(f"- MAE: {best_model[1]['MAE']:.3f}")

    return data_df, info_df, outlier_indices, pca


def plot_mean(data_name,X_name,y_name, data_X, data_y, scale=True,draw_y=True, save_dir=None):
    data_X_mean = np.mean(data_X, axis=1)
    if scale:
        data_X_mean = (data_X_mean-np.min(data_X_mean))/(np.max(data_X_mean)-np.min(data_X_mean))
        data_y = (data_y-np.min(data_y))/(np.max(data_y)-np.min(data_y))

    fig = go.Figure()

    # 设置不同颜色
    fig.add_trace(go.Scatter(x=np.arange(len(data_X_mean)),
                                y=data_X_mean,
                                mode='lines+markers',
                                marker=dict(color='blue'),
                                line=dict(color='blue'),
                                text=data_y, 
                                name=X_name))
    if draw_y:
        fig.add_trace(go.Scatter(x=np.arange(len(data_X_mean)),
                                y=data_y,
                                mode='lines',
                                name=y_name))
        # 设置图表布局，包括标题
    fig.update_layout(
        title=f"Mean of {data_name}",
        xaxis_title="X Axis Title",
        yaxis_title="Y Axis Title"
    )

    if save_dir is None:
        fig.show()
    else:
        fig.write_image( save_dir+ f'/{data_name}.png',width=1800, height=1200)

def plot_correlation_graph(X, Y,save_dir=None):
    import numpy as np
    import plotly.graph_objs as go
    from scipy.stats import pearsonr, spearmanr
    # 计算皮尔逊相关系数和斯皮尔曼相关系数
    pearson_corrs = []
    spearman_corrs = []

    for feature_idx in range(X.shape[1]):
        pearson_corr, _ = pearsonr(X[:, feature_idx], Y)
        spearman_corr, _ = spearmanr(X[:, feature_idx], Y)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)

    # 准备绘制数据
    trace_pearson = go.Scatter(
        x=list(range(X.shape[1])),
        y=pearson_corrs,
        mode='lines',
        name='Pearson Correlation'
    )

    trace_spearman = go.Scatter(
        x=list(range(X.shape[1])),
        y=spearman_corrs,
        mode='lines',
        name='Spearman Correlation'
    )

    data = [trace_pearson, trace_spearman]

    # 设置布局和绘制
    layout = go.Layout(
        title='Correlation Coefficients between Features and y',
        xaxis=dict(title='Feature Index'),
        yaxis=dict(title='Correlation Coefficient')
    )

    fig = go.Figure(data=data, layout=layout)
    if save_dir is not None:
        fig.write_image(save_dir+'correlation_coefficients.png')
    fig.show()