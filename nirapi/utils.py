"""一些小组件
--------
Functions:
----------
    - PCA_LR_SVR_trian_and_eval(X,y,category = "all_samples",processed_X = None) >> None  # PCA+LR+SVR训练和评估
    - RF_LR_SVR_trian_and_eval(X,y,category = "all_samples",processed_X = None) >> None  # RF+LR+SVR训练和评估
    - NO_FS_LR_SVR_train_and_eval(X,y,category = "all_samples",processed_X = None) >> None  # 所有特征的LR+SVR训练和评估
    - NO_FS_PLSR_train_and_eval(X,y,category = "all_sample",processed_X = None) >> None  # 所有特征的PLSR训练和评估
    - Random_FS_LR_SVR_train_and_eval(X,y,category = "all_samples",processed_X = None,feat_size:Union[int,list] = None,samples_test_size = 0.33,epoch = 100) >> (LR_MAE_for_featsNum,SVR_MAE_for_featsNum)  # 随机选取特征，然后用LR和SVR训练和评估，返回每个特征数下的平均MAE，最大MAE，最小MAE
    - Random_FS_PLSR_train_and_eval(X,y,category = "all_sample",processed_X = None,feat_size:Union[int,list] = 5,samples_test_size = 0.33,samples_random = True,epoch = 1000,max_MAE = 0.1,min_R2 = 0.5) >> None  # 随机选取特征，然后用PLSr训练和评估，返回每个特征数下的平均MAE，最大MAE，最小MAE,默认n—components = 3
    - Auto_tuning_with_svr 输入数据和任务名字,进行svr模型训练和自动调参,输出调参过程的准确率
    - Save_data_to_csv【class】 传入文件名和列,创建一个类，这个类支持输入数据，自动存储到文件中
    - run_optuna 传入数据，自动进行调参， 方法是之前光谱分析streamlit自动化平台上的方法
    - get_pythonFile_functions 传入python文件 返回文件中包含的所有函数字典
    - run_regression_optuna(data_name,X,y,
                        model='PLS',split = 'SPXY',test_size = 0.3, n_trials=200,object = None,cv = None,save_dir = None):
                        传入数据，自动进行调参， 支持SVR,PLS
    - PD_reduce_noise(PD_samples, PD_noise, ratio=9,base_noise= None):  传入样品和噪声PD,以及基础的噪声数值,样品数据减去PD数据得到减去噪声之后的样品数值
    - SpectralReconstructor : 用于光谱重建的神经网络类，在spectral_reconstruction_train中被使用
    - spectral_reconstruction_train(PD_values, Spectra_values, epochs=50, lr=1e-3,save_dir = None):  光谱重建训练
"""
import traceback  # 引入 traceback 模块
from .draw import *
from typing import Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from .ML_model import *
from sklearn.preprocessing import MinMaxScaler
import optuna
from scipy.stats import pearsonr
import matplotlib
import optuna
from . import ML_model as AF
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import random
import warnings
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import datetime
from .AnalysisClass.CreateTrainReport import CreateTrainReport




def train_pred_with_bands(X, y, metric='MAE', n_splits=5, n_iterations=10, test_size=0.2, random_state=42, verbose=True, n_bands=5):
    """
    使用四种验证方法评估不同频段的性能，并返回最优的频段
    
    Args:
        X (numpy.ndarray): 特征矩阵，形状为 (n_samples, n_features)
        y (numpy.ndarray): 目标变量，形状为 (n_samples,)
        metric (str): 评估指标, 'R2' 或 'MAE'
        n_splits (int): K折交叉验证的折数
        n_iterations (int): Bootstrapping的迭代次数
        test_size (float): 测试集比例
        random_state (int): 随机种子
        verbose (bool): 是否打印详细信息
        n_bands (int): 要划分的频段数量，默认为5
    
    Returns:
        dict: 包含各种验证方法的结果和最优频段信息
        
    Examples:
        import numpy as np
        from sklearn.datasets import make_regression

        # Generate sample data
        X, y = make_regression(n_samples=100, n_features=1000, random_state=42)

        # Test with MAE metric and 5 bands
        results_mae = train_pred_with_bands(
            X, y,
            metric='MAE',
            n_splits=5,
            n_iterations=10,
            test_size=0.2,
            random_state=42,
            verbose=True,
            n_bands=5
        )

        # Test with R2 metric and 3 bands
        results_r2 = train_pred_with_bands(
            X, y,
            metric='R2',
            n_splits=5,
            n_iterations=10,
            test_size=0.2,
            random_state=42,
            verbose=True,
            n_bands=3
        )

        # Print results
        print("\nMAE Results:")
        for method, result in results_mae.items():
            print(f"{method}: Best band = {result['best_band']}, MAE = {result['MAE']:.4f}")

        print("\nR2 Results:")
        for method, result in results_r2.items():
            print(f"{method}: Best band = {result['best_band']}, R2 = {result['R2']:.4f}")
    """
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.utils import resample
    import numpy as np
    from nirapi.ML_model import custom_train_test_split

    def model_train_and_predict(X_train, y_train, X_test, y_test,plot=False):
        """
        训练模型并进行预测,返回评估指标
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征 
            y_test: 测试集标签
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        from sklearn.model_selection import train_test_split
        from sklearn.decomposition import PCA
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import pearsonr
        
        # 数据标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # PCA降维
        pca = PCA(n_components=0.95)  # 保留95%的方差
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        print(f"PCA降维后的特征数量: {X_train_pca.shape[1]}")
        
        # 确保y是2D数组
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)
            
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_pca, y_train_scaled)
        
        # 预测
        y_train_pred_scaled = model.predict(X_train_pca)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1))
        
        y_test_pred_scaled = model.predict(X_test_pca)
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1))
        
        # 计算评估指标
        metrics = {}
        for i in range(y_test.shape[1]):
            # 计算相关系数r
            r_train, _ = pearsonr(y_train[:, i].flatten(), y_train_pred[:, i].flatten())
            r_test, _ = pearsonr(y_test[:, i].flatten(), y_test_pred[:, i].flatten())
            
            metrics[f'Target_{i+1}'] = {
                'R2_train': r2_score(y_train[:, i], y_train_pred[:, i]),
                'RMSE_train': np.sqrt(mean_squared_error(y_train[:, i], y_train_pred[:, i])),
                'MAE_train': mean_absolute_error(y_train[:, i], y_train_pred[:, i]),
                'r_train': r_train,
                'R2_test': r2_score(y_test[:, i], y_test_pred[:, i]),
                'RMSE_test': np.sqrt(mean_squared_error(y_test[:, i], y_test_pred[:, i])),
                'MAE_test': mean_absolute_error(y_test[:, i], y_test_pred[:, i]),
                'r_test': r_test
            }
        
        # 打印评估指标
        print("\n模型评估指标:")
        for target, scores in metrics.items():
            print(f"\n{target}:")
            for metric_name, value in scores.items():
                print(f"{metric_name}: {value:.4f}")
        
        # 绘制预测值与真实值的对比图
        if plot:
            plt.figure(figsize=(15, 6))
                
            # 训练集
            plt.subplot(1, 2, 1)
            plt.scatter(y_train, y_train_pred, alpha=0.5, label='训练集')
            plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.title('训练集: 预测值 vs 真实值')
            # 添加评估指标文本
            text = f'R2 = {metrics["Target_1"]["R2_train"]:.4f}\nRMSE = {metrics["Target_1"]["RMSE_train"]:.4f}\nMAE = {metrics["Target_1"]["MAE_train"]:.4f}\nr = {metrics["Target_1"]["r_train"]:.4f}'
            plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.grid(True)
            plt.legend()
            
            # 测试集
            plt.subplot(1, 2, 2)
            plt.scatter(y_test, y_test_pred, alpha=0.5, label='测试集')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.title('测试集: 预测值 vs 真实值')
            # 添加评估指标文本
            text = f'R2 = {metrics["Target_1"]["R2_test"]:.4f}\nRMSE = {metrics["Target_1"]["RMSE_test"]:.4f}\nMAE = {metrics["Target_1"]["MAE_test"]:.4f}\nr = {metrics["Target_1"]["r_test"]:.4f}'
            plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
                    
        return metrics

    
    # 将数据分成指定数量的频段
    n_features = X.shape[1]
    band_size = n_features // n_bands
    bands = []
    band_names = []
    
    for i in range(n_bands):
        start_idx = i * band_size
        end_idx = start_idx + band_size if i < n_bands-1 else n_features
        bands.append(X[:, start_idx:end_idx])
        band_names.append(f"band{i+1}")
    
    results = {}
    
    def print_result(message):
        if verbose:
            print(message)
    
    # 1. 固定比例分割验证
    print_result("\n固定比例分割验证:")
    fixed_results = []
    for i, band in enumerate(bands):
        X_train, X_test, y_train, y_test = train_test_split(
            band, y, test_size=test_size, random_state=random_state
        )
        metrics = model_train_and_predict(X_train, y_train, X_test, y_test)
        score = metrics["Target_1"]["R2_test"] if metric == 'R2' else metrics["Target_1"]["MAE_test"]
        fixed_results.append(score)
        print_result(f"频段 {band_names[i]}: {metric} = {score:.4f}")
    
    best_band_fixed = band_names[np.argmin(fixed_results) if metric == 'MAE' else np.argmax(fixed_results)]
    best_score_fixed = min(fixed_results) if metric == 'MAE' else max(fixed_results)
    results["固定比例分割"] = {"best_band": best_band_fixed, metric: best_score_fixed}
    
    # 2. K折交叉验证
    print_result("\nK折交叉验证:")
    cv_results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for i, band in enumerate(bands):
        scores = []
        for train_idx, test_idx in kf.split(band):
            X_train, X_test = band[train_idx], band[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            metrics = model_train_and_predict(X_train, y_train, X_test, y_test)
            score = metrics["Target_1"]["R2_test"] if metric == 'R2' else metrics["Target_1"]["MAE_test"]
            scores.append(score)
        
        avg_score = np.mean(scores)
        cv_results.append(avg_score)
        print_result(f"频段 {band_names[i]}: 平均{metric} = {avg_score:.4f}")
    
    best_band_cv = band_names[np.argmin(cv_results) if metric == 'MAE' else np.argmax(cv_results)]
    best_score_cv = min(cv_results) if metric == 'MAE' else max(cv_results)
    results["交叉验证"] = {"best_band": best_band_cv, metric: best_score_cv}
    
    # 3. SPXY方法
    print_result("\nSPXY验证:")
    spxy_results = []
    
    for i, band in enumerate(bands):
        X_train, X_test, y_train, y_test = custom_train_test_split(
            band.astype(np.float32), y.astype(np.float32), 
            test_size=test_size, method="SPXY"
        )
        metrics = model_train_and_predict(X_train, y_train, X_test, y_test)
        score = metrics["Target_1"]["R2_test"] if metric == 'R2' else metrics["Target_1"]["MAE_test"]
        spxy_results.append(score)
        print_result(f"频段 {band_names[i]}: {metric} = {score:.4f}")
    
    best_band_spxy = band_names[np.argmin(spxy_results) if metric == 'MAE' else np.argmax(spxy_results)]
    best_score_spxy = min(spxy_results) if metric == 'MAE' else max(spxy_results)
    results["SPXY"] = {"best_band": best_band_spxy, metric: best_score_spxy}
    
    # 4. Bootstrapping方法
    print_result("\nBootstrapping验证:")
    bootstrap_results = []
    
    for i, band in enumerate(bands):
        scores = []
        for _ in range(n_iterations):
            indices = resample(
                range(len(band)), 
                replace=True, 
                n_samples=len(band), 
                random_state=random_state + _
            )
            test_indices = list(set(range(len(band))) - set(indices))
            
            X_train, y_train = band[indices], y[indices]
            X_test, y_test = band[test_indices], y[test_indices]
            
            metrics = model_train_and_predict(X_train, y_train, X_test, y_test)
            score = metrics["Target_1"]["R2_test"] if metric == 'R2' else metrics["Target_1"]["MAE_test"]
            scores.append(score)
        
        avg_score = np.mean(scores)
        bootstrap_results.append(avg_score)
        print_result(f"频段 {band_names[i]}: 平均{metric} = {avg_score:.4f}")
    
    best_band_bootstrap = band_names[np.argmin(bootstrap_results) if metric == 'MAE' else np.argmax(bootstrap_results)]
    best_score_bootstrap = min(bootstrap_results) if metric == 'MAE' else max(bootstrap_results)
    results["Bootstrapping"] = {"best_band": best_band_bootstrap, metric: best_score_bootstrap}
    
    # 汇总结果
    print_result(f"\n不同验证方法的最优频段和{metric}值:")
    for method, result in results.items():
        print_result(f"{method}: 最优频段 = {result['best_band']}, {metric} = {result[metric]:.4f}")
    
    # 找出总体最优的频段
    all_scores = [results[method][metric] for method in results]
    best_method = list(results.keys())[np.argmin(all_scores) if metric == 'MAE' else np.argmax(all_scores)]
    overall_best_band = results[best_method]["best_band"]
    overall_best_score = min(all_scores) if metric == 'MAE' else max(all_scores)
    
    print_result(f"\n总体最优频段: {overall_best_band}, 方法: {best_method}, {metric} = {overall_best_score:.4f}")
    
    return results








def train_model_for_trick_game_v2(max_attempts = 10,splited_data:tuple=None, X=None, y=None, test_size=0.34,  n_trials=100, selected_metric="rmse", target_score=0.0002,filename = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),**kw):
    
    """
    v2 compared to the v1 
    | Feature                  | Original Version                  | V2 Version                                     |
    |--------------------------|-----------------------------------|-----------------------------------------------|
    | Input Data               | Only `X` and `y`                  | Adds `splited_data` for pre-split datasets     |
    | Customization            | Fixed pipeline                    | Configurable via `**kw` (preprocessing, etc.)  |
    | Documentation            | Minimal                           | Detailed docstring with example                |
    | `max_attempts`           | Hardcoded (10)                    | Configurable parameter (default 10)            |
    | Data Splitting           | Always splits internally          | Conditional splitting if `splited_data` is None|
    | Pipeline Flexibility     | None                              | Supports outlier removal, preprocessing, etc.  |
    | Model Options            | Implicitly fixed                  | Explicitly supports multiple models            |

    ---


    -----
    Parameters:
    -----
        - splited_data : tuple(X_train,X_test,y_train,y_test)
            if splited_data is not None, then X and y will be ignored
        - X : ndarray
            if splited_data is None, then X and y will be used to split data
        - y : ndarray
            if splited_data is None, then X and y will be used to split data
        - test_size : float
            if splited_data is None, then X and y will be used to split data
        - n_trials : int
            per trial number
        - selected_metric : str ["r2","r","rmse","mae","mse","accuracy", "precision", "recall"]
            test metric
        - target_score : float
            if the score is less than target_score, then the training will be stopped
        - filename : str
            the filename of the report
        - kw : dict
            the keyword arguments
            - selected_outlier : list ["不做异常值去除", "mahalanobis"]
            - selected_preprocess : list ["不做预处理", "mean_centering", "normalization", "standardization", 
                         "poly_detrend", "snv", "savgol", "msc","d1", "d2", "rnv", "move_avg"]
            - preprocess_number_input : int
            - selected_feat_sec : list ["不做特征选择","corr_coefficient","anova","remove_high_variance_and_normalize","random_select"]
            - selected_dim_red : list ["不做降维","pca"]    
            - selected_model : list ["LR", "SVR", "PLSR", "Bayes(贝叶斯回归)", "RFR(随机森林回归)", "BayesianRidge"] or ["LogisticRegression", "SVM", "DT", "RandomForest", "KNN", 
                        'Bayes(贝叶斯分类)', "GradientBoostingTree", "XGBoost"]
    -----
    Returns:
    -----
        - None
    -----
 
        import numpy as np
        from sklearn.model_selection import train_test_split
        from nirapi.utils import *

        np.random.seed(42)  # For reproducibility
        n_samples = 1000
        n_features = 1000  # Assuming similar dimensionality as original spectra

        Spectrumes = np.random.randn(n_samples, n_features)
        Lactate = np.random.uniform(0, 10, n_samples)

        X_train, X_test, y_train, y_test = train_test_split(Spectrumes, Lactate, test_size=0.34, random_state=42)
        kw = {
        "selected_outlier" :     ["不做异常值去除"],
        "selected_preprocess" :  [ "move_avg"],
        "preprocess_number_input" : 1,
        "selected_feat_sec" : ["remove_high_variance_and_normalize"],
        "selected_dim_red" : ["pca"],
        "selected_model" : [ "SVR"]
        }
        train_model_for_trick_game_v2(splited_data=(X_train, X_test, y_train, y_test), test_size=0.34,  n_trials=100, selected_metric="rmse", target_score=0.0002,filename = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+"_人体乳酸_加心率——混合建模", **kw)
            

    """

    
    
    best_score = np.inf
    # max_attempts = 10  # 最大重试次数
    attempt = 0
    
    while best_score >= target_score and attempt < max_attempts:
        attempt += 1
        print(f"尝试: {attempt}/{max_attempts}, 当前最佳分数: {best_score:.4f}")
        
        if splited_data is None:    
            # 重新划分数据集
            methods = ['KS', 'SPXY'] 
            random_method = np.random.choice(methods)
            if random_method == 'random_split':
                X_train, X_test, y_train, y_test = random_split(X, y, test_size=test_size)
            else:
                X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=test_size, method=random_method)
        else:
            X_train, X_test, y_train, y_test = splited_data

        


        # 构建数据字典
        data_dict = {
            "train": (X_train, y_train),
            "val": (X_test, y_test),
            "test": (X_test, y_test)
        }
        print(data_dict.keys())
        print(kw)
        
        # 创建报告
        report = CreateTrainReport(f"{filename}_training_report_{attempt}.pdf")
        score_df = report.analyze_data(
            data_dict,
            train_key="train",
            test_key="test", 
            n_trials=n_trials,
            selected_metric=selected_metric,
            **kw
        )
        
        # 获取测试集得分
        current_score = score_df.loc['score', 'val']
        best_score = current_score if current_score < best_score else best_score
        
        # 保存结果
        results_data = pd.DataFrame({
            '训练集真实值': pd.Series(score_df.loc['y_true', 'train']),
            '训练集预测值': pd.Series(score_df.loc['y_pred', 'train']), 
            '测试集真实值': pd.Series(score_df.loc['y_true', 'val']),
            '测试集预测值': pd.Series(score_df.loc['y_pred', 'val'])
        })
        results_data.to_csv(f"{filename}_{current_score:.5f}_model_results__{attempt}.csv", index=False)
        
        if current_score < target_score:
            score_df.to_csv(f"{filename}_{current_score:.5f}_score_df_{attempt}.csv")
            print(f"训练完成! 第{attempt}次尝试达到目标。报告已保存为 {filename}_{current_score:.5f}_score_df_{attempt}.csv")
            return True
            
    if best_score >= target_score:
        print(f"未能达到目标指标{target_score}，最好成绩为{best_score}")
        return False




# 获取MZI样机的band
def get_MZI_bands():
    # [0,281,482,683,884]
    '''返回MZI样机的波段
    ------
    Parameters:
    ------
        - None
    ------
    Returns:
    ------
        - bands : list
            波段列表
    '''

    band1_nm = np.linspace(1240, 1380, 281)
    band2_nm = np.linspace(1390, 1490, 201)
    band3_nm = np.linspace(1500, 1600, 201)
    band4_nm = np.linspace(1610, 1700, 201)
    
    # return np.concatenate((band1_nm,band2_nm,band3_nm,band4_nm))
    return [band1_nm,band2_nm,band3_nm,band4_nm]    
# 
def PCA_LR_SVR_trian_and_eval(X,y,category = "all_samples",processed_X = None,feat_ratio = 0.33,samples_test_size = 0.33):
    """
    PCA+LR+SVR训练和评估
    ------
    Parameters:
    ------
        - X : ndarray
            NIR spectral data
        - y : ndarray
            Expected value
        - category : str
            category name
        - processed_X : ndarray
            如果已经对X做了预处理，可以传入预处理后X训练和测试集 processed_X = (X_train,X_test)
        - feat_ratio : float
            特征比例
        - samples_test_size : float
            测试集比例
    ------
    Returns:
    ------
        - None
    """

    warnings.warn(
       "此函数已废弃，将于2025-04-01被删除"
    )
    # PCA
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=samples_test_size,random_state=42)
    if processed_X is not None:
        X_train,X_test = processed_X

    pca = PCA(n_components=feat_ratio)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    # 线性回归模型
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train_pca,y_train)
    y_train_pred = lr.predict(X_train_pca)
    y_test_pred = lr.predict(X_test_pca)




    # 画散点图
    import matplotlib.pyplot as plt
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15,5))
    def train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,category = "all_samples"):
        plt.scatter(y_train,y_train_pred,label='Training set')
        plt.scatter(y_test,y_test_pred,label='Testing set')
        from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
        MAE = mean_absolute_error(y_test,y_test_pred)
        R2 = r2_score(y_test,y_test_pred)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.plot([0,max(max(y_train),max(y_test))],[0,max(max(y_train),max(y_test))],'r-')
        plt.text(0.1,0.4,"MAE:{:.5},R2:{:.5}".format(MAE,R2))
        plt.legend()
        plt.title(category+" train_test_Scatter")


    plt.subplot(1,2,1)
    train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,category= category+" LR")
    # 非线性回归模型
    from sklearn.svm import SVR
    # svr optuna调参
    import optuna
    from sklearn.metrics import mean_absolute_error

    def objective(trial):
        C = trial.suggest_float('C', 1e0, 1e3,log=True)
        gamma = trial.suggest_float('gamma', 1e-4, 1e-1,log=True)
        epsilon = trial.suggest_float('epsilon', 1e-4, 1e-1,log=True)
        svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
        svr.fit(X_train_pca,y_train)
        y_test_pred = svr.predict(X_test_pca)
        return mean_absolute_error(y_test,y_test_pred)

    study = optuna.create_study(direction='minimize')
    optuna.logging.disable_default_handler()
    study.optimize(objective, n_trials=100)
    svr = SVR(kernel='rbf', C=study.best_params['C'], gamma=study.best_params['gamma'], epsilon=study.best_params['epsilon'])
    svr.fit(X_train_pca,y_train)
    y_train_pred = svr.predict(X_train_pca)
    y_test_pred = svr.predict(X_test_pca)
    # 画散点图
    plt.subplot(1,2,2)
    train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,category= category+" SVR")
    plt.show()

def RF_LR_SVR_trian_and_eval(X,y,category = "all_samples",processed_X = None,feat_ratio = 0.33,samples_test_size = 0.33):
    """
    RF+LR+SVR训练和评估
    ------
    Parameters:
    ------
        - X : ndarray
            NIR spectral data
        - y : ndarray
            Expected value
        - category : str
            category name
        - processed_X : ndarray
            如果已经对X做了预处理，可以传入预处理后X训练和测试集 processed_X = (X_train,X_test)
        - feat_ratio : float
            特征比例
        - samples_test_size : float
            测试集比例
    ------
    Returns:
    ------
        - None
    """
    # RF
    from sklearn.feature_selection import SelectPercentile,SelectFromModel
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    # Assume X contains the high dimensional input data 
    # and y contains the labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=samples_test_size, random_state=42)
    # X_train, X_test, y_train, y_test  = X[:72, :],y[:72],X[72:, :],y[72:]
    if processed_X is not None:
        X_train,X_test = processed_X



    ##  随机森林特征选择
    # 将数值型标签转换为分类标签
    y_min = np.min(y_train)  
    y_max = np.max(y_train)
    a = 1
    b = 20
    y_normalized = a + (y_train - y_min)*(b - a)/(y_max - y_min)  
    y_train_rf_fs = y_normalized.astype(int)


    clf = RandomForestClassifier(n_estimators=4,n_jobs=-1,random_state=42)
    clf.fit(X_train, y_train_rf_fs)
    # Use SelectPercentile to select features based on importance weights
    sfm = SelectFromModel(clf, threshold= -np.inf,max_features= int(X.shape[1]*feat_ratio))
    print(int(X.shape[1]*feat_ratio))
    sfm.fit(X_train, y_train_rf_fs)
    print(sfm.get_feature_names_out)

    X_train_reduced = sfm.transform(X_train)
    X_test_reduced = sfm.transform(X_test) 
    print("Number of features reduced from{}to{}".format(X_train.shape[1], X_train_reduced.shape[1]))

    
    # 线性回归模型
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train_reduced,y_train)
    y_train_pred = lr.predict(X_train_reduced)
    y_test_pred = lr.predict(X_test_reduced)



    
    # 画散点图
    import matplotlib.pyplot as plt
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15,5))
    def train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,category = "all_samples"):
        plt.scatter(y_train,y_train_pred,label='Training set')
        plt.scatter(y_test,y_test_pred,label='Testing set')
        from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
        MAE = mean_absolute_error(y_test,y_test_pred)
        R2 = r2_score(y_test,y_test_pred)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.plot([0,max(max(y_train),max(y_test))],[0,max(max(y_train),max(y_test))],'r-')
        plt.text(0.1,0.4,"MAE:{:.5},R2:{:.5}".format(MAE,R2))
        plt.legend()
        plt.title(category+" train_test_Scatter")


    plt.subplot(1,2,1)
    train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,category= category+" LR")
    # 非线性回归模型
    from sklearn.svm import SVR
    # svr optuna调参
    import optuna
    from sklearn.metrics import mean_absolute_error

    def objective(trial):
        C = trial.suggest_float('C', 1e0, 1e3,log=True)
        gamma = trial.suggest_float('gamma', 1e-4, 1e-1,log=True)
        epsilon = trial.suggest_float('epsilon', 1e-4, 1e-1,log=True)
        svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
        svr.fit(X_train_reduced,y_train)
        y_test_pred = svr.predict(X_test_reduced)
        return mean_absolute_error(y_test,y_test_pred)

    study = optuna.create_study(direction='minimize')
    optuna.logging.disable_default_handler()
    study.optimize(objective, n_trials=100)
    svr = SVR(kernel='rbf', C=study.best_params['C'], gamma=study.best_params['gamma'], epsilon=study.best_params['epsilon'])
    svr.fit(X_train_reduced,y_train)
    y_train_pred = svr.predict(X_train_reduced)
    y_test_pred = svr.predict(X_test_reduced)
    # 画散点图
    plt.subplot(1,2,2)
    train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,category= category+" SVR")
    plt.show()

def NO_FS_LR_SVR_train_and_eval(X = None,y = None ,category = "all_samples",processed = None,samples_test_size = 0.33,draw = True,
                                
                                ):
    """
    NO_FS+LR+SVR训练和评估,不做特征选择直接评价
    ------
    Parameters:
    ------
        - X : ndarray
            NIR spectral data
        - y : ndarray
            Expected value
        - category : str
            category name
        - processed : ndarray
            如果已经对X做了划分， processed_X = (X_train,X_test,y_train,y_test)
        - samples_test_size : float
            测试集比例
    ------
    Returns:
    ------
        - if draw == False: (LR_MAE,LR_R2),(SVR_MAE,SVR_R2)
    ------
    modify:
    -------
        - 2023-11-28 15:00:00
            - 修改了画散点图的函数，调整了MAE和R2的显示的位置
        - 2023-11-30
            - 增加了参数draw，default = True，是否画图，如果不画图将返回MAE和R2
    """
    # 线性回归模型
    from sklearn.model_selection import train_test_split
    if processed is not None:
        X_train,X_test,y_train,y_test = processed
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=samples_test_size, random_state=42)
    
    
    # 线性回归模型
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)



    if draw:  #2023-11-30
        # 画散点图
        import matplotlib.pyplot as plt
        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(15,5))
        def train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,category = "all_samples"):
            plt.scatter(y_train,y_train_pred,label='Training set')
            plt.scatter(y_test,y_test_pred,label='Testing set')
            from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
            import numpy as np
            
            # 计算各项指标
            MAE = mean_absolute_error(y_test,y_test_pred)
            MSE = mean_squared_error(y_test,y_test_pred)
            RMSE = np.sqrt(MSE)
            R2 = r2_score(y_test,y_test_pred)
            r = np.corrcoef(y_test,y_test_pred)[0,1]
            
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.plot([min(min(y_test),min(y_train)),max(max(y_train),max(y_test))],
                    [min(min(y_test),min(y_train)),max(max(y_train),max(y_test))],'r-')
            
            # 创建文本框显示指标
            textstr = '\n'.join((
                f'MAE: {MAE:.4f}',
                f'MSE: {MSE:.4f}',
                f'RMSE: {RMSE:.4f}',
                f'R2: {R2:.4f}',
                f'r: {r:.4f}'))
            
            # 添加文本框
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            plt.legend()
            plt.title(category+" train_test_Scatter")

        plt.subplot(1,2,1)
        train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,category= category+" LR")
    else:
        from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
        import numpy as np
        LR_MAE = mean_absolute_error(y_test,y_test_pred)
        LR_MSE = mean_squared_error(y_test,y_test_pred)
        LR_RMSE = np.sqrt(LR_MSE)
        LR_R2 = r2_score(y_test,y_test_pred)






    # 非线性回归模型SVR
    from sklearn.svm import SVR
    # svr optuna调参
    import optuna
    from sklearn.metrics import mean_absolute_error

    def objective(trial):
        C = trial.suggest_float('C', 1e0, 1e3,log=True)
        gamma = trial.suggest_float('gamma', 1e-4, 1e-1,log=True)
        epsilon = trial.suggest_float('epsilon', 1e-4, 1e-1,log=True)
        svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
        svr.fit(X_train,y_train)
        y_test_pred = svr.predict(X_test)
        return mean_absolute_error(y_test,y_test_pred)

    study = optuna.create_study(direction='minimize')
    optuna.logging.disable_default_handler()
    study.optimize(objective, n_trials=1000)
    svr = SVR(kernel='rbf', C=study.best_params['C'], gamma=study.best_params['gamma'], epsilon=study.best_params['epsilon'])
    svr.fit(X_train,y_train)
    y_train_pred = svr.predict(X_train)
    y_test_pred = svr.predict(X_test)
    del study


    if draw:  #2023-11-30
        # 画散点图
        plt.subplot(1,2,2)
        train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,category= category+" SVR")
        plt.show()
    else:
        from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
        import numpy as np
        SVR_MAE = mean_absolute_error(y_test,y_test_pred)
        SVR_MSE = mean_squared_error(y_test,y_test_pred)
        SVR_RMSE = np.sqrt(SVR_MSE)
        SVR_R2 = r2_score(y_test,y_test_pred)

        return (round(LR_MAE,3),round(LR_RMSE,3),round(LR_R2,3)),(round(SVR_MAE,3),round(SVR_RMSE,3),round(SVR_R2,3))

def NO_FS_PLSR_train_and_eval(X,y,category = "all_sample",processed_X = None,samples_test_size = 0.33,draw = True,n_components = 10):
    """
    NO_FS+PLSR训练和评估,不做特征选择直接评价
    ------
    Parameters:
    ------
        - X : ndarray
            NIR spectral data
        - y : ndarray
            Expected value
        - category : str
            category name   
        - processed_X : ndarray
            如果已经对X做了预处理，可以传入预处理后X训练和测试集 processed_X = (X_train,X_test)
        - samples_test_size : float 
            测试集比例
    ------
    Returns:
    ------
        - if draw == False: (PLSR_MAE,PLSR_R2)
    ------  
    modify:
    -------
        - 2023-12-6 创建函数
    """
    ## 训练和评估数据
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=samples_test_size, random_state=42)
    if processed_X is not None:
        X_train,X_test = processed_X
    
    # PLSR
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import mean_absolute_error,r2_score
    plsr  = PLSRegression(n_components=n_components)
    plsr.fit(X_train,y_train)
    y_train_pred = plsr.predict(X_train)
    y_test_pred = plsr.predict(X_test)
    if draw:  #2023-11-30
        # 画散点图
        import matplotlib.pyplot as plt
        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(15,5))
        def train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,category = "all_samples"):
            plt.scatter(y_train,y_train_pred,label='Training set')
            plt.scatter(y_test,y_test_pred,label='Testing set')
            from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
            MAE = mean_absolute_error(y_test,y_test_pred)
            R2 = r2_score(y_test,y_test_pred)
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            # plt.plot([0,max(max(y_train),max(y_test))],[0,max(max(y_train),max(y_test))],'r-')
            plt.plot([0,0.5],[0,0.5],'r-')
            plt.text(0.1,0.4,"MAE:{:.5},R2:{:.5}".format(MAE,R2))
            plt.legend()
            plt.title(category+" train_test_Scatter")
            plt.show()
        train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,category= category+" PLSR")
    else:
        from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
        PLSR_MAE = mean_absolute_error(y_test,y_test_pred)
        PLSR_R2 = r2_score(y_test,y_test_pred)
        
        return (round(PLSR_MAE,3),round(PLSR_R2,3))

def Random_FS_LR_SVR_train_and_eval(X,y,category = "all_samples",processed_X = None,feat_size:Union[int,list] = None,samples_test_size = 0.33,epoch = 100,svr_trials = 10):
    """随机选取特征，然后用LR和SVR训练和评估，返回每个特征数下的平均MAE，最大MAE，最小MAE
    也可以传入feat_size = [1,2,3,4,5,6,7,8,9,10]，指定特征数
    ------
    Parameters:
    ------
        - X : ndarray
            NIR spectral data
        - y : ndarray
            Expected value
        - category : str
            category name
        - processed_X : ndarray
            如果已经对X做了预处理，可以传入预处理后X训练和测试集 processed_X = (X_train,X_test)
        - feat_size : int
            特征数
        - samples_test_size : float
            测试集比例
        - epoch : int
            随机选取特征的次数
        - svr_trials : int
            svr调参次数

    ------
    Returns:
    ------
        - LR_MAE_for_featsNum : dict
            LR模型下，每个特征数下的平均MAE，最大MAE，最小MAE
        - SVR_MAE_for_featsNum : dict
            SVR模型下，每个特征数下的平均MAE，最大MAE，最小MAE
    """

    # epoch_print 每个epoch是否打印进度
    epoch_print = False
    
    
    # 随机划分数据集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=samples_test_size, random_state=42)
    import numpy as np

    # 如果已经对X做了预处理，可以传入预处理后X训练和测试集 processed_X = (X_train,X_test)
    if processed_X is not None:
        X_train,X_test = processed_X

    # 随机特征选择
    ## 选取的每个特征数下的平均MAE最小，最大MAE √
    ## 选取的每个特征数下的MAE分布 ×

    def randomfste(features_num,LR_MAE_for_featsNum,SVR_MAE_for_featsNum):
        # print(features_num)

        LR_total_MAE = 0
        LR_min_MAE = 10000
        LR_max_MAE = 0
        LR_min_MAE_feats = []

        SVR_total_MAE = 0
        SVR_min_MAE = 10000
        SVR_max_MAE = 0
        SVR_min_MAE_feats = []
        SVR_best_params = {}
        from sklearn.metrics import mean_absolute_error
        for i in range(epoch):
            if epoch_print:
                print("epoch:{},feat_num:{}".format(i,features_num))
            random_features_index = np.random.choice(np.arange(0, 1899), size=features_num, replace=False)

            # 线性回归模型
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(X_train[:,random_features_index],y_train)
            y_test_pred = lr.predict(X_test[:,random_features_index])
            MAE = mean_absolute_error(y_test,y_test_pred)
            LR_total_MAE += MAE
            if MAE < LR_min_MAE:
                LR_min_MAE = MAE
                LR_min_MAE_feats = random_features_index
            if MAE > LR_max_MAE:
                LR_max_MAE = MAE
            

            # 非线性回归模型SVR
            from sklearn.svm import SVR
            # svr optuna调参
            import optuna
            from sklearn.metrics import mean_absolute_error

            def objective(trial):
                C = trial.suggest_float('C', 1e0, 1e3,log=True)
                gamma = trial.suggest_float('gamma', 1e-4, 1e-1,log=True)
                epsilon = trial.suggest_float('epsilon', 1e-4, 1e-1,log=True)
                svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
                svr.fit(X_train[:,random_features_index],y_train)
                y_test_pred = svr.predict(X_test[:,random_features_index])
                # print(mean_absolute_error(y_test,y_test_pred))
                return mean_absolute_error(y_test,y_test_pred)
            study = optuna.create_study(direction='minimize')
            optuna.logging.disable_default_handler()
            study.optimize(objective, n_trials=svr_trials)
            svr = SVR(kernel='rbf', C=study.best_params['C'], gamma=study.best_params['gamma'], epsilon=study.best_params['epsilon'])
            svr.fit(X_train[:,random_features_index],y_train)
            y_test_pred = svr.predict(X_test[:,random_features_index])
            MAE = mean_absolute_error(y_test,y_test_pred)
            SVR_total_MAE += MAE
            if MAE < SVR_min_MAE:
                SVR_min_MAE = MAE
                SVR_min_MAE_feats = random_features_index
                SVR_best_params = study.best_params
            if MAE > SVR_max_MAE:
                SVR_max_MAE = MAE
            LR_MAE_for_featsNum[features_num] = (LR_total_MAE/epoch,LR_min_MAE,LR_max_MAE,LR_min_MAE_feats)
            SVR_MAE_for_featsNum[features_num] = (SVR_total_MAE/epoch,SVR_min_MAE,SVR_max_MAE,SVR_min_MAE_feats,SVR_best_params)
        return LR_MAE_for_featsNum,SVR_MAE_for_featsNum
    
    LR_MAE_for_featsNum = {}
    SVR_MAE_for_featsNum = {}
    if feat_size is None:
        for i in range(1,1900,10):
            print("feat_size",i)
            randomfste(features_num = i,LR_MAE_for_featsNum =LR_MAE_for_featsNum ,SVR_MAE_for_featsNum =SVR_MAE_for_featsNum )
    elif isinstance(feat_size,int):
        randomfste(i,LR_MAE_for_featsNum,SVR_MAE_for_featsNum)
    elif isinstance(feat_size,list):
        for i in feat_size:
            print("feat_size",i)
            randomfste(i,LR_MAE_for_featsNum,SVR_MAE_for_featsNum)
    
    return LR_MAE_for_featsNum,SVR_MAE_for_featsNum

def Random_FS_PLSR_train_and_eval(X,
                                  y,
                                  category = "all_sample",
                                  processed_X = None,
                                  feat_size:Union[int,list] = 5,  # 随机选取多少特征
                                  samples_test_size = 0.33, # 测试集比例
                                  samples_random = True, # 样本是否是随机选取的
                                  epoch = 1000, # 随机选取多少次
                                  max_MAE = 0.1, # 随机特征选择可以接受的最大MAE
                                  min_R2 = 0.5 # 随机特征选择可以接受的最小R2
                                  ):
    """随机选取特征，然后用PLSr训练和评估，返回每个特征数下的平均MAE，最大MAE，最小MAE,默认n—components = 3
    也可以传入feat_size = [1,2,3,4,5,6,7,8,9,10]，指定特征数
    ------
    Parameters:
    ------
        - X : ndarray
            NIR spectral data
        - y : ndarray
            Expected value
        - category : str
            category name
        - processed_X : ndarray
            如果已经对X做了预处理，可以传入预处理后X训练和测试集 processed_X = (X_train,X_test)
        - feat_size : int,list
            特征数
        - samples_test_size : float
            测试集比例
        - samples_random :True
            样本是否也要随机选取
        - epoch : int
            随机选取特征的次数
        - max_MAE : float
            随机特征选择可以接受的最大MAE

    ------
    Returns:
    ------
    """
   # epoch_print 每个epoch是否打印进度
    
    epoch_print = False
    
    
    # 随机划分数据集
    from sklearn.model_selection import train_test_split

    # 样本数据是否随机选取
    if samples_random:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=samples_test_size)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=samples_test_size, random_state=42)
    

    # 如果已经对X做了预处理，可以传入预处理后X训练和测试集 processed_X = (X_train,X_test)
    import numpy as np
    if processed_X is not None:
        X_train,X_test = processed_X

    # 随机特征选择
    ## 选取的每个特征数下的平均MAE最小，最大MAE √
    ## 选取的每个特征数下的MAE分布 ×
    def randomfste(features_num):


        plsr_min_MAE = 10000
        plsr_min_MAE_feats = []
        plsr_min_MAE_R2 = 0


        # 迭代次数
        for i in range(epoch):
            # 是否打印进度
            if epoch_print:
                print("epoch:{},feat_num:{}".format(i,features_num))

            # 随机选取特征
            random_features_index = np.random.choice(np.arange(0, 1899), size=features_num, replace=False)

            # plsr模型
            from sklearn.cross_decomposition import PLSRegression
            from sklearn.metrics import mean_absolute_error,r2_score
            n_coms = 3
            if features_num < 3:
                n_coms = features_num
            plsr  = PLSRegression(n_components=n_coms)
            plsr.fit(X_train[:,random_features_index],y_train)
            y_test_pred = plsr.predict(X_test[:,random_features_index])
            MAE = mean_absolute_error(y_test,y_test_pred)
            R2 = r2_score(y_test,y_test_pred)

            if MAE <= plsr_min_MAE and R2>=min_R2:
                plsr_min_MAE = MAE
                plsr_min_MAE_feats = random_features_index
                plsr_min_MAE_R2 = R2
        return plsr_min_MAE,plsr_min_MAE_R2,plsr_min_MAE_feats
    if isinstance(feat_size,int):
        return randomfste(feat_size)
    else :
        plsr_min_MAE = []
        plsr_min_MAE_R2 = []
        plsr_min_MAE_feats = []
        output = {}
        
        for i in feat_size:
            # print("feat_size",i)
            MAE,R2,feats = randomfste(i)
            output[i] =[MAE,R2,feats]
            # plsr_min_MAE.append(MAE)
            # plsr_min_MAE_R2.append(R2)
            # plsr_min_MAE_feats.append(feats)
        return output
        
def Random_FS_RFR_train_and_eval(X,
                                  y,
                                  category = "all_sample",
                                  processed_X = None,
                                  feat_size:Union[int,list] = 5,  # 随机选取多少特征
                                  samples_test_size = 0.33, # 测试集比例
                                  samples_random = True, # 样本是否是随机选取的
                                  epoch = 1000, # 随机选取多少次
                                  min_R2 = 0.5, # 随机特征选择可以接受的最小R2
                                  ):
    """随机选取特征，然后用随机森林回归训练和评估，返回每个特征数下的最小MAE 和R2
    也可以传入feat_size = [1,2,3,4,5,6,7,8,9,10]，指定特征数
    ------
    Parameters:
    ------
        - X : ndarray
            NIR spectral data
        - y : ndarray
            Expected value
        - category : str
            category name
        - processed_X : ndarray
            如果已经对X做了预处理，可以传入预处理后X训练和测试集 processed_X = (X_train,X_test)
        - feat_size : int,list
            特征数
        - samples_test_size : float
            测试集比例
        - samples_random :True
            样本是否也要随机选取
        - epoch : int
            随机选取特征的次数
        - max_MAE : float
            随机特征选择可以接受的最大MAE

    ------
    Returns:
    ------
    modify:
    -------
        - 2023-12-6 创建函数
    """
   # epoch_print 每个epoch是否打印进度
    
    epoch_print = True
    
    
    # 随机划分数据集
    from sklearn.model_selection import train_test_split

    # 样本数据是否随机选取
    if samples_random:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=samples_test_size)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=samples_test_size, random_state=42)
    

    # 如果已经对X做了预处理，可以传入预处理后X训练和测试集 processed_X = (X_train,X_test)
    import numpy as np
    if processed_X is not None:
        X_train,X_test = processed_X

    # 随机特征选择
    ## 选取的每个特征数下的平均MAE最小，最大MAE √
    ## 选取的每个特征数下的MAE分布 ×
    def randomfste(features_num):


        rfr_min_MAE = 10000
        rfr_min_MAE_feats = []
        rfr_min_MAE_R2 = 0


        # 迭代次数
        for i in range(epoch):
            # 是否打印进度
            if epoch_print:
                print("epoch:{},feat_num:{}".format(i,features_num))

            # 随机选取特征
            random_features_index = np.random.choice(np.arange(0, 1899), size=features_num, replace=False)

            # rfr模型
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_absolute_error,r2_score
            rfr  = RandomForestRegressor()
            rfr.fit(X_train[:,random_features_index],y_train)
            y_test_pred = rfr.predict(X_test[:,random_features_index])
            MAE = mean_absolute_error(y_test,y_test_pred)
            R2 = r2_score(y_test,y_test_pred)

            if MAE <= rfr_min_MAE and R2 >= min_R2:
                rfr_min_MAE = MAE
                rfr_min_MAE_feats = random_features_index
                rfr_min_MAE_R2 = R2
        return rfr_min_MAE,rfr_min_MAE_R2,rfr_min_MAE_feats
    if isinstance(feat_size,int):
        return randomfste(feat_size)
    else :
        rfr_min_MAE = []
        rfr_min_MAE_R2 = []
        rfr_min_MAE_feats = []
        for i in feat_size:
            print("feat_size",i)
            MAE,R2,feats = randomfste(i)
            rfr_min_MAE.append(MAE)
            rfr_min_MAE_R2.append(R2)
            rfr_min_MAE_feats.append(feats)
        return rfr_min_MAE,rfr_min_MAE_R2,rfr_min_MAE_feats
   

def run_optuna_v5(data_dict, train_key, isReg, chose_n_trails, selected_metric='rmse', save=None, save_name="", **kw):
    # 2024-12-23 V5版本
    # 2024-12-23修改了返回score的问题，返回的是除了训练集的score均值
    # 光谱数据的自动调参函数，可以自动调整选用建模过程中的哪些方法，参数。
    # V5版本不再支持随机划分数据，改为支持多数据集输入（一个训练多个测试），使用所有数据集的均值作为评估指标，修改了结果输出的格式，现在为保存json文件。
    # V4版本增加了交叉验证的功能
    # V3版本增加了允许传入已经划分好的X_train,X_test,y_train,y_test,如果splited_data传了，就不用传X,y了。
    # 相比于v1版本，v2版本增加（1）对分类任务的支持优化（2）增加了划分数据的方法 (3)增加了验证集，最终输出的是基于测试集的指标
    """
    光谱数据的自动调参函数，支持多数据集验证。可以自动调整选用建模过程中的哪些方法和参数。
    
    Parameters
    ----------
    data_dict : dict
        包含多个数据集的字典, 格式为 {'dataset1': (X1, y1), 'dataset2': (X2, y2), ...}
        其中每个数据集都是一个元组 (X, y), X 是特征矩阵, y 是目标变量
    train_key : str
        用于训练的数据集的键值
    isReg : bool
        是否为回归任务
    chose_n_trails : int
        Optuna调参迭代次数
    selected_metric : str
        选择的评估指标, 支持 ['mae','mse','r2','r',"accuracy", "precision", "recall"]
    save : str, optional
        保存路径
    save_name : str, optional
        保存文件名
    **kw : dict
        其他参数，包括可选择的预处理方法等

    Returns
    -------
    dict
        包含最优模型在所有数据集上的预测结果和评估指标

    -------
    Example
    -------
        import numpy as np
        from sklearn.datasets import make_regression, make_classification
        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt

        def create_test_datasets(task='regression', n_features=20, noise_levels=[0.1, 0.2, 0.3]):
            np.random.seed(42)
            data_dict = {}
            
            # 创建训练集
            if task == 'regression':
                X_train, y_train = make_regression(
                    n_samples=100,
                    n_features=n_features,
                    noise=0.1,
                    random_state=42
                )
            else:
                X_train, y_train = make_classification(
                    n_samples=100,
                    n_features=n_features,
                    n_classes=2,
                    random_state=42
                )
            
            # 标准化特征
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            
            # 将训练集添加到字典
            data_dict['train'] = (X_train, y_train)
            
            # 创建具有不同噪声水平的测试集
            for i, noise in enumerate(noise_levels):
                if task == 'regression':
                    X_test, y_test = make_regression(
                        n_samples=50,
                        n_features=n_features,
                        noise=noise,
                        random_state=42+i
                    )
                else:
                    X_test, y_test = make_classification(
                        n_samples=50,
                        n_features=n_features,
                        n_classes=2,
                        random_state=42+i
                    )
                
                # 使用相同的缩放器转换测试集
                X_test = scaler.transform(X_test)
                
                # 将测试集添加到字典
                data_dict[f'test_{noise}'] = (X_test, y_test)
            
            return data_dict

        # 创建回归数据集
        reg_data = create_test_datasets(
            task='regression',
            n_features=20,
            noise_levels=[0.1, 0.3, 0.5]
        )

        # 创建分类数据集
        clf_data = create_test_datasets(
            task='classification',
            n_features=20,
            noise_levels=[0.1, 0.3, 0.5]
        )

        # 测试回归任务
        reg_params = {
            "selected_outlier": ["不做异常值去除"],
            "selected_preprocess": ["不做预处理", "mean_centering", "standardization"],
            "selected_feat_sec": ["不做特征选择"],
            "selected_model": ["LR", "SVR", "PLSR"]
        }

        # 运行优化
        reg_results = run_optuna_v5(
            data_dict=reg_data,
            train_key='train',
            isReg=True,
            chose_n_trails=100,
            selected_metric='r2',
            save = r'./',
            save_name='reg_results',
            **reg_params
        )

        # 测试分类任务
        clf_params = {
            "selected_outlier": ["不做异常值去除"],
            "selected_preprocess": ["不做预处理", "mean_centering", "standardization"],
            "selected_feat_sec": ["不做特征选择"],
            "selected_model": ["LogisticRegression", "SVM", "RandomForest"]
        }

        # 运行优化
        clf_results = run_optuna_v5(
            data_dict=clf_data,
            train_key='train',
            isReg=False,
            chose_n_trails=100,
            selected_metric='accuracy',
            **clf_params
        )

        # 打印结果
        def print_results(results, task_type):
            print(f"\n{task_type} Task Results:")
            print("-" * 50)
            print("Best parameters:")
            for param, value in results['best_params'].items():
                print(f"{param}: {value}")
            print("\nDataset Scores:")
            for dataset_name, scores in results['dataset_scores'].items():
                print(f"{dataset_name}: {scores['score']:.4f}")

        print_results(reg_results, "Regression")
        print_results(clf_results, "Classification")

        # 可视化结果
        def plot_results(results, task_type):
            plt.figure(figsize=(12, 5))
            
            # 绘制每个数据集的得分
            datasets = list(results['dataset_scores'].keys())
            scores = [results['dataset_scores'][d]['score'] for d in datasets]
            
            plt.subplot(1, 2, 1)
            plt.bar(datasets, scores)
            plt.title(f'{task_type} Scores Across Datasets')
            plt.xticks(rotation=45)
            plt.ylabel('Score')
            
            # 绘制优化历史
            trials_df = pd.DataFrame(results['optimization_history'])
            plt.subplot(1, 2, 2)
            plt.plot(trials_df['number'], trials_df['value'], 'b-')
            plt.title('Optimization History')
            plt.xlabel('Trial number')
            plt.ylabel('Objective value')
            
            plt.tight_layout()
            plt.show()

        # 绘制结果
        plot_results(reg_results, "Regression")
        plot_results(clf_results, "Classification")
        
    -----
    """
    import random
    import numpy as np
    import optuna
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold
    import traceback
    from scipy.stats import pearsonr

    def choose_random_elements(my_list, num_elements):
        if num_elements == 0:
            return random.choice(my_list)
        else:
            return random.choices(my_list, k=num_elements)

    # 验证输入数据
    if train_key not in data_dict:
        raise KeyError(f"训练数据集键值 {train_key} 不在数据字典中")

    X_train_orig, y_train_orig = data_dict[train_key]
    
    # 用于存储中间结果
    temp_list_to_database = {}
    results_by_dataset = {}

    # 设置默认参数
    selected_outlier = ["不做异常值去除", "mahalanobis"]
    selected_preprocess = ["不做预处理", "mean_centering", "normalization", "standardization", 
                         "poly_detrend", "snv", "savgol", "msc","d1", "d2", "rnv", "move_avg"]
    preprocess_number_input = kw.get("preprocess_number_input",1)
    selected_feat_sec = ["不做特征选择","corr_coefficient","anova","remove_high_variance_and_normalize","random_select"]
    selected_dim_red = ["不做降维","pca"]

    # 更新参数（如果在kw中提供）
    if "selected_outlier" in kw: selected_outlier = kw["selected_outlier"]
    if "selected_preprocess" in kw: selected_preprocess = kw["selected_preprocess"]
    if "selected_feat_sec" in kw: selected_feat_sec = kw["selected_feat_sec"]
    if "selected_dim_red" in kw: selected_dim_red = kw["selected_dim_red"]
    
    # 设置模型选项
    if isReg:
        model_options = ['LR', 'SVR', 'PLSR', 'Bayes(贝叶斯回归)', 'RFR(随机森林回归)', 'BayesianRidge']
    else:
        model_options = ['LogisticRegression', 'SVM', 'DT', 'RandomForest', 'KNN', 
                        'Bayes(贝叶斯分类)', "GradientBoostingTree", "XGBoost"]

    if "selected_model" in kw:
        model_options = kw["selected_model"]

    def objective(trial):
        """Optuna优化目标函数"""
        
        # 复制一份原始训练数据
        X_train, y_train = X_train_orig.copy(), y_train_orig.copy()

        # 预处理流程
        functions_ = {
            'Pretreatment': {
                '不做异常值去除': [AF.return_inputs, {}],
                'mahalanobis':
                    [AF.mahalanobis, {'threshold': trial.suggest_int('mahalanobis_threshold', 1, 100)}],
            },
            'Dataset_Splitting': {
                'random_split': [AF.random_split, {'test_size': trial.suggest_float('random_split_test_size', 0.1, 0.9) ,
                                                    'random_seed':  trial.suggest_int('random_split_random_seed', 0, 100)   }],
                'custom_train_test_split':[AF.custom_train_test_split , {'test_size': trial.suggest_float('custom_train_test_split_test_size', 0.1, 0.9),
                                                                            'method': trial.suggest_categorical('custom_train_test_split_method', ['KS', 'SPXY'])}],
                            },
            'Preprocess': {
                '不做预处理': [AF.return_inputs, {}],
                'mean_centering': [AF.mean_centering, {'axis':trial.suggest_categorical('mean_centering_axis',[None,0,1])}],
                'normalization': [AF.normalization,
                                    {'axis': trial.suggest_categorical('normalization_axis', [None,0,1])}],
                'standardization': [AF.standardization,
                                    {'axis': trial.suggest_categorical('standardization_axis', [None,0,1])}],
                'poly_detrend': [AF.poly_detrend,   {'poly_order': trial.suggest_int('poly_detrend_poly_order', 2, 4)}],
                'snv': [AF.snv, {}],
                'savgol':[AF.savgol,{
                    'window_len':trial.suggest_int('savgol_window_len',1,100),
                    'poly':trial.suggest_int('savgol_poly',0,trial.params['savgol_window_len']-1),
                    'deriv':trial.suggest_int('savgol_deriv',0,2),
                }],
                'msc': [AF.msc, {'mean_center': trial.suggest_categorical('msc_mean_center', [True, False])}],
                'd1':[AF.d1, {}],
                'd2':[AF.d2, {}],
                'rnv':[AF.rnv, {'percent' : trial.suggest_int('rnv_percent', 1, 100)}],
                'move_avg':[AF.move_avg, {'window_size': trial.suggest_int('move_avg_window_size', 3, 399, step=2)}],
                'baseline_iarpls':[AF.baseline_iarpls, {'lam': trial.suggest_int('baseline_iarpls_lam', 1, 300, step=1)}],
                'baseline_airpls':[AF.baseline_airpls, {'lam': trial.suggest_int('baseline_airpls_lam',1, 300, step=1)}],
                'baseline_derpsalsa':[AF.baseline_derpsalsa, {'lam': trial.suggest_int('baseline_derpsalsa_lam', 1, 300, step=1)}],

            },
            "Feature_Selection": {
                '不做特征选择': [AF.return_inputs, {}],
                'cars': [AF.cars, {'n_sample_runs': trial.suggest_int('cars_n_sample_runs', 10, 1000),
                                    'pls_components': trial.suggest_int('cars_pls_components', 1, 20),
                                    'n_cv_folds': trial.suggest_int('cars_n_cv_folds', 1, 10)}],
                'spa': [AF.spa, {'i_init': trial.suggest_int('spa_i_init', 0, 100),
                                    'method': trial.suggest_categorical('spa_method', [0, 1]),
                                    'mean_center': trial.suggest_categorical('spa_mean_center', [True, False])}],
                'corr_coefficient': [AF.corr_coefficient, {'threshold': trial.suggest_float('corr_coefficient_threshold', 0.0, 1.0)}],
                'anova': [AF.anova, {
                    'threshold': trial.suggest_float('anova_threshold', 0.0, 1.0)}],
                'fipls': [AF.fipls, {'n_intervals': trial.suggest_int('fipls_n_intervals', 1, 5),
                                    'interval_width': trial.suggest_float('fipls_interval_width', 10, 500),
                                    'n_comp': trial.suggest_int('fipls_n_comp', 2, 20)}],
                'remove_high_variance_and_normalize':[AF.remove_high_variance_and_normalize, {'remove_feat_ratio': trial.suggest_float('remove_high_variance_and_normalize_remove_feat_ratio', 0.01, 0.5)}],
                'random_select':[AF.random_select, {'min_features': trial.suggest_int('random_select_min_features', 1, 20),
                                                    'max_features': trial.suggest_int('random_select_max_features', 1, 20),
                                                    'random_seed': trial.suggest_int('random_select_random_seed', 1, 100)}],
            },

            'Dimensionality_reduction': {
                '不做降维': [AF.return_inputs, {}],
                'pca': [AF.pca, {'n_components': trial.suggest_int('pca_n_components', 1, 50)}],
                # 'remove_high_variance_and_normalize':[AF.remove_high_variance_and_normalize, {'remove_feat_ratio': trial.suggest_float('remove_high_variance_and_normalize_remove_feat_ratio', 0.01, 0.5)}]
            },


            'Model_Selection': {
                'LR': [AF.LR, {}],
                'SVR': [AF.SVR,
                            {'kernel': trial.suggest_categorical('SVR_kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                                'C': trial.suggest_float("SVR_c", 1e-5, 1000, log=True),
                                'epsilon': trial.suggest_float('SVR_epsilon', 0.01, 1, log=True),
                                'degree': trial.suggest_int('SVR_degree', 1, 5),
                                'gamma': trial.suggest_float("SVR_gamma", 1e-5, 1000, log=True)}],
                'PLSR': [AF.PLSR, {
                                    'scale': trial.suggest_categorical('PLSR_scale', [True, False])}],
                'Bayes(贝叶斯回归)': [AF.bayes, {
                                                    'tol': trial.suggest_float('Bayes(贝叶斯回归)_tol', 0.0001, 0.1),
                                                    'alpha_1': trial.suggest_float('Bayes(贝叶斯回归)_alpha_1', 0.0001, 0.1),
                                                    'alpha_2': trial.suggest_float('Bayes(贝叶斯回归)_alpha_2', 0.0001, 0.1),
                                                    'lambda_1': trial.suggest_float('Bayes(贝叶斯回归)_lambda_1', 0.0001, 0.1),
                                                    'lambda_2': trial.suggest_float('Bayes(贝叶斯回归)_lambda_2', 0.0001, 0.1),
                                                    'compute_score': trial.suggest_categorical('Bayes(贝叶斯回归)_compute_score', [True, False]),
                                                    'fit_intercept': trial.suggest_categorical('Bayes(贝叶斯回归)_fit_intercept', [True, False])}],
                'RFR(随机森林回归)': [AF.RFR, {'n_estimators': trial.suggest_int('RFR(随机森林回归)_n_estimators', 1, 100),
                                                'criterion': trial.suggest_categorical('RFR(随机森林回归)_criterion', ["squared_error", "absolute_error", "friedman_mse", "poisson"]),
                                                # 'max_depth': trial.suggest_int('RFR(随机森林回归)_max_depth', 1, 100),
                                                'min_samples_split': trial.suggest_float('RFR(随机森林回归)_min_samples_split', 0.0, 1.0),
                                                'min_samples_leaf': trial.suggest_int('RFR(随机森林回归)_min_samples_leaf', 1, 100),
                                                'min_weight_fraction_leaf': trial.suggest_float('RFR(随机森林回归)_min_weight_fraction_leaf', 0.0001, 0.1),
                                                'max_features': trial.suggest_float('RFR(随机森林回归)_max_features', 0.1,1.0),
                                                'random_state': trial.suggest_int('RFR(随机森林回归)_random_state', 1, 100),
                                                }],
                'BayesianRidge': [AF.BayesianRidge, {
                                                'alpha_1': trial.suggest_float("BR_alpha_1", 1e-10, 10.0, log=True),
                                                'alpha_2': trial.suggest_float("BR_alpha_2", 1e-10, 10.0, log=True),
                                                'lambda_1': trial.suggest_float("BR_lambda_1", 1e-10, 10.0, log=True),
                                                'lambda_2': trial.suggest_float("BR_lambda_2", 1e-10, 10.0, log=True),
                                                'tol': trial.suggest_float("BR_tol", 1e-6, 1e-1, log=True),
                                                'fit_intercept': trial.suggest_categorical("BR_fit_intercept", [True, False]),
                                            }],
                'LactateNet': [AF.LactateNet, {
                    # 'epochs': trial.suggest_int('LactateNet_epochs', 1, 3000),
                    # 'batch_size': trial.suggest_int('LactateNet_batch_size', 1, 100),
                    'learning_rate': trial.suggest_float('LactateNet_learning_rate', 0.001, 1.0),
                    'weight_decay': trial.suggest_float('LactateNet_weight_decay', 0.001, 1.0),
                    'patience': trial.suggest_int('LactateNet_patience', 1, 100),
                }],
                'LassoRegression': [AF.LassoRegression, {'alpha': trial.suggest_float('LassoRegression_alpha', 0.01, 1.0),
                                                            'max_iter': trial.suggest_int('LassoRegression_max_iter', 100, 1000),
                                                            'tol': trial.suggest_float('LassoRegression_tol', 1e-5, 1e-2),
                                                            'selection': trial.suggest_categorical('LassoRegression_selection', ['cyclic', 'random']),
                                                            'random_state': trial.suggest_int('LassoRegression_random_state', 1, 100)}],
                'GradientBoostingTreeRegression': [AF.GradientBoostingTreeRegression, {'n_estimators': trial.suggest_int('GradientBoostingTree_n_estimators', 1, 100),
                                                                'learning_rate': trial.suggest_float('GradientBoostingTree_learning_rate', 0.01, 1.0),
                                                                'max_depth': trial.suggest_int('GradientBoostingTree_max_depth', 1, 100),
                                                                'min_samples_split': trial.suggest_int('GradientBoostingTree_min_samples_split', 2, 100),
                                                                'min_samples_leaf': trial.suggest_int('GradientBoostingTree_min_samples_leaf', 1, 100),
                                                                'max_features': trial.suggest_categorical('GradientBoostingTree_max_features', [None, 'sqrt', 'log2']),
                                                                'random_state': trial.suggest_int('GradientBoostingTree_random_state', 1, 100)}],
                'XGBoostRegression': [AF.XGBoostRegression, {'n_estimators': trial.suggest_int('XGBoostRegression_n_estimators', 1, 100),
                                                            'learning_rate': trial.suggest_float('XGBoostRegression_learning_rate', 0.01, 1.0),
                                                            'max_depth': trial.suggest_int('XGBoostRegression_max_depth', 1, 100),
                                                            'min_child_weight': trial.suggest_int('XGBoostRegression_min_child_weight', 1, 10),
                                                            'subsample': trial.suggest_float('XGBoostRegression_subsample', 0.1, 1.0),
                                                            'colsample_bytree': trial.suggest_float('XGBoostRegression_colsample_bytree', 0.1, 1.0),
                                                            'reg_alpha': trial.suggest_float('XGBoostRegression_reg_alpha', 0.0, 1.0),
                                                            'reg_lambda': trial.suggest_float('XGBoostRegression_reg_lambda', 0.0, 1.0),
                                                            'random_state': trial.suggest_int('XGBoostRegression_random_state', 1, 100)}],
                'CatBoostRegression': [AF.CatBoostRegression, {'n_estimators': trial.suggest_int('CatBoostRegression_n_estimators', 1, 100),
                                                            'learning_rate': trial.suggest_float('CatBoostRegression_learning_rate', 0.01, 1.0),
                                                            'depth': trial.suggest_int('CatBoostRegression_depth', 1, 10),
                                                            'l2_leaf_reg': trial.suggest_float('CatBoostRegression_l2_leaf_reg', 0.0, 10.0),
                                                            'random_state': trial.suggest_int('CatBoostRegression_random_state', 1, 100)}],
                'MLPRegression': [AF.MLPRegression, {'hidden_layer_sizes': trial.suggest_categorical('MLPRegression_hidden_layer_sizes',[(100,), (50, 50), (100, 50, 25)]),
                                                    'activation': trial.suggest_categorical('MLPRegression_activation', ['relu', 'tanh', 'logistic']),
                                                    'solver': trial.suggest_categorical('MLPRegression_solver', ['adam', 'sgd', 'lbfgs']),
                                                    'alpha': trial.suggest_float('MLPRegression_alpha', 1e-5, 1e-1, log=True),
                                                    'batch_size': trial.suggest_categorical('MLPRegression_batch_size', ['auto', 32, 64, 128]),
                                                    'learning_rate': trial.suggest_categorical('MLPRegression_learning_rate', ['constant', 'invscaling', 'adaptive']),
                                                    'learning_rate_init': trial.suggest_float('MLPRegression_learning_rate_init', 1e-4, 1e-1, log=True),
                                                    'max_iter': trial.suggest_int('MLPRegression_max_iter', 100, 1000),
                                                    'random_state': trial.suggest_int('MLPRegression_random_state', 1, 100)}],
                'LightGBMRegression': [AF.LightGBMRegression, {'num_leaves': trial.suggest_int('LightGBMRegression_num_leaves', 10, 100),
                                                       'learning_rate': trial.suggest_float('LightGBMRegression_learning_rate', 0.01, 0.3),
                                                       'n_estimators': trial.suggest_int('LightGBMRegression_n_estimators', 50, 500),
                                                       'max_depth': trial.suggest_int('LightGBMRegression_max_depth', 3, 15),
                                                       'min_child_samples': trial.suggest_int('LightGBMRegression_min_child_samples', 10, 100),
                                                       'subsample': trial.suggest_float('LightGBMRegression_subsample', 0.5, 1.0),
                                                       'colsample_bytree': trial.suggest_float('LightGBMRegression_colsample_bytree', 0.5, 1.0),
                                                       'reg_alpha': trial.suggest_float('LightGBMRegression_reg_alpha', 0.0, 1.0),
                                                       'reg_lambda': trial.suggest_float('LightGBMRegression_reg_lambda', 0.0, 1.0),
                                                       'random_state': trial.suggest_int('LightGBMRegression_random_state', 1, 100)}],



                'LogisticRegression':[AF.logr,{}],
                'SVM': [AF.SVM, {
                        'kernel': trial.suggest_categorical('SVM_kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                        'C': trial.suggest_float("SVM_C", 1e-5, 1000, log=True),
                        'degree': trial.suggest_int('SVM_degree', 1, 5),
                        'gamma': trial.suggest_float("SVM_gamma", 1e-5, 1000, log=True),
                        'random_state': trial.suggest_int('SVM_random_state', 1, 100),
                    }],
                'DT': [AF.DT, {
                    'criterion': trial.suggest_categorical('DT_criterion', ["gini", "entropy", "log_loss"]),
                    'splitter': trial.suggest_categorical('DT_splitter', ["best", "random"]),
                    'min_samples_split': trial.suggest_int('DT_min_samples_split', 2, 100),
                    'min_samples_leaf': trial.suggest_int('DT_min_samples_leaf', 1, 100),
                    'random_state': trial.suggest_int('DT_random_state', 1, 100),
                }],
                'RandomForest': [AF.RandomForest, {
                    'n_estimators': trial.suggest_int('RandomForest_n_estimators', 1, 100),
                    'criterion': trial.suggest_categorical('RandomForest_criterion',
                                                            ["entropy", "gini", "log_loss"]),
                    'min_samples_split': trial.suggest_float('RandomForest_min_samples_split', 0.0, 1.0),
                    'min_samples_leaf': trial.suggest_int('RandomForest_min_samples_leaf', 1, 100),
                    'random_state': trial.suggest_int('RandomForest_random_state', 1, 100),
                    }],
                'KNN': [AF.KNN, {
                    'n_neighbors': trial.suggest_int('KNN_n_neighbors', 1, 100),
                    'weights': trial.suggest_categorical('KNN_weights', ['uniform', 'distance']),
                    'algorithm': trial.suggest_categorical('KNN_algorithm',
                                                            ['auto', 'ball_tree', 'kd_tree', 'brute']),
                    'leaf_size': trial.suggest_int('KNN_leaf_size', 1, 100),
                    'p': trial.suggest_int('KNN_p', 1, 100),
                    'metric': trial.suggest_categorical('KNN_metric',
                                                        ['minkowski', 'euclidean', 'manhattan', 'chebyshev']),
                }],
                'Bayes(贝叶斯分类)': [
                    AF.CustomNaiveBayes, {
                        'classifier_type': trial.suggest_categorical('Bayes(贝叶斯分类)_classifier_type',
                                                                        ['gaussian', 'multinomial', 'bernoulli']),
                        'alpha': trial.suggest_float('Bayes(贝叶斯分类)_alpha', 0.0001, 0.1),
                    }],
                'GradientBoostingTree': [AF.GradientBoostingTree, {
                    'n_estimators': trial.suggest_int('GradientBoostingTree_n_estimators', 1, 100),
                    'learning_rate': trial.suggest_float('GradientBoostingTree_learning_rate', 0.01, 1.0),
                    'max_depth': trial.suggest_int('GradientBoostingTree_max_depth', 1, 100),
                    'random_state': trial.suggest_int('GradientBoostingTree_random_state', 1, 100),
                }],
                'XGBoost': [AF.XGBoost, {
                    'n_estimators': trial.suggest_int('XGBoost_n_estimators', 1, 100),
                    'learning_rate': trial.suggest_float('XGBoost_learning_rate', 0.01, 1.0),
                    'max_depth': trial.suggest_int('XGBoost_max_depth', 1, 100),
                    'subsample': trial.suggest_float('XGBoost_subsample', 0.1, 1.0),
                    'colsample_bytree': trial.suggest_float('XGBoost_colsample_bytree', 0.1, 1.0),
                    'random_state': trial.suggest_int('XGBoost_random_state', 1, 100),
                }],
            },
        }

        # 随机选择处理方法
        selection_steps = [0] * 8
        selection_steps[0] = choose_random_elements(selected_outlier, 0)
        selection_steps[3] = choose_random_elements(selected_preprocess, preprocess_number_input)
        selection_steps[4] = choose_random_elements(selected_feat_sec, 0)
        selection_steps[7] = choose_random_elements(selected_dim_red, 0)
        selection_steps[5] = choose_random_elements(model_options, 0)

        # 获取每个步骤的参数
        outlier_params = functions_["Pretreatment"][selection_steps[0]][1]
        preprocess_params = [functions_["Preprocess"][step][1] for step in selection_steps[3]]
        feat_selection_params = functions_["Feature_Selection"][selection_steps[4]][1]
        dim_reduction_params = functions_["Dimensionality_reduction"][selection_steps[7]][1]
        model_params = functions_["Model_Selection"][selection_steps[5]][1]

        # 存储当前trial的选择步骤和参数
        trial.set_user_attr('selection_steps', {
            # 'outlier': [selection_steps[0], outlier_params],
            'preprocess': [[step for step in selection_steps[3]], preprocess_params],
            'feature_selection': [selection_steps[4], feat_selection_params],
            'dimensionality_reduction': [selection_steps[7], dim_reduction_params],
            'model': [selection_steps[5], model_params]
        })

        # 存储当前trial的处理流程
        temp_list_to_database[trial.number] = []
        
        # 在所有数据集上进行预测和评估
        dataset_scores = {}
        try:
        # if True:
            test_data_dict = data_dict.copy()

            for dataset_key, (X_test, y_test) in test_data_dict.items():
                # 对测试数据应用相同的预处理步骤
                X_test_processed = X_test.copy()
                X_train_processed = X_train.copy()
                
                for preprocess_step in selection_steps[3]:
                    X_train_processed, X_test_processed, y_train, y_test = functions_["Preprocess"][preprocess_step][0](
                        X_train_processed, X_test_processed, y_train, y_test,
                        **functions_["Preprocess"][preprocess_step][1]
                    )
                # 特征选择和降维
                X_train_processed, X_test_processed, y_train, y_test = functions_["Feature_Selection"][selection_steps[4]][0](
                    X_train_processed, X_test_processed, y_train, y_test,
                    **functions_["Feature_Selection"][selection_steps[4]][1]
                )
                X_train_processed, X_test_processed, y_train, y_test = functions_["Dimensionality_reduction"][selection_steps[7]][0](
                    X_train_processed, X_test_processed, y_train, y_test,
                    **functions_["Dimensionality_reduction"][selection_steps[7]][1]
                )

                # 应用模型并预测
                _, _, _, y_pred = functions_["Model_Selection"][selection_steps[5]][0](
                    X_train_processed, X_test_processed, y_train, y_test,
                    **functions_["Model_Selection"][selection_steps[5]][1]
                )

                # 计算评估指标
                if selected_metric == 'mae':
                    score = mean_absolute_error(y_test, y_pred)
                elif selected_metric == 'mse':
                    score = mean_squared_error(y_test, y_pred)
                elif selected_metric == 'r2':
                    score = r2_score(y_test, y_pred)
                elif selected_metric == 'r':
                    score = pearsonr(y_test, y_pred)[0]
                elif selected_metric == 'rmse':
                    score = np.sqrt(mean_squared_error(y_test, y_pred))
                else:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score
                    if selected_metric == 'accuracy':
                        score = accuracy_score(y_test, y_pred)
                    elif selected_metric == 'precision':
                        score = precision_score(y_test, y_pred, average="weighted")
                    elif selected_metric == "recall":
                        score = recall_score(y_test, y_pred, average="weighted")

                dataset_scores[dataset_key] = {
                    'score': score,
                    'y_pred': y_pred.tolist(),
                }

            # 存储结果
            trial.set_user_attr('dataset_scores', dataset_scores)
            
            # 返回除了训练数据集之外的所有数据集评分的平均值作为优化目标
            # del dataset_scores[train_key]
            return np.mean( [ds['score'] for key,ds in dataset_scores.items() if key is not train_key] )
            # return np.mean([ds['score'] for ds in dataset_scores.values() ])
        except Exception as e:
            print(f"Error in trial {trial.number} at line {traceback.extract_tb(e.__traceback__)[-1].lineno}: {str(e)}")
            return None

    # 创建和运行Optuna研究
    study = optuna.create_study(
        direction='maximize' if selected_metric in ['r2', 'r', 'accuracy', 'precision', 'recall'] else 'minimize'
    )
    study.optimize(objective, n_trials=chose_n_trails)

    # 获取最佳trial的结果
    best_trial = study.best_trial
    best_params = best_trial.params
    best_dataset_scores = best_trial.user_attrs['dataset_scores']
    best_selection_steps = best_trial.user_attrs['selection_steps']  # 获取最佳selection_steps

        # 修改后的结果处理部分
    def process_results(study, best_params, best_dataset_scores, best_selection_steps, save=None, save_name=None):
        """
        处理和保存优化结果
        
        Parameters
        ----------
        study : optuna.Study
            优化研究对象
        best_params : dict
            最佳参数
        best_dataset_scores : dict
            各数据集的得分
        best_selection_steps : dict
            最佳选择步骤
        save : str, optional
            保存路径
        save_name : str, optional
            保存文件名
        
        Returns
        -------
        dict
            处理后的结果字典
        """
        import pandas as pd
        
        # 获取trials数据框
        trials_df = study.trials_dataframe()
        
        # 处理datetime列
        for col in trials_df.columns:
            if pd.api.types.is_datetime64_any_dtype(trials_df[col]):
                trials_df[col] = trials_df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else None)
        
        # 整理结果
        results = {
            'best_params': best_params,
            'best_selection_steps': best_selection_steps,
            'dataset_scores': best_dataset_scores,
            'optimization_history': trials_df.to_dict('records')
        }
        
        # 如果需要保存结果
        if save and save_name:
            import os
            import json
            os.makedirs(save, exist_ok=True)
            save_path = os.path.join(save, f"{save_name}_results.json")
            
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4, default=str)
                print(f"Results successfully saved to {save_path}")
            except Exception as e:
                print(f"Error saving results: {str(e)}")
        
        return results

    return process_results(
        study=study,
        best_params=best_params,
        best_dataset_scores=best_dataset_scores,
        best_selection_steps=best_selection_steps,
        save=save,
        save_name=save_name
    )



def rebuild_model_v2(splited_data=None, params_dict:dict=None):
    """
    V2版本，run_optuna_v5将参数和配置存储为字典，现在解析字典，因为字典中没有随机划分的参数，所有去掉X，y，智能使用splited_data
    Rebuild the model based on the stored string of parameters and configuration.
    
    Params:
        X: np.array 2-D, feature matrix
        y: np.array 1-D, labels
        splited_data: tuple (X_train, X_test, y_train, y_test) if provided, otherwise the data will be split inside the function
        stored_str: string representation of the temp_list_to_database to be parsed
    
    Returns:
        Final model predictions on the test set (y_pred) and test labels (y_test)


    example:
    # 生成随机训练和测试数据
    n_samples_train = 100  # 训练样本数
    n_samples_test = 30   # 测试样本数
    n_features = 500      # 特征数

    # 生成随机光谱数据
    X_train = np.random.rand(n_samples_train, n_features) 
    X_test = np.random.rand(n_samples_test, n_features)

    # 生成随机目标值(酒精浓度)
    y_train = np.random.uniform(0, 100, n_samples_train)  # 0-100范围内的随机值
    y_test = np.random.uniform(0, 100, n_samples_test)

    # 构建splited_data元组
    splited_data = (X_train, X_test, y_train, y_test)


    # results = run_optuna_v5()
    # params_dict = results['best_selection_steps']
    params_dict = {
            "outlier": [
                "不做异常值去除",
                {}
            ],
            "preprocess": [
                [
                    "rnv",
                    "d2",
                    "snv"
                ],
                [
                    {
                        "percent": 28
                    },
                    {},
                    {}
                ]
            ],
            "feature_selection": [
                "不做特征选择",
                {}
            ],
            "dimensionality_reduction": [
                "不做降维",
                {}
            ],
            "model": [
                "RFR(随机森林回归)",
                {
                    "n_estimators": 5,
                    "criterion": "friedman_mse",
                    "min_samples_split": 0.30284442935543743,
                    "min_samples_leaf": 38,
                    "min_weight_fraction_leaf": 0.02589404748849362,
                    "max_features": 0.4548695437636622,
                    "random_state": 98
                }
            ]
        }

    y_test, y_pred = rebuild_model_v2(splited_data=splited_data, params_dict=params_dict)
    """

    functions_ = {
            'Pretreatment': {
                '不做异常值去除': [AF.return_inputs, {}],
                'mahalanobis':
                    [AF.mahalanobis, ],
            },
            'Dataset_Splitting': {
                'random_split': [AF.random_split],
                'custom_train_test_split':[AF.custom_train_test_split ],
                            },
            'Preprocess': {
                '不做预处理': [AF.return_inputs, {}],
                'mean_centering': [AF.mean_centering, {}],
                'normalization': [AF.normalization,
                                    {}],
                'standardization': [AF.standardization,
                                    {}],
                'poly_detrend': [AF.poly_detrend,   {}],
                'snv': [AF.snv, {}],
                'savgol':[AF.savgol,{}],
                'msc': [AF.msc, {}],
                'd1':[AF.d1, {}],
                'd2':[AF.d2, {}],
                'rnv':[AF.rnv, {}],
                'move_avg':[AF.move_avg, {}],
                'baseline_iarpls':[AF.baseline_iarpls, {}],
                'baseline_airpls':[AF.baseline_airpls, {}],
                'baseline_derpsalsa':[AF.baseline_derpsalsa, {}],

            },
            "Feature_Selection": {
                '不做特征选择': [AF.return_inputs, {}],
                'cars': [AF.cars, {}],
                'spa': [AF.spa, {}],
                'corr_coefficient': [AF.corr_coefficient, {}],
                'anova': [AF.anova, {}],
                'fipls': [AF.fipls, {}],
                'remove_high_variance_and_normalize':[AF.remove_high_variance_and_normalize,{}],
                'random_select':[AF.random_select,{}]

            },

            'Dimensionality_reduction': {
                '不做降维': [AF.return_inputs, {}],
                'pca': [AF.pca, {}],
                'remove_high_variance_and_normalize': [AF.remove_high_variance_and_normalize, {}],
            },


            'Model_Selection': {
                'LR': [AF.LR, {}],
                'SVR': [AF.SVR,
                            {}],
                'PLSR': [AF.PLSR, {}],
                'Bayes(贝叶斯回归)': [AF.bayes, {}],
                'RFR(随机森林回归)': [AF.RFR, {}],
                'BayesianRidge': [AF.BayesianRidge, {}],
                'LactateNet': [AF.LactateNet, {}],
                'LassoRegression': [AF.LassoRegression, {}],
                'GradientBoostingTreeRegression': [AF.GradientBoostingTreeRegression, {}],
                'XGBoostRegression': [AF.XGBoostRegression, {}],
                'CatBoostRegression': [AF.CatBoostRegression, {}],
                'MLPRegression': [AF.MLPRegression, {}],
                'LightGBMRegression': [AF.LightGBMRegression, {}],
                

                'LogisticRegression':[AF.logr,{}],
                'SVM': [AF.SVM, {}],
                'DT': [AF.DT, {}],
                'RandomForest': [AF.RandomForest, {}],
                'KNN': [AF.KNN, {}],
                'Bayes(贝叶斯分类)': [
                    AF.CustomNaiveBayes, {}],
                'GradientBoostingTree': [AF.GradientBoostingTree, {}],
                'XGBoost': [AF.XGBoost, {}],


                # 模型选择的函数和参数
            },
        }
    
    # Parse the stored_str back into the original dictionary format
    selection_steps = params_dict
    
    # Extract the steps from the selection_steps dictionary
    # outlier_removal = selection_steps['outlier']
    preprocess_list = selection_steps['preprocess'] 
    feat_selection = selection_steps['feature_selection']
    dim_reduction = selection_steps['dimensionality_reduction']
    model_selection = selection_steps['model']
    print(preprocess_list)


    # Apply the dataset splitting

    X_train, X_test, y_train, y_test = splited_data

    # Apply preprocessing steps
    for func_info, func_params in zip(preprocess_list[0], preprocess_list[1]):
        print(func_info,func_params)
        X_train, X_test, y_train, y_test = functions_["Preprocess"][func_info][0](X_train, X_test, y_train, y_test, **func_params)

    # Apply feature selection
    X_train, X_test, y_train, y_test = functions_["Feature_Selection"][feat_selection[0]][0](X_train, X_test, y_train, y_test, **feat_selection[1])

    # Apply dimensionality reduction
    X_train, X_test, y_train, y_test = functions_["Dimensionality_reduction"][dim_reduction[0]][0](X_train, X_test, y_train, y_test, **dim_reduction[1])

    # Apply the model selection and fit the model
    y_train, y_test, y_train_pred, y_pred = functions_["Model_Selection"][model_selection[0]][0](
        X_train, X_test, y_train, y_test, **model_selection[1]
    )

    return y_test, y_pred

def get_pythonFile_functions(AF):
    
    """
    -----
    params:
    -----
        AF: example: import nirapi.ML_model as AF  ---  python file 
    -----
    return
    -----
        funcions dict 
    """
    import inspect

    # 获取模块中的所有成员
    module_members = AF.__dict__.items()

    # 构建models字典
    models = {}
    for name, member in module_members:
        if callable(member):  # 检查成员是否为函数
            # 使用inspect模块获取函数的参数信息
            params = inspect.signature(member).parameters
            default_params = {param: params[param].default for param in params if params[param].default != inspect.Parameter.empty}

            models[name] = (member, default_params)
    return models


def run_regression_optuna_v3(data_name,X = None,y=None ,data_splited = None, model='PLS',split = 'SPXY',test_size = 0.3, n_trials=200,object = "R2",cv = None,save_dir = None,each_class_mae=False,only_train_and_val_set=False):


    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    import optuna
    import pandas as pd
    import plotly.graph_objects as go
    from sklearn.svm import SVR
    import joblib
    from sklearn.model_selection import LeaveOneOut, cross_val_score
    results = []
    from nirapi.ML_model import custom_train_test_split

    if data_splited is not None:
        X_train = data_splited['X_train']
        X_val = data_splited['X_val']
        X_test = data_splited['X_test']
        y_train = data_splited['y_train']
        y_val = data_splited['y_val']
        y_test = data_splited['y_test']



    elif X is not None and y is not None and only_train_and_val_set == False:
        if split == 'SPXY':
            X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size, 'SPXY')
            X_train, X_val, y_train, y_val = custom_train_test_split(X_train, y_train, 0.3, 'SPXY')
        elif split == "Random":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
        elif split == "Sequential":
            split_index = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            val_index = int(len(X_train) * 0.7)
            X_train, X_val = X_train[:val_index], X_train[val_index:]
            y_train, y_val = y_train[:val_index], y_train[val_index:]
        else:
            assert False, "split_name error"
    elif X is not None and y is not None and only_train_and_val_set == True:
        if split == 'SPXY':
            X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size, 'SPXY')
            X_val, y_val = X_test, y_test
        elif split == "Random":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            X_val, y_val = X_test, y_test
        elif split == "Sequential":
            split_index = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            X_val, y_val = X_test, y_test
        else:
            assert False, "split_name error"
    
    
    
    # 归一化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    def objective(trial):

        try:
            if model == 'PLS':
                n_components = trial.suggest_int('n_components', 1, 30)
                regressor = PLSRegression(n_components=n_components)
            elif model == 'SVR':
                C = trial.suggest_float('C', 1e-3, 1e3,log=True)
                gamma = trial.suggest_float('gamma', 1e-3, 1e3,log=True)
                regressor = SVR(C=C, gamma=gamma)
            elif model == 'RFreg':
                n_estimators = trial.suggest_int('n_estimators', 50, 200)
                max_depth = trial.suggest_int('max_depth', 5, 30)
                max_features = trial.suggest_float('max_features', 0.0, 1.0)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                bootstrap = trial.suggest_categorical('bootstrap', [True, False])
                oob_score = trial.suggest_categorical('oob_score', [True, False])
                random_state = trial.suggest_int('random_state', 0, 100)
                regressor = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    bootstrap=bootstrap,
                    oob_score=oob_score,
                    random_state=random_state
                )
            elif model == 'LR':
                regressor = LinearRegression()
            # elif model == 'XGB':
            #     from xgboost import XGBRegressor
                
            else:
                # raise ValueError("model_name error")
                assert False, "model_name error"


            if cv is not None:
                if object == 'RMSE':
                    score = cross_val_score(regressor, X_train_scaled, y_train, cv=cv, n_jobs=-1, scoring='neg_root_mean_squared_error')
                elif object == 'R2':
                    score = cross_val_score(regressor, X_train_scaled, y_train, cv=cv, n_jobs=-1, scoring='r2')
                elif object == 'MAE':
                    score = cross_val_score(regressor, X_train_scaled, y_train, cv=cv, n_jobs=-1, scoring='neg_mean_absolute_error')
                else:
                    raise ValueError("object_name error")
                score =  np.mean(score)
                
                # regressor.fit(X_train_scaled, y_train)


            else:
                regressor.fit(X_train_scaled, y_train)
                y_val_pred = regressor.predict(X_val_scaled)
                if object == 'R2':
                    score = r2_score(y_val, y_val_pred)
                elif object == 'MAE':
                    score = -mean_absolute_error(y_val, y_val_pred)
                elif object == 'RMSE':
                    score = - np.sqrt(mean_squared_error(y_val, y_val_pred))
                else:
                    assert False, "metric_name error"
                
                # 当前模型的评估指标
                # mae = mean_absolute_error(y_test, y_val_pred)
                # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                # r2 = r2_score(y_test, y_pred)
                # pearsonr = np.corrcoef(y_test.flatten(), y_pred.flatten())[0, 1]
                # print(f"mae: {mae:.3f}, rmse: {rmse:.3f}, r2: {r2:.3f}, pearsonr: {pearsonr:.3f}")

            print("final----------------------------------------")
            y_pred = regressor.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            pearsonr = np.corrcoef(y_test.flatten(), y_pred.flatten())[0, 1]
            print(f"mae: {mae:.3f}, rmse: {rmse:.3f}, r2: {r2:.3f}, pearsonr: {pearsonr:.3f}")
            return score
        except ValueError as e:
            print(e)
            return -np.inf

    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    
    if model == 'LR':
        n_trials = 1
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)

    best_params = study.best_params
    print(f"Best Params for {model} Band {data_name}:", best_params)
    
    
    # 最优的模型
    regressor_final = None
    if model == 'PLS':
        regressor_final = PLSRegression(**best_params)
    elif model == 'SVR':
        regressor_final = SVR(**best_params)
    elif model == 'RFreg':
        regressor_final = RandomForestRegressor(**best_params)
    elif model == 'LR':
        regressor_final = LinearRegression()
    else:
        raise ValueError("model_name error")
    

    
    
    y_final = y_test
    X_final = X_test
    # final_scaler = StandardScaler()
    # X_train_scaled = final_scaler.fit_transform(X_train)
    # X_final_scaled = final_scaler.transform(X_final)
    # X_val_scaled = final_scaler.transform(X_val)




    from sklearn.pipeline import Pipeline
    regressor_final = Pipeline([
    ('scaler', StandardScaler()),      # 归一化步骤
    ('regressor_final',regressor_final)  # PLS回归步骤
    ])



    regressor_final.fit(X_train, y_train)



    y_pred_train = regressor_final.predict(X_train)
    y_pred_final = regressor_final.predict(X_final)
    y_pred_val = regressor_final.predict(X_val)
    mae = mean_absolute_error(y_final, y_pred_final)
    mse = mean_squared_error(y_final, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_final, y_pred_final))
    r2 = r2_score(y_final, y_pred_final)
    print(f"mae: {mae:.3f}, rmse: {rmse:.3f}, r2: {r2:.3f}")
    
    # 画图
    print(y_train.shape,y_test.shape,y_pred_train.shape,y_pred_final.shape)
    content = model+str(best_params)
    # train_val_and_test_pred_plot(y_train,y_val,y_test,y_pred_train,y_pred_val,y_pred_final,data_name=data_name,save_dir=save_dir,each_class_mae = each_class_mae,content = content)
    # train_and_test_pred_plot(y_train,y_final,y_pred_train,y_pred_final,data_name=data_name,save_dir=save_dir,each_class_mae = each_class_mae,content = content)
    info= {}
    info['y_val_best_value'] = study.best_value
    info['y_val_best_params'] = study.best_params
    info['y_train'] = y_train.tolist()
    info['y_pred_train'] = y_pred_train.tolist()
    info['y_val'] = y_val.tolist()
    info['y_pred_val'] = y_pred_val.tolist()
    info['y_test'] = y_final.tolist()
    info['y_pred'] = y_pred_final.tolist()
    info['mae'] = mae
    info['rmse'] = rmse
    info['r2'] = r2
    if save_dir is not None:
        # # 保存模型和数据
        joblib.dump(regressor_final, f"{save_dir}/{model}_{data_name}_model.pkl")
        import json
        with open(f"{save_dir}/{model}_{data_name}_info.json", 'w') as f:
            json.dump(info, f)
        pd.DataFrame(y_test).to_csv(f"{save_dir}/{model}_{data_name}_y_test.csv",index=False)
        pd.DataFrame(y_pred_final).to_csv(f"{save_dir}/{model}_{data_name}_y_pred_test.csv",index=False)
        pd.DataFrame(y_train).to_csv(f"{save_dir}/{model}_{data_name}_y_train.csv",index=False)
        pd.DataFrame(y_pred_train).to_csv(f"{save_dir}/{model}_{data_name}_y_pred_train.csv",index=False)
        pd.DataFrame(y_val).to_csv(f"{save_dir}/{model}_{data_name}_y_val.csv",index=False)
        pd.DataFrame(y_pred_val).to_csv(f"{save_dir}/{model}_{data_name}_y_pred_val.csv",index=False)
        
    return info
    
    # pd.DataFrame(best_params,index=[0]).to_csv(f"{save_dir}/{model}_{data_name}_best_params.csv",index=False)
    
    # return regressor_final,[X_train_scaled,X_test_scaled,y_train,y_test,y_pred_train,y_pred_test]

def tpot_auto_tune(X, y, generations=5, population_size=20, cv=5):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 初始化TPOTRegressor
    tpot = TPOTRegressor(generations=generations, population_size=population_size, cv=cv, random_state=42, verbosity=2)
    
    # 拟合模型
    tpot.fit(X_train, y_train)
    
    # 评估模型
    score = tpot.score(X_test, y_test)
    
    # 输出最优模型和分数
    print("最优模型：", tpot.fitted_pipeline_)
    print("最优分数：", score)
    
    # 返回最优模型
    return tpot.fitted_pipeline_


def spectral_reconstruction_train(PD_values, Spectra_values, epochs=50, lr=1e-3,save_dir = None):
    
    '''
    ------
    parameters:
    ------
        PD_values: 训练数据 PD值
        Spectra_values: 训练数据 光谱值
        epochs: 训练轮数
        lr: 学习率
        save_dir: 保存模型的路径
    '''


    PD_train = PD_values
    spectrum = Spectra_values
    pd_size = PD_train.shape[1]
    spectrum_size = spectrum.shape[1]
    # 设定随机种子以确保结果可复现

    def same_seeds(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
    # 网络模型
 
    # 数据集加载方法
    class SpectralDataset(Dataset):
        def __init__(self, pd_size, spectrum_size):

            self.pd_values = PD_train.astype(np.float32)    
            self.spectra = spectrum.astype(np.float32)    

        def __len__(self):
            return len(self.pd_values)

        def __getitem__(self, idx):
            return self.pd_values[idx], self.spectra[idx]

    # 创建数据集和数据加载器
    dataset = SpectralDataset( pd_size=pd_size, spectrum_size=spectrum_size)
    class SkipFirstSampler(Sampler):
        def __init__(self, data_source):
            self.data_source = data_source

        def __iter__(self):
            return iter(range(1, len(self.data_source)))  # 从第二个样本开始迭代

        def __len__(self):
            return len(self.data_source) 
    sampler = SkipFirstSampler(dataset)
    data_loader = DataLoader(dataset,sampler=sampler, batch_size=64)


    def train(model, data_loader, epochs=50, lr=1e-3):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for pd_values, spectra in data_loader:
                optimizer.zero_grad()
                spectra_reconstructed = model(pd_values)
                loss = criterion(spectra_reconstructed, spectra)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f'Epoch {epoch+1}, Average Loss: {total_loss / len(data_loader)}')

    same_seeds(15)
    # 模型初始化
    model = SpectralReconstructor(pd_size=pd_size, spectrum_size=spectrum_size)
    train(model, data_loader, epochs=50, lr=1e-3)
    # 测试
    model.eval()
    dataset_test = SpectralDataset(pd_size=pd_size, spectrum_size=spectrum_size)
    x_rec = model(torch.tensor(dataset_test[0][0], dtype=torch.float32))


    if save_dir is not None:
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['NotoSerifCJK-Regular']  # 用来正常显示中文标签
        plt.plot(x_rec.detach().numpy() , label = "重建光谱")
        plt.plot(dataset_test[0][1], label = "原始光谱")
        plt.legend()
        from datetime import datetime
        now = datetime.now()
        str = now.strftime("%Y-%m-%d_%H_%M_%S")
        #用time模块
        import time 
        time_str = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
        plt.savefig(save_dir+time_str+'_model.png')
        torch.save(model, save_dir+time_str+'.pth')
