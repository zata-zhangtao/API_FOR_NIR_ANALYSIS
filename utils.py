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
from nirapi.draw import *
from typing import Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from nirapi.ML_model import *
from sklearn.preprocessing import MinMaxScaler
import optuna
from scipy.stats import pearsonr
import matplotlib
import optuna
import nirapi.ML_model as AF
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import random
import warnings
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D



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
            MAE = mean_absolute_error(y_test,y_test_pred)
            R2 = r2_score(y_test,y_test_pred)
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.plot([min(min(y_test),min(y_train)),max(max(y_train),max(y_test))],[min(min(y_test),min(y_train)),max(max(y_train),max(y_test))],'r-')
            plt.text(min(y_test),min(y_train),"MAE:{:.5},R2:{:.5}".format(MAE,R2))
            plt.legend()
            plt.title(category+" train_test_Scatter")


        plt.subplot(1,2,1)
        train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,category= category+" LR")
    else:
        from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
        LR_MAE = mean_absolute_error(y_test,y_test_pred)
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
        SVR_MAE = mean_absolute_error(y_test,y_test_pred)
        SVR_R2 = r2_score(y_test,y_test_pred)

        return (round(LR_MAE,3),round(LR_R2,3)),(round(SVR_MAE,3),round(SVR_R2,3))

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
        
def Auto_tuning_with_svr(X=None,y=None,name="auto_tuning",epoch=100,n_trial=200,n_jobs=2,test_set = False,**kw):
    """
    输入数据和名字，输出一个以名字命名的文件
    -----
    params:
    -----
        X: X input data (sample_nums,feature_nums)
        y: y input data (sample_nums,)   
        name: task's name
        epoch: the num of while
        n_trial: the num of optuna
    """
    # 循环读取文件并且自动调参
    # y = y.ravel()

    print("太久没有使用，这个文件即将删除")
    while_time = 0
    while while_time < epoch:
        while_time += 1
        try:
            ## 调参
            def SVR(x_train, x_test, y_train, y_test,kernel = 'rbf',C = 1.0,epsilon = 0.1,degree = 3,gamma = 'scale'):
                from sklearn.svm import SVR
                svr = SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree, gamma=gamma, max_iter=100000)
                svr.fit(x_train, y_train)
                y_train_pred = svr.predict(x_train)
                y_test_pred = svr.predict(x_test)
                return y_train, y_test, y_train_pred, y_test_pred
            
            # X_train,X_test,y_train,y_test = custom_train_test_split(X,y,test_size=0.33,method='SPXY')
            if test_set == False:
                X_train,X_test,y_train,y_test = random_split(X,y,test_size=0.33)
            else:
                X_train,X_test,y_train,y_test = kw.get('splited_data')
                

            def objective(trial):
        
                svr_dict = {'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                                    'C': trial.suggest_float("C", 1e-5, 1000, log=True),
                                    'epsilon': trial.suggest_float('epsilon', 0.01, 1, log=True),
                                    'degree': trial.suggest_int('degree', 1, 5),
                                    'gamma': trial.suggest_float("gamma", 1e-5, 1000, log=True)}


                # 初始化归一化器
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()

                # 仅用训练数据拟合归一化器
                scaler_X.fit(X_train)
                scaler_y.fit(y_train.reshape(-1, 1))

                # 转换训练和测试数据
                X_train_scaled = scaler_X.transform(X_train)
                X_test_scaled = scaler_X.transform(X_test)
                y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
                # 训练和评估SVM模型
                _, _, y_train_pred, y_test_pred = SVR(X_train_scaled,X_test_scaled,y_train_scaled,y_test,**svr_dict)

                
                
                # 将预测结果反归一化回原来的尺度
                y_test_pred = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()
        # 指标
                # 计算并返回准确率作为优化目标
                # r2 = r2_score(y_test, y_test_pred)
                print(y_test.shape)
                print(y_test_pred.shape)
                rp = pearsonr(y_test, y_test_pred)[0]
                rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                trial.set_user_attr("rmse", rmse)
                trial.set_user_attr("y_test_pred", y_test_pred)
                return rp
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trial,n_jobs=n_jobs)
            best_trial = study.best_trial
            best_rp = best_trial.value
            best_rmse = best_trial.user_attrs['rmse']
        # 把结果都存下来
            try:
                existing_df = pd.read_csv(name + "_optuna_params.csv")
            except FileNotFoundError:
                # 如果文件不存在，则创建一个新的DataFrame
                existing_df = pd.DataFrame(columns=["RP", "RMSE", "params", "y_test_pred"])
            new_data = {
                "RP": best_rp,
                "RMSE": best_rmse,
                "params": str(best_trial.params),
                "y_test_pred":np.array2string(best_trial.user_attrs['y_test_pred'], precision=5, separator=',')
            }
            new_df = pd.DataFrame(new_data,index=[0])
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_csv(n_trial + "_optuna_params.csv", index=False)
            print(new_df)
            print("----------------------------------------------------------------------------------------------------------")
            print("----------------------------------------------------------------------------------------------------------")
        except Exception as e:
            pass

class save_data_to_csv:
    
    '''import filename with path and colums , if path is not exist ,create it
    '''
    def __init__(self,file_name,columns):
        
        self.file_name = file_name
        self.columns = columns
        try:
            self.existing_df = pd.read_csv(file_name) # if the file exitsts 
        except:
            self.existing_df = pd.DataFrame(columns =columns) # if the file dose not exits ，create it
    def put(self,data):
        df = pd.DataFrame(data,columns = self.columns)
        updated_df = pd.concat([self.existing_df,df],ignore_index=True)
        updated_df.to_csv(self.file_name,index=False)

# TODO remove this function
def run_optuna_v3(X,y,isReg,chose_n_trails,selected_metric = 'r', splited_data=None,save=None,save_name= "",
                  **kw):
    
    # 2024-10-10
    # 光谱数据的自动调参函数，可以自动调整选用建模过程中的哪些方法，参数。
    # V3版本增加了允许传入已经划分好的X_train,X_test,y_train,y_test,如果splited_data传了，就不用传X,y了。
    # 相比于v1版本，v2版本增加（1）对分类任务的支持优化（2）增加了划分数据的方法 (3)增加了验证集，最终输出的是基于测试集的指标
    
    """
    -----
    params
    -----
        X: np.array 2-D
        y: np.array 1-D
        isReg: if it is a reg task, then True
        chose_n_trails: Optuna tuning iterations
        selected_metric:example: 'mae'  ---   ['mae','mse','r2','r',"accuracy", "precision", "recall"] Selecting metrics for Optuna hyperparameter tuning.  
        save: save_path  example: "./data/"
        save_name: save_name  example: "test"
        kw: example: {"selected_outlier":["不做异常值去除","mahalanobis"] ,
                       "selected_data_split":["random_split"],
                       "selected_preprocess":["不做预处理","mean_centering","snv"],
                       "selected_feat_sec":["不做特征选择"]}   --  Fill in the model dictionary included in the hyperparameter tuning
    """

    import random
    warnings.warn("此函数已废弃，将于2024-12-23被删除，请使用run_optuna_v4函数替代", DeprecationWarning, stacklevel=2)
    time.sleep(10)

    



    def choose_random_elements(my_list, num_elements):
        """从函数列表中随机选择一定数量的函数
        Parameters
        ----------
        my_list : list
            函数列表,其中的元素是函数名，字符串类型，如["snv", "msc"]
        num_elements: int
            选择的函数数量, 0选择其中一个, 其他值表示根据值随机可重复组合
        Returns
        -------
        str or list
            如果只有一个元素，返回的是字符串，否则返回的是列表
        ------
        """
    #

        if num_elements == 0:
            return random.choice(my_list)
        else:
            return random.choices(my_list, k=num_elements)

    # 修改数据
    #################################################################################################################################################
    # 获得光谱数据和标签
    X = X
    y = y
    isReg = isReg
    #################################################################################################################################################

    # 一个列表，用于存储每次调参的模型
    selection_summary_5_tab2 = [0 for i in range(8)]
    # 在选择回归任务时，提供回归模型的选择
    if isReg:
        model_options = ['LR', 'SVR','PLSR', 'Bayes(贝叶斯回归)','RFR(随机森林回归)']
        # model_options = ['BayesianRidge']
    else:
        # 在选择分类任务时，提供分类模型的选择
        model_options = ['LogisticRegression','SVM','DT','RandomForest','KNN','Bayes(贝叶斯分类)',"GradientBoostingTree","XGBoost"]

    temp_list_to_database = {}
    y_data_to_csv = {}

    ####################################################################################################################################################
    ##################################                     设置调参使用哪些参数               begin     #################################################
    ####################################################################################################################################################
    ## 选择异常值去除的选项
    selected_outlier = ["不做异常值去除","mahalanobis"]
    selected_data_split = [ "custom_train_test_split"]
    data_split_ratio  = 0.25
    selected_preprocess = ["不做预处理", "mean_centering", "normalization", "standardization", "poly_detrend", "snv", "savgol", "msc","d1", "d2", "rnv", "move_avg"]
    preprocess_number_input = 3
    selected_feat_sec = ["不做特征选择"]
    selected_dim_red = ["不做降维"]


    
    # selected_outlier = ["不做异常值去除","mahalanobis"]
    # selected_data_split = [ "custom_train_test_split"]
    # data_split_ratio  = 0.25
    # selected_preprocess = ["snv"]
    # preprocess_number_input = 1
    # selected_feat_sec = ["不做特征选择"]
    # selected_dim_red = ["不做降维"]

    
    

    if "selected_outlier" in kw.keys():
        selected_outlier = kw["selected_outlier"]
    if "selected_data_split" in kw.keys():
        selected_data_split = kw["selected_data_split","random_split"]
    if "selected_preprocess" in kw.keys():
        selected_preprocess = kw["selected_preprocess"]
    if "selected_feat_sec" in kw.keys():
        selected_feat_sec = kw["selected_feat_sec"]
    if "selected_dim_red" in kw.keys():
        selected_dim_red = kw["selected_dim_red"]
    selected_model = model_options
    if "selected_model" in kw.keys():
        selected_model = kw["selected_model"]
    reg_metric = ["mae", "mse", "r2",'r']
    classification_metric = ["accuracy", "precision", "recall"]
    selected_metric = selected_metric



    # 调参次数
    ####################################################################################################################################################
    ##################################                     设置调参使用哪些参数               end        #################################################
    ####################################################################################################################################################





 

    def param_by_optuna(trial):
        # 命名规则： {类别名：{方法名：[方法函数（对象）,{参数名: trainl对象（名称为:函数名_参数名） }]}}

        functions_ = {
            'Pretreatment': {
                '不做异常值去除': [AF.return_inputs, {}],
                'mahalanobis':
                    [AF.mahalanobis, {'threshold': trial.suggest_int('mahalanobis_threshold', 1, 100)}],
            },
            'Dataset_Splitting': {
                'random_split': [AF.random_split, {'test_size': trial.suggest_float('random_split_test_size', 0.1, 0.9) if data_split_ratio == 0 else data_split_ratio,
                                                    'random_seed':  trial.suggest_int('random_split_random_seed', 0, 100)   }],
                'custom_train_test_split':[AF.custom_train_test_split , {'test_size': trial.suggest_float('custom_train_test_split_test_size', 0.1, 0.9) if data_split_ratio == 0 else data_split_ratio,
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
                                    'n_comp': trial.suggest_int('fipls_n_comp', 2, 20)}]
            },

            'Dimensionality_reduction': {
                '不做降维': [AF.return_inputs, {}],
                'pca': [AF.pca, {'n_components': trial.suggest_int('pca_n_components', 1, 50)}],
            },


            'Model_Selection': {
                'LR': [AF.LR, {}],
                'SVR': [AF.SVR,
                            {'kernel': trial.suggest_categorical('SVR_kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                                'C': trial.suggest_float("SVR_c", 1e-5, 1000, log=True),
                                'epsilon': trial.suggest_float('SVR_epsilon', 0.01, 1, log=True),
                                'degree': trial.suggest_int('SVR_degree', 1, 5),
                                'gamma': trial.suggest_float("SVR_gamma", 1e-5, 1000, log=True)}],
                'PLSR': [AF.PLSR, {'n_components': trial.suggest_int('PLSR_n_components', 1, 20),
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
                                                'min_samples_split': trial.suggest_int('RFR(随机森林回归)_min_samples_split', 1, 100),
                                                'min_samples_leaf': trial.suggest_int('RFR(随机森林回归)_min_samples_leaf', 1, 100),
                                                'min_weight_fraction_leaf': trial.suggest_float('RFR(随机森林回归)_min_weight_fraction_leaf', 0.0001, 0.1),
                                                'max_features': trial.suggest_float('RFR(随机森林回归)_max_features', 0.1,1.0),
                                                # 'max_leaf_nodes': trial.suggest_int('RFR(随机森林回归)_max_leaf_nodes', 1, 100),
                                                # 'min_impurity_decrease': trial.suggest_float('RFR(随机森林回归)_min_impurity_decrease', 0.0001, 0.1),
                                                # 'bootstrap': trial.suggest_categorical('RFR(随机森林回归)_bootstrap', [True, False]),
                                                # 'oob_score': trial.suggest_categorical('RFR(随机森林回归)_oob_score', [True, False]),
                                                # 'n_jobs': trial.suggest_int('RFR(随机森林回归)_n_jobs', 1, 100),
                                                'random_state': trial.suggest_int('RFR(随机森林回归)_random_state', 1, 100),
                                                # 'verbose': trial.suggest_int('RFR(随机森林回归)_verbose', 0, 100),
                                                # 'warm_start': trial.suggest_categorical('RFR(随机森林回归)_warm_start', [True, False]),
                                                # 'ccp_alpha': trial.suggest_float('RFR(随机森林回归)_ccp_alpha', 0.0001, 0.1),
                                                # 'max_samples': (trial.suggest_float('RFR(随机森林回归)_max_samples', 0.1, 1.0) if trial.params['RFR(随机森林回归)_bootstrap'] else None)
                                                }],
                'BayesianRidge': [AF.BayesianRidge, {
                                                'alpha_1': trial.suggest_float("BR_alpha_1", 1e-10, 10.0, log=True),
                                                'alpha_2': trial.suggest_float("BR_alpha_2", 1e-10, 10.0, log=True),
                                                'lambda_1': trial.suggest_float("BR_lambda_1", 1e-10, 10.0, log=True),
                                                'lambda_2': trial.suggest_float("BR_lambda_2", 1e-10, 10.0, log=True),
                                                'tol': trial.suggest_float("BR_tol", 1e-6, 1e-1, log=True),
                                                'fit_intercept': trial.suggest_categorical("BR_fit_intercept", [True, False]),
                                            }],


                'LogisticRegression':[AF.logr,{}],
                'SVM': [AF.SVM, {
                        'kernel': trial.suggest_categorical('SVM_kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                        'C': trial.suggest_float("SVM_C", 1e-5, 1000, log=True),
                        'degree': trial.suggest_int('SVM_degree', 1, 5),
                        'gamma': trial.suggest_float("SVM_gamma", 1e-5, 1000, log=True),
                        # 'coef0': trial.suggest_float('SVM_coef0', 0.01, 100),
                        'random_state': trial.suggest_int('SVM_random_state', 1, 100),
                    }],
                'DT': [AF.DT, {
                    'criterion': trial.suggest_categorical('DT_criterion', ["gini", "entropy", "log_loss"]),
                    'splitter': trial.suggest_categorical('DT_splitter', ["best", "random"]),
                    'min_samples_split': trial.suggest_int('DT_min_samples_split', 2, 100),
                    'min_samples_leaf': trial.suggest_int('DT_min_samples_leaf', 1, 100),
                    # 'max_features': trial.suggest_float('DT_max_features', 0.1, 1.0),
                    'random_state': trial.suggest_int('DT_random_state', 1, 100),
                }],
                'RandomForest': [AF.RandomForest, {
                    'n_estimators': trial.suggest_int('RandomForest_n_estimators', 1, 100),
                    'criterion': trial.suggest_categorical('RandomForest_criterion',
                                                            ["entropy", "gini", "log_loss"]),
                    'min_samples_split': trial.suggest_int('RandomForest_min_samples_split', 1, 100),
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

                # 模型选择的函数和参数
            },
        }


        selection_summary_5_tab2[0] = choose_random_elements(selected_outlier, 0)
        selection_summary_5_tab2[2] = choose_random_elements(selected_data_split, 0)
        selection_summary_5_tab2[3] = choose_random_elements(selected_preprocess, preprocess_number_input)
        selection_summary_5_tab2[4] = choose_random_elements(selected_feat_sec, 0)
        selection_summary_5_tab2[7] = choose_random_elements(selected_dim_red, 0)
        selection_summary_5_tab2[5] = choose_random_elements(selected_model, 0)

    




        ## begin: 参数列表
        # global  temp_list_to_database
        temp_list_to_database[trial.number] = []
        temp_list_to_database[trial.number].append([[selection_summary_5_tab2[0]],[functions_["Pretreatment"][selection_summary_5_tab2[0]][1]]])
        temp_list_to_database[trial.number].append([[selection_summary_5_tab2[2]], [functions_["Dataset_Splitting"][selection_summary_5_tab2[2]][1]]])
        #
        #
        temp_process_name = []
        temp_process_func = []
        for func_info in selection_summary_5_tab2[3]:
            temp_process_name.append(func_info)
            temp_process_func.append(functions_["Preprocess"][func_info][1])
        temp_list_to_database[trial.number].append([temp_process_name, temp_process_func])
        # del temp_process_name, temp_process_func
        #
        #
        if selection_summary_5_tab2[4] != '':
            temp_list_to_database[trial.number].append(
                [[selection_summary_5_tab2[4]], [functions_["Feature_Selection"][selection_summary_5_tab2[4]][1]]])
        else:
            temp_list_to_database[trial.number].append([[selection_summary_5_tab2[4]], [{}]])

        temp_list_to_database[trial.number].append([[selection_summary_5_tab2[7]], [functions_["Dimensionality_reduction"][selection_summary_5_tab2[7]][1]]])
        temp_list_to_database[trial.number].append([[selection_summary_5_tab2[5]], [functions_["Model_Selection"][selection_summary_5_tab2[5]][1]]])
        #


        

        if splited_data is None:
            X_new, y_new = functions_["Pretreatment"][selection_summary_5_tab2[0]][0](X, y,
                                                                                **functions_["Pretreatment"][
                                                                                    selection_summary_5_tab2[0]][1])
            X_train, X_test, y_train, y_test = functions_["Dataset_Splitting"][selection_summary_5_tab2[2]][0](X_new,
                                                                                                                y_new,
                                                                                                                **functions_[
                                                                                                                    "Dataset_Splitting"][
                                                                                                                    selection_summary_5_tab2[
                                                                                                                        2]][
                                                                                                                    1])
        else:
            X_train, X_test, y_train, y_test = splited_data


        for func_info in selection_summary_5_tab2[3]:
            X_train, X_test,y_train,y_test = functions_["Preprocess"][func_info][0](X_train, X_test,y_train,y_test,
                                                                                    **functions_["Preprocess"][func_info][1])
        X_train, X_test,y_train,y_test = functions_["Feature_Selection"][selection_summary_5_tab2[4]][0](X_train, X_test,y_train,y_test,
                                                                                                **functions_["Feature_Selection"][
                                                                                                    selection_summary_5_tab2[
                                                                                                        4]][1])
        X_train, X_test,y_train,y_test = functions_["Dimensionality_reduction"][selection_summary_5_tab2[7]][0](X_train, X_test,y_train,y_test,
                                                                                                **functions_[
                                                                                                    "Dimensionality_reduction"][
                                                                                                    selection_summary_5_tab2[
                                                                                                        7]][1])
        y_train,y_test,y_train_pred, y_pred = functions_["Model_Selection"][selection_summary_5_tab2[5]][0](X_train, X_test,
                                                                                                    y_train, y_test,
                                                                                                    **functions_[
                                                                                                        "Model_Selection"][
                                                                                                        selection_summary_5_tab2[
                                                                                                            5]][1])
        y_data_to_csv[trial.number] = [y_test,y_pred]

        trial.set_user_attr("y_test", y_test)
        trial.set_user_attr("y_pred", y_pred)
        trial.set_user_attr("y_train", y_train)
        trial.set_user_attr("y_train_pred", y_train_pred)
        if selected_metric in reg_metric:
            if selected_metric == 'mae':
                res = mean_absolute_error(y_test, y_pred)
            elif selected_metric == 'mse':
                res = mean_squared_error(y_test, y_pred)
            elif selected_metric == 'r2':
                
                res = r2_score(y_test, y_pred)
            elif selected_metric =='r':
                from scipy.stats import pearsonr
                res = pearsonr(y_test, y_pred)[0]
            return res


        else:
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            if selected_metric == 'accuracy':
                res = accuracy_score(y_test, y_pred)
            elif selected_metric == 'precision':
                
                res = precision_score(y_test, y_pred,average="weighted")
            elif selected_metric =="recall":
                res = recall_score(y_test, y_pred,average="weighted")
            return res



    def objective(trial):
        # sc = param_by_optuna(trial)
        # return sc
        try:
            sc = param_by_optuna(trial)
            return sc
        except Exception as e:
            print("An error occurred:")
            traceback.print_exc()  # 打印完整的错误信息和堆栈跟踪


            if selected_metric in ["accuracy", "precision", "recall",'r2','r']:
                return 0  # 回归问题的默认值
            else:
                return 100000000  # 分类问题的默认值
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    # optuna.logging.enable_default_handler()
    if selected_metric in ["accuracy", "precision", "recall",'r2','r']:
        direction ='maximize'
    else:
        # 否则，默认为最小化目标，例如MAE、MSE或R2
        direction = 'minimize'
    optuna.logging.set_verbosity(optuna.logging.DEBUG) 
    study = optuna.create_study(direction=direction)
    study.optimize(objective,n_trials=chose_n_trails)
    print(study.best_value)
    print(str(temp_list_to_database[study.best_trial.number]))
    return str(temp_list_to_database[study.best_trial.number]),study.best_value

def run_optuna_v4(X,y,isReg,chose_n_trails,selected_metric = 'r', splited_data=None,save=None,save_name= "",**kw):
    # 2024-10-23
    # 光谱数据的自动调参函数，可以自动调整选用建模过程中的哪些方法，参数。
    # V4版本增加了交叉验证的功能
    # V3版本增加了允许传入已经划分好的X_train,X_test,y_train,y_test,如果splited_data传了，就不用传X,y了。
    # 相比于v1版本，v2版本增加（1）对分类任务的支持优化（2）增加了划分数据的方法 (3)增加了验证集，最终输出的是基于测试集的指标
    
    """
    功能：这是一个基于Optuna框架的自动化光谱数据建模函数，它能够自动选择和优化从数据预处理到模型训练的整个流程（包括异常值处理、数据集划分、特征选择、降维等步骤），支持多种回归和分类模型，通过5折交叉验证来评估模型性能，最终输出最优的参数组合和模型结果，是一个全面的自动化建模工具。
    -----
    params
    -----
        -  
        - X: np.array 2-D
        - y: np.array 1-D
        - isReg: if it is a reg task, then True
        - chose_n_trails: Optuna tuning iterations
        - selected_metric:example: 'mae'  ---   ['mae','mse','r2','r',"accuracy", "precision", "recall"] Selecting metrics for Optuna hyperparameter tuning.  
        - save: save_path  example: "./data/"
        - save_name: save_name  example: "test"
        - kw: example: {"selected_outlier":["不做异常值去除","mahalanobis"] ,
                       "selected_data_split":["random_split"],
                       "selected_preprocess":["不做预处理","mean_centering","snv"],
                       "selected_feat_sec":["不做特征选择"]}   --  Fill in the model dictionary included in the hyperparameter tuning

    -----
    example
    -----
        # 回归任务
        import numpy as np
        from sklearn.datasets import make_regression

        # 1. 生成示例数据
        X, y = make_regression(n_samples=500, n_features=100, noise=0.1, random_state=42)

        # 2. 配置参数
        params = {
            "selected_outlier": ["不做异常值去除", "mahalanobis"],  # 选择异常值处理方法
            "selected_preprocess": ["不做预处理", "mean_centering", "normalization", "standardization", "poly_detrend", "snv", "savgol", "msc","d1", "d2", "rnv", "move_avg"],  # 选择预处理方法 
            "selected_feat_sec": ["不做特征选择"],  # 选择特征选择方法
            "selected_model":  ['LR', 'SVR','PLSR', 'Bayes(贝叶斯回归)','RFR(随机森林回归)','BayesianRidge'] 
        }

        # 3. 运行自动建模
        results = run_optuna_v4(
            X=X,  # 输入特征矩阵 
            y=y,  # 目标变量
            isReg=True,  # 这是一个回归任务
            chose_n_trails=20,  # 尝试20次不同的参数组合
            selected_metric='r2',  # 使用R2作为评估指标
            save=None,  # 不保存结果
            save_name="",  # 不指定保存名称
            **params  # 传入上面配置的参数
        )

        # 4. 使用预先划分好的数据集的情况
        from sklearn.model_selection import train_test_split

        # 预先划分数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        splited_data = (X_train, X_test, y_train, y_test)

        # 使用预先划分的数据集运行
        results_with_split = run_optuna_v4(
            X=None,  # 当使用splited_data时,X可以为None
            y=None,  # 当使用splited_data时,y可以为None
            isReg=True,
            chose_n_trails=20,
            selected_metric='r2',
            splited_data=splited_data,  # 传入预先划分好的数据集
            **params
        )
    -----

    
    """

    import random
    import numpy as np
    import optuna
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold
    import traceback

    def choose_random_elements(my_list, num_elements):
        """从函数列表中随机选择一定数量的函数
        Parameters
        ----------
        my_list : list
            函数列表,其中的元素是函数名，字符串类型，如["snv", "msc"]
        num_elements: int
            选择的函数数量, 0选择其中一个, 其他值表示根据值随机可重复组合
        Returns
        -------
        str or list
            如果只有一个元素，返回的是字符串，否则返回的是列表
        ------
        """
        if num_elements == 0:
            return random.choice(my_list)
        else:
            return random.choices(my_list, k=num_elements)

    # 修改数据
    #################################################################################################################################################
    # 获得光谱数据和标签
    X = X
    y = y
    isReg = isReg
    #################################################################################################################################################

    # 一个列表，用于存储每次调参的模型
    selection_summary_5_tab2 = [0 for i in range(8)]
    # 在选择回归任务时，提供回归模型的选择
    if isReg:
        model_options = ['LR', 'SVR','PLSR', 'Bayes(贝叶斯回归)','RFR(随机森林回归)','BayesianRidge']
    else:
        # 在选择分类任务时，提供分类模型的选择
        model_options = ['LogisticRegression','SVM','DT','RandomForest','KNN','Bayes(贝叶斯分类)',"GradientBoostingTree","XGBoost"]

    temp_list_to_database = {}
    y_data_to_csv = {}

    ####################################################################################################################################################
    ##################################                     设置调参使用哪些参数               begin     #################################################
    ####################################################################################################################################################
    ## 选择异常值去除的选项
    selected_outlier = ["不做异常值去除","mahalanobis"]
    selected_data_split = [ "custom_train_test_split"]
    data_split_ratio  = 0.25
    selected_preprocess = ["不做预处理", "mean_centering", "normalization", "standardization", "poly_detrend", "snv", "savgol", "msc","d1", "d2", "rnv", "move_avg"]
    preprocess_number_input = 3
    selected_feat_sec = ["不做特征选择"]
    selected_dim_red = ["不做降维"]

    if "selected_outlier" in kw.keys():
        selected_outlier = kw["selected_outlier"]
    if "selected_data_split" in kw.keys():
        selected_data_split = kw["selected_data_split"]
    if "selected_preprocess" in kw.keys():
        selected_preprocess = kw["selected_preprocess"]
    if "selected_feat_sec" in kw.keys():
        selected_feat_sec = kw["selected_feat_sec"]
    if "selected_dim_red" in kw.keys():
        selected_dim_red = kw["selected_dim_red"]
    selected_model = model_options
    if "selected_model" in kw.keys():
        selected_model = kw["selected_model"]
    reg_metric = ["mae", "mse", "r2",'r']
    classification_metric = ["accuracy", "precision", "recall"]
    selected_metric = selected_metric

    def param_by_optuna(trial):
        functions_ = {
            'Pretreatment': {
                '不做异常值去除': [AF.return_inputs, {}],
                'mahalanobis':
                    [AF.mahalanobis, {'threshold': trial.suggest_int('mahalanobis_threshold', 1, 100)}],
            },
            'Dataset_Splitting': {
                'random_split': [AF.random_split, {'test_size': trial.suggest_float('random_split_test_size', 0.1, 0.9) if data_split_ratio == 0 else data_split_ratio,
                                                    'random_seed':  trial.suggest_int('random_split_random_seed', 0, 100)   }],
                'custom_train_test_split':[AF.custom_train_test_split , {'test_size': trial.suggest_float('custom_train_test_split_test_size', 0.1, 0.9) if data_split_ratio == 0 else data_split_ratio,
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
                                    'n_comp': trial.suggest_int('fipls_n_comp', 2, 20)}]
            },

            'Dimensionality_reduction': {
                '不做降维': [AF.return_inputs, {}],
                'pca': [AF.pca, {'n_components': trial.suggest_int('pca_n_components', 1, 50)}],
            },


            'Model_Selection': {
                'LR': [AF.LR, {}],
                'SVR': [AF.SVR,
                            {'kernel': trial.suggest_categorical('SVR_kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                                'C': trial.suggest_float("SVR_c", 1e-5, 1000, log=True),
                                'epsilon': trial.suggest_float('SVR_epsilon', 0.01, 1, log=True),
                                'degree': trial.suggest_int('SVR_degree', 1, 5),
                                'gamma': trial.suggest_float("SVR_gamma", 1e-5, 1000, log=True)}],
                'PLSR': [AF.PLSR, {'n_components': trial.suggest_int('PLSR_n_components', 1, 20),
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
                                                'min_samples_split': trial.suggest_int('RFR(随机森林回归)_min_samples_split', 1, 100),
                                                'min_samples_leaf': trial.suggest_int('RFR(随机森林回归)_min_samples_leaf', 1, 100),
                                                'min_weight_fraction_leaf': trial.suggest_float('RFR(随机森林回归)_min_weight_fraction_leaf', 0.0001, 0.1),
                                                'max_features': trial.suggest_float('RFR(随机森林回归)_max_features', 0.1,1.0),
                                                # 'max_leaf_nodes': trial.suggest_int('RFR(随机森林回归)_max_leaf_nodes', 1, 100),
                                                # 'min_impurity_decrease': trial.suggest_float('RFR(随机森林回归)_min_impurity_decrease', 0.0001, 0.1),
                                                # 'bootstrap': trial.suggest_categorical('RFR(随机森林回归)_bootstrap', [True, False]),
                                                # 'oob_score': trial.suggest_categorical('RFR(随机森林回归)_oob_score', [True, False]),
                                                # 'n_jobs': trial.suggest_int('RFR(随机森林回归)_n_jobs', 1, 100),
                                                'random_state': trial.suggest_int('RFR(随机森林回归)_random_state', 1, 100),
                                                # 'verbose': trial.suggest_int('RFR(随机森林回归)_verbose', 0, 100),
                                                # 'warm_start': trial.suggest_categorical('RFR(随机森林回归)_warm_start', [True, False]),
                                                # 'ccp_alpha': trial.suggest_float('RFR(随机森林回归)_ccp_alpha', 0.0001, 0.1),
                                                # 'max_samples': (trial.suggest_float('RFR(随机森林回归)_max_samples', 0.1, 1.0) if trial.params['RFR(随机森林回归)_bootstrap'] else None)
                                                }],
                'BayesianRidge': [AF.BayesianRidge, {
                                                'alpha_1': trial.suggest_float("BR_alpha_1", 1e-10, 10.0, log=True),
                                                'alpha_2': trial.suggest_float("BR_alpha_2", 1e-10, 10.0, log=True),
                                                'lambda_1': trial.suggest_float("BR_lambda_1", 1e-10, 10.0, log=True),
                                                'lambda_2': trial.suggest_float("BR_lambda_2", 1e-10, 10.0, log=True),
                                                'tol': trial.suggest_float("BR_tol", 1e-6, 1e-1, log=True),
                                                'fit_intercept': trial.suggest_categorical("BR_fit_intercept", [True, False]),
                                            }],


                'LogisticRegression':[AF.logr,{}],
                'SVM': [AF.SVM, {
                        'kernel': trial.suggest_categorical('SVM_kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                        'C': trial.suggest_float("SVM_C", 1e-5, 1000, log=True),
                        'degree': trial.suggest_int('SVM_degree', 1, 5),
                        'gamma': trial.suggest_float("SVM_gamma", 1e-5, 1000, log=True),
                        # 'coef0': trial.suggest_float('SVM_coef0', 0.01, 100),
                        'random_state': trial.suggest_int('SVM_random_state', 1, 100),
                    }],
                'DT': [AF.DT, {
                    'criterion': trial.suggest_categorical('DT_criterion', ["gini", "entropy", "log_loss"]),
                    'splitter': trial.suggest_categorical('DT_splitter', ["best", "random"]),
                    'min_samples_split': trial.suggest_int('DT_min_samples_split', 2, 100),
                    'min_samples_leaf': trial.suggest_int('DT_min_samples_leaf', 1, 100),
                    # 'max_features': trial.suggest_float('DT_max_features', 0.1, 1.0),
                    'random_state': trial.suggest_int('DT_random_state', 1, 100),
                }],
                'RandomForest': [AF.RandomForest, {
                    'n_estimators': trial.suggest_int('RandomForest_n_estimators', 1, 100),
                    'criterion': trial.suggest_categorical('RandomForest_criterion',
                                                            ["entropy", "gini", "log_loss"]),
                    'min_samples_split': trial.suggest_int('RandomForest_min_samples_split', 1, 100),
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

                # 模型选择的函数和参数
            },
        }
        
        selection_summary_5_tab2[0] = choose_random_elements(selected_outlier, 0)
        selection_summary_5_tab2[2] = choose_random_elements(selected_data_split, 0)
        selection_summary_5_tab2[3] = choose_random_elements(selected_preprocess, preprocess_number_input)
        selection_summary_5_tab2[4] = choose_random_elements(selected_feat_sec, 0)
        selection_summary_5_tab2[7] = choose_random_elements(selected_dim_red, 0)
        selection_summary_5_tab2[5] = choose_random_elements(selected_model, 0)

        temp_list_to_database[trial.number] = []
        temp_list_to_database[trial.number].append([[selection_summary_5_tab2[0]],[functions_["Pretreatment"][selection_summary_5_tab2[0]][1]]])
        temp_list_to_database[trial.number].append([[selection_summary_5_tab2[2]], [functions_["Dataset_Splitting"][selection_summary_5_tab2[2]][1]]])
        
        temp_process_name = []
        temp_process_func = []
        for func_info in selection_summary_5_tab2[3]:
            temp_process_name.append(func_info)
            temp_process_func.append(functions_["Preprocess"][func_info][1])
        temp_list_to_database[trial.number].append([temp_process_name, temp_process_func])
        
        if selection_summary_5_tab2[4] != '':
            temp_list_to_database[trial.number].append([[selection_summary_5_tab2[4]], [functions_["Feature_Selection"][selection_summary_5_tab2[4]][1]]])
        else:
            temp_list_to_database[trial.number].append([[selection_summary_5_tab2[4]], [{}]])

        temp_list_to_database[trial.number].append([[selection_summary_5_tab2[7]], [functions_["Dimensionality_reduction"][selection_summary_5_tab2[7]][1]]])
        temp_list_to_database[trial.number].append([[selection_summary_5_tab2[5]], [functions_["Model_Selection"][selection_summary_5_tab2[5]][1]]])

        if splited_data is None:
            X_new, y_new = functions_["Pretreatment"][selection_summary_5_tab2[0]][0](X, y,
                                                                                **functions_["Pretreatment"][
                                                                                    selection_summary_5_tab2[0]][1])
            X_train, X_test, y_train, y_test = functions_["Dataset_Splitting"][selection_summary_5_tab2[2]][0](X_new,
                                                                                                                y_new,
                                                                                                                **functions_[
                                                                                                                    "Dataset_Splitting"][
                                                                                                                    selection_summary_5_tab2[
                                                                                                                        2]][
                                                                                                                    1])
        else:
            X_train, X_test, y_train, y_test = splited_data

        # 应用预处理
        for func_info in selection_summary_5_tab2[3]:
            X_train, X_test, y_train, y_test = functions_["Preprocess"][func_info][0](X_train, X_test, y_train, y_test,
                                                                                    **functions_["Preprocess"][func_info][1])
        
        # 特征选择
        X_train, X_test, y_train, y_test = functions_["Feature_Selection"][selection_summary_5_tab2[4]][0](X_train, X_test, y_train, y_test,
                                                                                                **functions_["Feature_Selection"][
                                                                                                    selection_summary_5_tab2[4]][1])
        
        # 降维
        X_train, X_test, y_train, y_test = functions_["Dimensionality_reduction"][selection_summary_5_tab2[7]][0](X_train, X_test, y_train, y_test,
                                                                                                **functions_["Dimensionality_reduction"][
                                                                                                    selection_summary_5_tab2[7]][1])

        # 添加交叉验证
        n_splits = 5  # 5折交叉验证
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        
        # 在训练集上进行交叉验证
        for train_idx, val_idx in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            
            _, _, y_train_fold_pred, y_val_fold_pred = functions_["Model_Selection"][selection_summary_5_tab2[5]][0](
                X_train_fold, X_val_fold, y_train_fold, y_val_fold,
                **functions_["Model_Selection"][selection_summary_5_tab2[5]][1]
            )
            
            # 计算验证集上的性能
            if selected_metric in reg_metric:
                if selected_metric == 'mae':
                    fold_score = mean_absolute_error(y_val_fold, y_val_fold_pred)
                elif selected_metric == 'mse':
                    fold_score = mean_squared_error(y_val_fold, y_val_fold_pred)
                elif selected_metric == 'r2':
                    fold_score = r2_score(y_val_fold, y_val_fold_pred)
                elif selected_metric == 'r':
                    from scipy.stats import pearsonr
                    fold_score = pearsonr(y_val_fold, y_val_fold_pred)[0]
            else:
                from sklearn.metrics import accuracy_score, precision_score, recall_score
                if selected_metric == 'accuracy':
                    fold_score = accuracy_score(y_val_fold, y_val_fold_pred)
                elif selected_metric == 'precision':
                    fold_score = precision_score(y_val_fold, y_val_fold_pred, average="weighted")
                elif selected_metric == "recall":
                    fold_score = recall_score(y_val_fold, y_val_fold_pred, average="weighted")
            
            cv_scores.append(fold_score)

        # 在完整训练集上训练最终模型并评估测试集性能
        y_train, y_test, y_train_pred, y_pred = functions_["Model_Selection"][selection_summary_5_tab2[5]][0](
            X_train, X_test, y_train, y_test,
            **functions_["Model_Selection"][selection_summary_5_tab2[5]][1]
        )

        y_data_to_csv[trial.number] = [y_test, y_pred]
        
        # 保存结果到trial
        trial.set_user_attr("y_test", y_test)
        trial.set_user_attr("y_pred", y_pred)
        trial.set_user_attr("y_train", y_train)
        trial.set_user_attr("y_train_pred", y_train_pred)
        trial.set_user_attr("cv_scores", cv_scores)
        trial.set_user_attr("cv_score_mean", np.mean(cv_scores))
        trial.set_user_attr("cv_score_std", np.std(cv_scores))
        
        # 返回交叉验证的平均分数
        return np.mean(cv_scores)

def run_optuna_v5(data_dict, train_key, isReg, chose_n_trails, selected_metric='r', save=None, save_name="", **kw):
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
    preprocess_number_input = 3
    selected_feat_sec = ["不做特征选择"]
    selected_dim_red = ["不做降维"]

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
                                    'n_comp': trial.suggest_int('fipls_n_comp', 2, 20)}]
            },

            'Dimensionality_reduction': {
                '不做降维': [AF.return_inputs, {}],
                'pca': [AF.pca, {'n_components': trial.suggest_int('pca_n_components', 1, 50)}],
                'remove_high_variance_and_normalize':[AF.remove_high_variance_and_normalize, {'remove_feat_ratio': trial.suggest_float('remove_high_variance_and_normalize_remove_feat_ratio', 0.01, 0.5)}]
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
        # 存储当前trial的选择步骤
        trial.set_user_attr('selection_steps', {
            'outlier': selection_steps[0],
            'preprocess': selection_steps[3],
            'feature_selection': selection_steps[4],
            'dimensionality_reduction': selection_steps[7],
            'model': selection_steps[5]
        })

        # 存储当前trial的处理流程
        temp_list_to_database[trial.number] = []
        
        # # 应用预处理步骤到训练数据
        # X_processed, y_processed = functions_["Pretreatment"][selection_steps[0]][0](
        #     X_train, y_train, **functions_["Pretreatment"][selection_steps[0]][1]
        # )

        # # 应用其他预处理步骤
        # for preprocess_step in selection_steps[3]:
        #     X_processed, _, y_processed, _ = functions_["Preprocess"][preprocess_step][0](
        #         X_processed, X_processed, y_processed, y_processed,
        #         **functions_["Preprocess"][preprocess_step][1]
        #     )

        # # 特征选择和降维
        # X_processed, _, y_processed, _ = functions_["Feature_Selection"][selection_steps[4]][0](
        #     X_processed, X_processed, y_processed, y_processed,
        #     **functions_["Feature_Selection"][selection_steps[4]][1]
        # )

        # X_processed, _, y_processed, _ = functions_["Dimensionality_reduction"][selection_steps[7]][0](
        #     X_processed, X_processed, y_processed, y_processed,
        #     **functions_["Dimensionality_reduction"][selection_steps[7]][1]
        # )

        # 在所有数据集上进行预测和评估
        dataset_scores = {}
        try:
        # if True:
            test_data_dict = data_dict.copy()
            # del test_data_dict[train_key]

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
                    # 'y_test': y_test,
                    'y_pred': y_pred.tolist(),
                }

            # 存储结果
            trial.set_user_attr('dataset_scores', dataset_scores)
            
            # 返回除了训练数据集之外的所有数据集评分的平均值作为优化目标
            # del dataset_scores[train_key]
            return np.mean( [ds['score'] for key,ds in dataset_scores.items() if key is not train_key] )
            # return np.mean([ds['score'] for ds in dataset_scores.values() ])
        except Exception as e:
            print(f"Error in trial {trial.number}: {str(e)}")
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

def rebuild_model(X, y, splited_data=None, stored_str=None):

    import ast
    """
    Rebuild the model based on the stored string of parameters and configuration.
    
    Params:
        X: np.array 2-D, feature matrix
        y: np.array 1-D, labels
        splited_data: tuple (X_train, X_test, y_train, y_test) if provided, otherwise the data will be split inside the function
        stored_str: string representation of the temp_list_to_database to be parsed
    
    Returns:
        Final model predictions on the test set (y_pred) and test labels (y_test)
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
                'fipls': [AF.fipls, {}]
            },

            'Dimensionality_reduction': {
                '不做降维': [AF.return_inputs, {}],
                'pca': [AF.pca, {}],
            },


            'Model_Selection': {
                'LR': [AF.LR, {}],
                'SVR': [AF.SVR,
                            {}],
                'PLSR': [AF.PLSR, {}],
                'Bayes(贝叶斯回归)': [AF.bayes, {}],
                'RFR(随机森林回归)': [AF.RFR, {}],
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
    
    # Parse the stored_str back into the original list format
    param_list = ast.literal_eval(stored_str)
    
    # Extract the steps in the stored parameter list
    outlier_removal = param_list[0]
    data_splitting = param_list[1]
    preprocess_list = param_list[2]
    feat_selection = param_list[3]
    dim_reduction = param_list[4]
    model_selection = param_list[5]


    # Apply the dataset splitting
    if splited_data is None:
        X_new, y_new = functions_["Pretreatment"][outlier_removal[0][0]][0](X, y, **outlier_removal[1][0])

        X_train, X_test, y_train, y_test = functions_["Dataset_Splitting"][data_splitting[0][0]][0](
            X_new, y_new, **data_splitting[1][0]
        )
    else:
        X_train, X_test, y_train, y_test = splited_data

    # Apply preprocessing steps
    for func_info, func_params in zip(preprocess_list[0], preprocess_list[1]):
        X_train, X_test, y_train, y_test = functions_["Preprocess"][func_info][0](X_train, X_test, y_train, y_test, **func_params)

    # Apply feature selection
    X_train, X_test, y_train, y_test = functions_["Feature_Selection"][feat_selection[0][0]][0](X_train, X_test, y_train, y_test, **feat_selection[1][0])

    # Apply dimensionality reduction
    X_train, X_test, y_train, y_test = functions_["Dimensionality_reduction"][dim_reduction[0][0]][0](X_train, X_test, y_train, y_test, **dim_reduction[1][0])

    # Apply the model selection and fit the model
    y_train, y_test, y_train_pred, y_pred = functions_["Model_Selection"][model_selection[0][0]][0](
        X_train, X_test, y_train, y_test, **model_selection[1][0]
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

def run_regression_optuna(data_name,X,y,model='PLS',split = 'SPXY',test_size = 0.3, n_trials=200,object = None,cv = None,save_dir = None):
    
    
    ''''
    -----
    params:
    -----
        data_name: 数据集名称
        X: 特征数据
        y: 标签数据
        model: 选择模型，{"PLS","SVR","RFreg","LR"}
        split: 选择数据集划分方式，{"SPXY","Random"}
        n_trials: 选择优化次数
        object: 选择优化目标，{"R2","MAE","MSE","Pearsonr"}
        cv: 交叉验证方式,填整数就是k折交叉验证，填None不做交叉验证
        save_dir: 保存结果的名称
        
    -----
    return:
    -----
        模型，数据
    '''
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
    print("警告：当前函数即将废弃，请使用最新版本 (run_regression_optuna ")
    
    
    if split == 'SPXY':
        X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size, 'SPXY')
    elif split == "Random":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        # raise ValueError("split_name error")
        # 结束
        assert False, "split_name error"
    
    # 归一化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)
    
    def objective(trial):

        try:
            if model == 'PLS':
                n_components = trial.suggest_int('n_components', 1, 600)
                regressor = PLSRegression(n_components=n_components)
            elif model == 'SVR':
                C = trial.suggest_float('C', 1e-3, 1e3,log=True)
                gamma = trial.suggest_float('gamma', 1e-3, 1e3,log=True)
                regressor = SVR(C=C, gamma=gamma)
            elif model == 'RFreg':
                n_estimators = trial.suggest_int('n_estimators', 50, 200)
                max_depth = trial.suggest_int('max_depth', 5, 30)
                max_features = trial.suggest_float('max_features', 0.0, 1.0)
                min_samples_split = trial.suggest_float('min_samples_split', 0, 1.0)
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
                regressor.fit(X_train_scaled, y_train)
                y_pred = regressor.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                pearsonr = np.corrcoef(y_test.flatten(), y_pred.flatten())[0, 1]
                print(f"mae: {mae:.3f}, rmse: {rmse:.3f}, r2: {r2:.3f}, pearsonr: {pearsonr:.3f}")
                return np.mean(score)
            else:
                regressor.fit(X_train_scaled, y_train)
                y_pred = regressor.predict(X_test_scaled)
                if object == 'R2':
                    score = r2_score(y_test, y_pred)
                elif object == 'MAE':
                    score = -mean_absolute_error(y_test, y_pred)
                elif object == 'RMSE':
                    score = - np.sqrt(mean_squared_error(y_test, y_pred))
                else:
                    # raise ValueError("object_name error")
                    assert False, "object_name error"
                

                # 当前模型的评估指标
                mae = mean_absolute_error(y_test, y_pred)
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

    regressor_final.fit(X_train_scaled, y_train)



    y_pred_train = regressor_final.predict(X_train_scaled)
    y_pred_test = regressor_final.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    print(f"mae: {mae:.3f}, rmse: {rmse:.3f}, r2: {r2:.3f}")
    
    train_and_test_pred_plot(y_train,y_test,y_pred_train,y_pred_test,data_name=data_name,save_dir=save_dir)
    
    # 保存模型和数据
    joblib.dump(regressor_final, f"{save_dir}/{model}_{data_name}_model.pkl")
    pd.DataFrame(y_test).to_csv(f"{save_dir}/{model}_{data_name}_y_test.csv",index=False)
    pd.DataFrame(y_pred_test).to_csv(f"{save_dir}/{model}_{data_name}_y_pred_test.csv",index=False)
    pd.DataFrame(y_train).to_csv(f"{save_dir}/{model}_{data_name}_y_train.csv",index=False)
    pd.DataFrame(y_pred_train).to_csv(f"{save_dir}/{model}_{data_name}_y_pred_train.csv",index=False)
    pd.DataFrame(best_params,index=[0]).to_csv(f"{save_dir}/{model}_{data_name}_best_params.csv",index=False)
    
    return regressor_final,[X_train_scaled,X_test_scaled,y_train,y_test,y_pred_train,y_pred_test]

def run_regression_optuna_v2(data_name,X = None,y=None ,data_splited = None,  model='PLS',split = 'SPXY',test_size = 0.3, n_trials=200,object = None,cv = None,save_dir = None):
    # 增加功能，支持自定义输入训练测试集  data_splited
    
    ''''
    -----
    params:
    -----
        data_name: 数据集名称
        X: 特征数据
        y: 标签数据
        data_splited: 划分好训练测试数据的字典，{'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test}
        model: 选择模型，{"PLS","SVR","RFreg","LR"}
        split: 选择数据集划分方式，{"SPXY","Random"}
        n_trials: 选择优化次数
        object: 选择优化目标，{"R2","MAE","RMSE","Pearsonr"}
        cv: 交叉验证方式,填整数就是k折交叉验证，填None不做交叉验证
        save_dir: 保存结果的名称
        
    -----
    return:
    -----
        模型，数据
    '''
    print("警告：该函数即将废弃，请使用最新版本 (run_regression_optuna_v3 ")
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
        X_test = data_splited['X_test']
        y_train = data_splited['y_train']
        y_test = data_splited['y_test']
    elif X is not None and y is not None:
        if split == 'SPXY':
            X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size, 'SPXY')

        elif split == "Random":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        else:
            assert False, "split_name error"
    
    
    # 归一化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)
    
    def objective(trial):

        try:
            if model == 'PLS':
                n_components = trial.suggest_int('n_components', 1, 600)
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
                regressor.fit(X_train_scaled, y_train)
                y_pred = regressor.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                pearsonr = np.corrcoef(y_test.flatten(), y_pred.flatten())[0, 1]
                print(f"mae: {mae:.3f}, rmse: {rmse:.3f}, r2: {r2:.3f}, pearsonr: {pearsonr:.3f}")
                return np.mean(score)
            else:
                regressor.fit(X_train_scaled, y_train)
                y_pred = regressor.predict(X_test_scaled)
                if object == 'R2':
                    score = r2_score(y_test, y_pred)
                elif object == 'MAE':
                    score = -mean_absolute_error(y_test, y_pred)
                elif object == 'RMSE':
                    score = - np.sqrt(mean_squared_error(y_test, y_pred))
                else:
                    # raise ValueError("object_name error")
                    assert False, "object_name error"
                

                # 当前模型的评估指标
                mae = mean_absolute_error(y_test, y_pred)
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

    regressor_final.fit(X_train_scaled, y_train)



    y_pred_train = regressor_final.predict(X_train_scaled)
    y_pred_test = regressor_final.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    print(f"mae: {mae:.3f}, rmse: {rmse:.3f}, r2: {r2:.3f}")
    
    train_and_test_pred_plot(y_train,y_test,y_pred_train,y_pred_test,data_name=data_name,save_dir=save_dir)
    
    # 保存模型和数据
    joblib.dump(regressor_final, f"{save_dir}/{model}_{data_name}_model.pkl")
    pd.DataFrame(y_test).to_csv(f"{save_dir}/{model}_{data_name}_y_test.csv",index=False)
    pd.DataFrame(y_pred_test).to_csv(f"{save_dir}/{model}_{data_name}_y_pred_test.csv",index=False)
    pd.DataFrame(y_train).to_csv(f"{save_dir}/{model}_{data_name}_y_train.csv",index=False)
    pd.DataFrame(y_pred_train).to_csv(f"{save_dir}/{model}_{data_name}_y_pred_train.csv",index=False)
    pd.DataFrame(best_params,index=[0]).to_csv(f"{save_dir}/{model}_{data_name}_best_params.csv",index=False)
    
    return regressor_final,[X_train_scaled,X_test_scaled,y_train,y_test,y_pred_train,y_pred_test]

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

# 把所有PD减取连续噪声
def PD_reduce_noise(PD_samples, PD_noise, ratio=9,base_noise= None):

    if base_noise is None:
        base_noise = np.mean(PD_noise[0])
    PD_noise_X_mean = np.mean(PD_noise, axis=1)
    # data_X_mean_diff = data_X_mean[1:] - data_X_mean[:-1]
    PD_samples_new = np.zeros_like(PD_samples)
    for i in range(0,len(PD_samples)):
        PD_samples_new[i,:] = PD_samples[i,:] - (PD_noise_X_mean[i]-base_noise)*ratio
    return PD_samples_new

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

def convex_optimization_recon_for_MZI(PD_list,s21_data_path = 'S21.mat'):
    '''MZI样机的重建算法，采用凸优化方法，输入为S21数据和PD值列表，输出为重建的S21数据
    ------
    parameters:
    ------
        s21_data: 输入的S21数据，格式为mat文件的地址
        PD_list: 输入的PD值列表，格式为csv文件  PD_mW_band1, PD_mW_band2, PD_mW_band3, PD_mW_band4, PD_source_mW_band1, PD_source_mW_band2, PD_source_mW_band3, PD_source_mW_band4 = PD_list
    ------
    return: [样品重建光谱、光源重建光谱]
    ------
        S21_rec: 重建的S21数据
    
    '''
    import os
    import numpy as np
    import scipy.io as sio
    import matplotlib.pyplot as plt
    from datetime import datetime
    from os.path import join, exists
    from os import getcwd, mkdir
    from sporco.admm import bpdn
    import cvxpy as cp
    from sklearn.preprocessing import normalize
    from scipy.io import loadmat
    from scipy.interpolate import interp1d
    def recon_core_body_v2(wl_num_rec, PD_mW, Trans_cut, alpha1, flag):

        a = np.ones(wl_num_rec - 1)
        gamma1_1 = np.diag(-1 * a, 0)
        gamma1_1 = np.pad(gamma1_1, ((0, 1), (0, 1)), mode='constant')
        gamma1_2 = np.diag(a, 1)
        gamma1 = gamma1_1 + gamma1_2
        gamma1 = gamma1[:-1, :]

        if flag == 1:
            alpha1 = alpha1[:-1]
            # Optimization
        rec = cp.Variable(wl_num_rec)
        objective = cp.Minimize(
            cp.sum_squares(Trans_cut @ rec - PD_mW) + cp.sum_squares(cp.multiply(alpha1, gamma1 @ rec)))
        constraints = [rec >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, verbose=True)

        rec_abs = rec.value
        rec_norm = rec_abs / np.linalg.norm(rec_abs, np.inf)
        rec_norm = normalize(rec_abs.reshape(1, -1), norm='max')

        return rec_abs

    a_band1 = np.zeros(281)
    a_band1[:63] = np.linspace(1.5, 0.5, 63)
    a_band1[63:243] = 0.5
    a_band1[243:] = np.linspace(0.5, 1.5, 38)

    a_band2 = np.zeros(201)
    a_band2[:39] = np.linspace(0.7, 0.1, 39)
    a_band2[39:151] = 0.1
    a_band2[151:181] = np.linspace(0.1, 1.5, 30)
    a_band2[181:] = np.linspace(1.5, 0.5, 20)

    a_band3 = np.zeros(201)
    a_band3[:36] = np.linspace(0.7, 0.1, 36)
    a_band3[36:177] = 0.1
    a_band3[177:] = np.linspace(0.1, 0.7, 24)

    a_band4 = np.zeros(201)
    a_band4[:23] = np.linspace(1, 0.2, 23)
    a_band4[23:184] = 0.2
    a_band4[184:] = np.linspace(0.2, 1, 17)

    alpha1_meas = [a_band1, a_band2, a_band3, a_band4]  # 四个区间的参数
    alpha1_source = np.array([0.05, 0.05, 0.05, 0.05])
    s21_data = loadmat(s21_data_path)
    wl = s21_data['wl'].squeeze()

    S21_main_95_T = s21_data['S21_main_95_T'].T  # (81,2000)
    S21_main_05_T = s21_data['S21_main_05_T'].T  # (81,2000)
    S21_2nd_95_T = s21_data['S21_2nd_95_T'].T
    S21_2nd_05_T = s21_data['S21_2nd_05_T'].T

    band1_nm = np.linspace(1240, 1380, 281)
    band2_nm = np.linspace(1390, 1490, 201)
    band3_nm = np.linspace(1500, 1600, 201)
    band4_nm = np.linspace(1610, 1700, 201)

    # 插值
    interp_func1 = interp1d(wl, S21_main_95_T)  # 定义一个插值函数。它根据给定的数据点生成一个连续的、可调用的函数，你可以用这个函数来计算任何在原始数据点范围内的点的插值。
    S21_band1_meas_T = interp_func1(band1_nm)
    S21_band2_meas_T = interp_func1(band2_nm)
    interp_func1 = interp1d(wl, S21_2nd_95_T)
    S21_band3_meas_T = interp_func1(band3_nm)
    S21_band4_meas_T = interp_func1(band4_nm)

    interp_func2 = interp1d(wl, S21_main_05_T)
    S21_band1_source_T = interp_func2(band1_nm)
    S21_band2_source_T = interp_func2(band2_nm)
    interp_func2 = interp1d(wl, S21_2nd_05_T)
    S21_band3_source_T = interp_func2(band3_nm)
    S21_band4_source_T = interp_func2(band4_nm)

    # PD_file_name = 'PD_current_%s.csv' % project_name
    # PD_path = join(getcwd(), 'PD', PD_file_name)


    # 检查PD_current文件是否存在，PD文件夹中最多只能有一个文件
    PD_mW_band1, PD_mW_band2, PD_mW_band3, PD_mW_band4, PD_source_mW_band1, PD_source_mW_band2, PD_source_mW_band3, PD_source_mW_band4 = PD_list
    if PD_mW_band1 is None:
        print("请检查文件是否正确")
        return
    t_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    rec_meas_band1 = recon_core_body_v2(len(band1_nm), PD_mW_band1, S21_band1_meas_T, alpha1_meas[0], 1)
    rec_meas_band2 = recon_core_body_v2(len(band2_nm), PD_mW_band2, S21_band2_meas_T, alpha1_meas[1], 1)
    rec_meas_band3 = recon_core_body_v2(len(band3_nm), PD_mW_band3, S21_band3_meas_T, alpha1_meas[2], 1)
    rec_meas_band4 = recon_core_body_v2(len(band4_nm), PD_mW_band4, S21_band4_meas_T, alpha1_meas[3], 1)

    rec_source_band1 = recon_core_body_v2(len(band1_nm), PD_source_mW_band1, S21_band1_source_T, alpha1_source[0],
                                            0)
    rec_source_band2 = recon_core_body_v2(len(band2_nm), PD_source_mW_band2, S21_band2_source_T, alpha1_source[1],
                                            0)
    rec_source_band3 = recon_core_body_v2(len(band3_nm), PD_source_mW_band3, S21_band3_source_T, alpha1_source[2],
                                            0)
    rec_source_band4 = recon_core_body_v2(len(band4_nm), PD_source_mW_band4, S21_band4_source_T, alpha1_source[3],
                                            0)

    Recon_band1 = rec_meas_band1
    Recon_band2 = rec_meas_band2
    Recon_band3 = rec_meas_band3
    Recon_band4 = rec_meas_band4

    return np.concatenate([Recon_band1, Recon_band2, Recon_band3, Recon_band4]), np.concatenate(
        [rec_source_band1, rec_source_band2, rec_source_band3, rec_source_band4])

def create_dataset_by_file_path_v1(file_path):
    from nirapi.load_data import load_prototype_data
    """Load the datasets from a given file path.
    -----
    params:
    -----
        file_path: str, the file path of the dataset.  (.xlsx file)
    -----
    return:
    -----
        data: pandas DataFrame, the combined dataset 
    """
    corrected_spectrum, labels = load_prototype_data(file_path=file_path, pos="Corrected spectrum"), load_prototype_data(file_path=file_path, pos="y")

    data = pd.DataFrame(corrected_spectrum)
    data["label"] = labels
    return data

def load_training_data_v1(file_path=r"../data/MZI酒精数据20240921 - 校正-旧数据.xlsx",type = "Corrected spectrum"):
    '''
    -------
    params:
    -------
    file_path: str, 文件路径
    type: str, 数据类型，可'PD Sample','Recon Sample','Corrected spectrum','Biomark','Measured_Value','y','date_time','volunteer'

    '''

    if type == "Corrected spectrum":
        return load_prototype_data(file_path = file_path,pos="Corrected spectrum"),load_prototype_data(file_path = file_path,pos="y")
        
    elif type == "PD Sample":
        return None
    return  NotImplementedError("暂不支持该数据类型")

def load_and_split_data_v1(file_path=r"../data/MZI酒精数据20240921 - 校正-旧数据.xlsx", type="Corrected spectrum", split_index=670, train_ratio=0.6, val_ratio=0.2):
    '''
    功能：
    该函数从 Excel 文件中加载特定类型的数据（这里默认为“Corrected spectrum”类型）并根据指定的比例分割成训练集、验证集和测试集。
    -------
    params:
    -------
    file_path: str, 文件路径
    type: str, 数据类型, 可'PD Sample', 'Recon Sample', 'Corrected spectrum', 'Biomark', 'Measured_Value', 'y', 'date_time', 'volunteer'
    split_index: int, 分割数据集的位置, 默认670
    train_ratio: float, 训练集比例, 默认60%
    val_ratio: float, 验证集比例, 默认20%

    -------
    returns:
    -------
    train_data: DataFrame, 训练集
    val_data: DataFrame, 验证集
    test_data: DataFrame, 测试集

    # 使用例子
    train_data, val_data, test_data = load_and_split_data_v2(file_path=r"../data/MZI酒精数据20240921 - 校正-旧数据.xlsx")
    '''
    
    # 加载数据
    if type == "Corrected spectrum":
        corrected_spectrum, labels = load_prototype_data(file_path=file_path, pos="Corrected spectrum"), load_prototype_data(file_path=file_path, pos="y")
    else:
        raise NotImplementedError("暂不支持该数据类型")
    
    # 合并数据
    data = pd.DataFrame(corrected_spectrum)
    data['label'] = labels
    
    # 按时间顺序进行分割，前670条用于训练和验证，剩余作为测试集
    train_val_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    
    # 计算训练集和验证集的分割位置
    train_size = int(train_ratio * split_index)
    val_size = int(val_ratio * split_index)
    
    train_data = train_val_data.iloc[:train_size]
    val_data = train_val_data.iloc[train_size:train_size + val_size]
    
    return train_data, val_data, test_data

def load_and_split_data_v2(file_path=r"../data/MZI酒精数据20240921 - 校正-旧数据.xlsx", 
                           type="Corrected spectrum", 
                           split_index=670, 
                           train_ratio=0.6, 
                           val_ratio=0.2, 
                           volunteer_name=None):
    '''
    -------
    params:
    -------
    file_path: str, 文件路径
    type: str, 数据类型, 可'PD Sample', 'Recon Sample', 'Corrected spectrum', 'Biomark', 'Measured_Value', 'y', 'date_time', 'volunteer'
    split_index: int, 分割数据集的位置, 默认670
    train_ratio: float, 训练集比例, 默认60%
    val_ratio: float, 验证集比例, 默认20%
    volunteer_name: str, 志愿者名字，选择特定志愿者的数据 (可选参数)

    -------
    returns:
    -------
    train_data: DataFrame, 训练集
    val_data: DataFrame, 验证集
    test_data: DataFrame, 测试集

    # 使用例子
    train_data, val_data, test_data = load_and_split_data_v2(file_path=r"../data/MZI酒精数据20240921 - 校正-旧数据.xlsx", volunteer_name="志愿者A")
    '''
    
    # 加载数据
    if type == "Corrected spectrum":
        corrected_spectrum, labels = load_prototype_data(file_path=file_path, pos="Corrected spectrum"), load_prototype_data(file_path=file_path, pos="y")
        volunteer_names = load_prototype_data(file_path=file_path, pos="volunteer")
    else:
        raise NotImplementedError("暂不支持该数据类型")
    
    # 合并数据
    data = pd.DataFrame(corrected_spectrum)
    data['label'] = labels
    data['volunteer'] = volunteer_names

    # 如果指定了志愿者名字，筛选出该志愿者的数据
    if volunteer_name is not None:
        data = data[data['volunteer'] == volunteer_name]
    
    # 按时间顺序进行分割，前670条用于训练和验证，剩余作为测试集
    train_val_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    
    # 计算训练集和验证集的分割位置
    train_size = int(train_ratio * split_index)
    val_size = int(val_ratio * split_index)
    
    train_data = train_val_data.iloc[:train_size]
    val_data = train_val_data.iloc[train_size:train_size + val_size]
    
    return train_data, val_data, test_data

def load_and_split_data_v3(file_path=r"../data/MZI酒精数据20240921 - 校正-旧数据.xlsx", 
                           type="Corrected spectrum", 
                           split_index=670, 
                           train_ratio=0.6, 
                           val_ratio=0.2, 
                           volunteer_name=None):
    '''
    这个函数是 load_and_split_data_v1 的改进版本，除了支持分割数据之外，还增加了对志愿者名字的筛选功能(先分割再筛选）。
    -------
    params:
    -------
    file_path: str, 文件路径
    type: str, 数据类型, 可'PD Sample', 'Recon Sample', 'Corrected spectrum', 'Biomark', 'Measured_Value', 'y', 'date_time', 'volunteer'
    split_index: int, 分割数据集的位置, 默认670
    train_ratio: float, 训练集比例, 默认60%
    val_ratio: float, 验证集比例, 默认20%
    volunteer_name: str, 志愿者名字，选择特定志愿者的数据 (可选参数)

    -------
    returns:
    -------
    train_data: DataFrame, 训练集
    val_data: DataFrame, 验证集
    test_data: DataFrame, 测试集

    # 使用例子
    train_data, val_data, test_data = load_and_split_data_v3(file_path=r"../data/MZI酒精数据20240921 - 校正-旧数据.xlsx", volunteer_name="志愿者A")
    '''
    
    # 加载数据
    if type == "Corrected spectrum":
        corrected_spectrum, labels = load_prototype_data(file_path=file_path, pos="Corrected spectrum"), load_prototype_data(file_path=file_path, pos="y")
        volunteer_names = load_prototype_data(file_path=file_path, pos="volunteer")
    else:
        raise NotImplementedError("暂不支持该数据类型")
    
    # 合并数据
    data = pd.DataFrame(corrected_spectrum)
    data['label'] = labels
    data['volunteer'] = volunteer_names
    
    # 按时间顺序进行分割，前670条用于训练和验证，剩余作为测试集
    train_val_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    
    # 计算训练集和验证集的分割位置
    train_size = int(train_ratio * split_index)
    val_size = int(val_ratio * split_index)
    
    train_data = train_val_data.iloc[:train_size]
    val_data = train_val_data.iloc[train_size:train_size + val_size]
    
    # 如果指定了志愿者名字，筛选出该志愿者的数据
    if volunteer_name is not None:
        train_data = train_data[train_data['volunteer'] == volunteer_name]
        val_data = val_data[val_data['volunteer'] == volunteer_name]
        test_data = test_data[test_data['volunteer'] == volunteer_name]

        train_data.drop(columns=['volunteer'], inplace=True)
        val_data.drop(columns=['volunteer'], inplace=True)
        test_data.drop(columns=['volunteer'], inplace=True)
    
    return train_data, val_data, test_data

def plot_distribution(train_data, val_data, test_data, label_column = 'label'):
    """
    Plots the distribution of labels in training, validation, and test sets.

    Parameters:
    train_data (pd.DataFrame): Training dataset
    val_data (pd.DataFrame): Validation dataset
    test_data (pd.DataFrame): Test dataset
    label_column (str): Name of the column containing labels
    """
    # Example usage:
    #  plot_distribution(train_data, val_data, test_data, 'label')
    plt.figure(figsize=(12, 6))


    # Plotting Training set
    plt.subplot(1, 3, 1)
    plt.hist(train_data[label_column], bins=2, edgecolor='black')
    plt.title('Training Set Distribution')
    plt.xlabel('Label')
    plt.ylabel('Frequency')

    # Plotting Validation set
    plt.subplot(1, 3, 2)
    plt.hist(val_data[label_column], bins=2, edgecolor='black')
    plt.title('Validation Set Distribution')
    plt.xlabel('Label')
    plt.ylabel('Frequency')

    # Plotting Test set
    plt.subplot(1, 3, 3)
    plt.hist(test_data[label_column], bins=2, edgecolor='black')
    plt.title('Test Set Distribution')
    plt.xlabel('Label')
    plt.ylabel('Frequency')

    # Displaying the plots
    plt.tight_layout()
    plt.show()

def plot_3d_pca_scatter(data, label_col='label', n_components=3):
    """
    使用PCA降维并绘制3D散点图
    -----------
    params:
    -----------
    data: pd.DataFrame, 包含特征和标签的数据集
    label_col: str, 标签列的名称，默认 'label'
    n_components: int, PCA降维的目标维度，默认 3
    
    -----------
    returns:
    -----------
    None: 直接展示3D散点图
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 提取特征和标签
    X = data.drop(columns=[label_col])
    y = data[label_col]

    # 使用PCA进行降维
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # 绘制三维散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 不同标签用不同颜色表示
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', s=50)

    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Labels')

    # 添加标题和坐标轴标签
    ax.set_title('3D PCA Scatter Plot')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')

    # 显示图形
    plt.show()

def plot_3d_pca_combined(train_data, val_data, test_data, label_col='label', n_components=3):
    """
    Perform PCA on the combined train, validation, and test datasets, and plot them in a 3D scatter plot.
    
    Parameters:
    ----------
    train_data : pd.DataFrame
        Training dataset including features and labels.
    val_data : pd.DataFrame
        Validation dataset including features and labels.
    test_data : pd.DataFrame
        Test dataset including features and labels.
    label_col : str, optional
        The name of the label column. Default is 'label'.
    n_components : int, optional
        The number of PCA components to reduce to. Default is 3.
        
    Returns:
    -------
    None
        Displays the 3D scatter plot.
    """
    # Combine the datasets and drop the label column
    combined_data = pd.concat([train_data, val_data, test_data], axis=0).drop(columns=[label_col])
    
    # Perform PCA to reduce the combined data to the specified number of components
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(combined_data)
    
    # Get the sizes of the train, validation, and test datasets
    train_size = len(train_data)
    val_size = len(val_data)
    
    # Split the PCA results back into the respective datasets
    train_pca = pca_result[:train_size]
    val_pca = pca_result[train_size:train_size + val_size]
    test_pca = pca_result[train_size + val_size:]
    
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot train data
    ax.scatter(train_pca[:, 0], train_pca[:, 1], train_pca[:, 2], c='blue', label='Train', alpha=0.6)

    # Plot validation data
    ax.scatter(val_pca[:, 0], val_pca[:, 1], val_pca[:, 2], c='green', label='Validation', alpha=0.6)

    # Plot test data
    ax.scatter(test_pca[:, 0], test_pca[:, 1], test_pca[:, 2], c='red', label='Test', alpha=0.6)

    # Set axis labels and title
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title('PCA 3D Visualization of Train, Validation, and Test Data')

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

def plot_2d_pca_combined(train_data, val_data, test_data, label_col='label', n_components=2):
    """
    Perform PCA on the combined train, validation, and test datasets, and plot them in a 2D scatter plot.
    
    Parameters:
    ----------
    train_data : pd.DataFrame
        Training dataset including features and labels.
    val_data : pd.DataFrame
        Validation dataset including features and labels.
    test_data : pd.DataFrame
        Test dataset including features and labels.
    label_col : str, optional
        The name of the label column. Default is 'label'.
    n_components : int, optional
        The number of PCA components to reduce to. Default is 2.
        
    Returns:
    -------
    None
        Displays the 2D scatter plot.
    """
    # Combine the datasets and drop the label column
    combined_data = pd.concat([train_data, val_data, test_data], axis=0).drop(columns=[label_col])
    
    # Perform PCA to reduce the combined data to the specified number of components
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(combined_data)
    
    # Get the sizes of the train, validation, and test datasets
    train_size = len(train_data)
    val_size = len(val_data)
    
    # Split the PCA results back into the respective datasets
    train_pca = pca_result[:train_size]
    val_pca = pca_result[train_size:train_size + val_size]
    test_pca = pca_result[train_size + val_size:]
    
    # Create a 2D scatter plot
    plt.figure(figsize=(10, 8))

    # Plot train data
    plt.scatter(train_pca[:, 0], train_pca[:, 1], c='blue', label='Train', alpha=0.6)

    # Plot validation data
    plt.scatter(val_pca[:, 0], val_pca[:, 1], c='green', label='Validation', alpha=0.6)

    # Plot test data
    plt.scatter(test_pca[:, 0], test_pca[:, 1], c='red', label='Test', alpha=0.6)

    # Set axis labels and title
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('PCA 2D Visualization of Train, Validation, and Test Data')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

def reconForMZI_CVX(PD_list,s21_data_path = 'S21.mat'):
    '''MZI样机的重建算法，采用凸优化方法，输入为S21数据和PD值列表，输出为重建的S21数据
    ------
    parameters:
    ------
        s21_data: 输入的S21数据，格式为mat文件的地址
        PD_list: 输入的PD值列表 PD_mW_band1, PD_mW_band2, PD_mW_band3, PD_mW_band4, PD_source_mW_band1, PD_source_mW_band2, PD_source_mW_band3, PD_source_mW_band4 = PD_list
    ------
    return: [样品重建光谱、光源重建光谱]
    ------
        S21_rec: 重建的S21数据
    
    '''
    import os
    import numpy as np
    import scipy.io as sio
    import matplotlib.pyplot as plt
    from datetime import datetime
    from os.path import join, exists
    from os import getcwd, mkdir
    from sporco.admm import bpdn
    import cvxpy as cp
    from sklearn.preprocessing import normalize
    from scipy.io import loadmat
    from scipy.interpolate import interp1d
    def recon_core_body_v2(wl_num_rec, PD_mW, Trans_cut, alpha1, flag):

        a = np.ones(wl_num_rec - 1)
        gamma1_1 = np.diag(-1 * a, 0)
        gamma1_1 = np.pad(gamma1_1, ((0, 1), (0, 1)), mode='constant')
        gamma1_2 = np.diag(a, 1)
        gamma1 = gamma1_1 + gamma1_2
        gamma1 = gamma1[:-1, :]

        if flag == 1:
            alpha1 = alpha1[:-1]
            # Optimization
        rec = cp.Variable(wl_num_rec)
        objective = cp.Minimize(
            cp.sum_squares(Trans_cut @ rec - PD_mW) + cp.sum_squares(cp.multiply(alpha1, gamma1 @ rec)))
        constraints = [rec >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)

        rec_abs = rec.value
        rec_norm = rec_abs / np.linalg.norm(rec_abs, np.inf)
        rec_norm = normalize(rec_abs.reshape(1, -1), norm='max')

        return rec_abs

    a_band1 = np.zeros(281)
    a_band1[:63] = np.linspace(1.5, 0.5, 63)
    a_band1[63:243] = 0.5
    a_band1[243:] = np.linspace(0.5, 1.5, 38)

    a_band2 = np.zeros(201)
    a_band2[:39] = np.linspace(0.7, 0.1, 39)
    a_band2[39:151] = 0.1
    a_band2[151:181] = np.linspace(0.1, 1.5, 30)
    a_band2[181:] = np.linspace(1.5, 0.5, 20)

    a_band3 = np.zeros(201)
    a_band3[:36] = np.linspace(0.7, 0.1, 36)
    a_band3[36:177] = 0.1
    a_band3[177:] = np.linspace(0.1, 0.7, 24)

    a_band4 = np.zeros(201)
    a_band4[:23] = np.linspace(1, 0.2, 23)
    a_band4[23:184] = 0.2
    a_band4[184:] = np.linspace(0.2, 1, 17)

    alpha1_meas = [a_band1, a_band2, a_band3, a_band4]  # 四个区间的参数
    alpha1_source = np.array([0.05, 0.05, 0.05, 0.05])
    s21_data = loadmat(s21_data_path)
    wl = s21_data['wl'].squeeze()

    S21_main_95_T = s21_data['S21_main_95_T'].T  # (81,2000)
    S21_main_05_T = s21_data['S21_main_05_T'].T  # (81,2000)
    S21_2nd_95_T = s21_data['S21_2nd_95_T'].T
    S21_2nd_05_T = s21_data['S21_2nd_05_T'].T

    band1_nm = np.linspace(1240, 1380, 281)
    band2_nm = np.linspace(1390, 1490, 201)
    band3_nm = np.linspace(1500, 1600, 201)
    band4_nm = np.linspace(1610, 1700, 201)

    # 插值
    interp_func1 = interp1d(wl, S21_main_95_T)  # 定义一个插值函数。它根据给定的数据点生成一个连续的、可调用的函数，你可以用这个函数来计算任何在原始数据点范围内的点的插值。
    S21_band1_meas_T = interp_func1(band1_nm)
    S21_band2_meas_T = interp_func1(band2_nm)
    interp_func1 = interp1d(wl, S21_2nd_95_T)
    S21_band3_meas_T = interp_func1(band3_nm)
    S21_band4_meas_T = interp_func1(band4_nm)

    interp_func2 = interp1d(wl, S21_main_05_T)
    S21_band1_source_T = interp_func2(band1_nm)
    S21_band2_source_T = interp_func2(band2_nm)
    interp_func2 = interp1d(wl, S21_2nd_05_T)
    S21_band3_source_T = interp_func2(band3_nm)
    S21_band4_source_T = interp_func2(band4_nm)

    # PD_file_name = 'PD_current_%s.csv' % project_name
    # PD_path = join(getcwd(), 'PD', PD_file_name)


    # 检查PD_current文件是否存在，PD文件夹中最多只能有一个文件
    PD_mW_band1, PD_mW_band2, PD_mW_band3, PD_mW_band4, PD_source_mW_band1, PD_source_mW_band2, PD_source_mW_band3, PD_source_mW_band4 = PD_list
    if PD_mW_band1 is None:
        print("请检查文件是否正确")
        return
    t_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    rec_meas_band1 = recon_core_body_v2(len(band1_nm), PD_mW_band1, S21_band1_meas_T, alpha1_meas[0], 1)
    rec_meas_band2 = recon_core_body_v2(len(band2_nm), PD_mW_band2, S21_band2_meas_T, alpha1_meas[1], 1)
    rec_meas_band3 = recon_core_body_v2(len(band3_nm), PD_mW_band3, S21_band3_meas_T, alpha1_meas[2], 1)
    rec_meas_band4 = recon_core_body_v2(len(band4_nm), PD_mW_band4, S21_band4_meas_T, alpha1_meas[3], 1)

    rec_source_band1 = recon_core_body_v2(len(band1_nm), PD_source_mW_band1, S21_band1_source_T, alpha1_source[0],
                                            0)
    rec_source_band2 = recon_core_body_v2(len(band2_nm), PD_source_mW_band2, S21_band2_source_T, alpha1_source[1],
                                            0)
    rec_source_band3 = recon_core_body_v2(len(band3_nm), PD_source_mW_band3, S21_band3_source_T, alpha1_source[2],
                                            0)
    rec_source_band4 = recon_core_body_v2(len(band4_nm), PD_source_mW_band4, S21_band4_source_T, alpha1_source[3],
                                            0)

    Recon_band1 = rec_meas_band1
    Recon_band2 = rec_meas_band2
    Recon_band3 = rec_meas_band3
    Recon_band4 = rec_meas_band4

    return np.concatenate([Recon_band1, Recon_band2, Recon_band3, Recon_band4]), np.concatenate(
        [rec_source_band1, rec_source_band2, rec_source_band3, rec_source_band4])