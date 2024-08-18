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
# matplotlib.use('TkAgg') 


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



import optuna
import nirapi.ML_model as AF
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import random
def run_optuna(X,y,isReg,chose_n_trails,selected_metric = 'r',save=None,save_name= "",**kw):
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
        kw: example: {"selected_outlier":["不做异常值去除","mahalanobis"] ,"selected_data_split":["random_split"],"selected_preprocess":["不做预处理","mean_centering","snv"],"selected_feat_sec":["不做特征选择"]}   --  Fill in the model dictionary included in the hyperparameter tuning
    """

    import random



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
    else:
        # 在选择分类任务时，提供分类模型的选择
        model_options = ['LogisticRegression','SVM','DT','RandomForest','KNN','Bayes(贝叶斯分类)',"GradientBoostingTree","XGBoost"]

    temp_list_to_database = {}
    y_data_to_csv = {}

    ####################################################################################################################################################
    ##################################                     设置调参使用哪些参数               begin     #################################################
    ####################################################################################################################################################
    ## 选择异常值去除的选项
    selected_outlier = ["mahalanobis"]
    if "selected_outlier" in kw.keys():
        selected_outlier = kw["selected_outlier"]
    # 选择数据拆分的选项
    selected_data_split = [ "custom_train_test_split"]
    if "selected_data_split" in kw.keys():
        selected_data_split = kw["selected_data_split"]
    data_split_ratio  = 0.3

    #选择预处理的选项
    selected_preprocess = ["不做预处理", "mean_centering", "normalization", "standardization", "poly_detrend", "snv", "savgol", "msc","d1", "d2", "rnv", "move_avg"]
    if "selected_preprocess" in kw.keys():
        selected_preprocess = kw["selected_preprocess"]
    preprocess_number_input = 2


    #选择特征选择的选项")
    selected_feat_sec = ["不做特征选择"]
    if "selected_feat_sec" in kw.keys():
        selected_feat_sec = kw["selected_feat_sec"]
    #选择降维的选项")
    selected_dim_red = ["不做降维"]
    if "selected_dim_red" in kw.keys():
        selected_dim_red = kw["selected_dim_red"]
    #选择参与调参的模型")
    selected_model = model_options
    if "selected_model" in kw.keys():
        selected_model = kw["selected_model"]
    #选择以那种指标进行调参")
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
                'spa': [AF.spa, {'i_init': trial.suggest_int('spa_i_init', 0, X.shape[1]-1),
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
                'Bayes(贝叶斯回归)': [AF.bayes, {'n_iter': trial.suggest_int('Bayes(贝叶斯回归)_n_iter', 1, 100),
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
                    # 'max_depth': trial.suggest_int('RandomForest_max_depth', 1, 100),
                    'min_samples_split': trial.suggest_int('RandomForest_min_samples_split', 1, 100),
                    'min_samples_leaf': trial.suggest_int('RandomForest_min_samples_leaf', 1, 100),
                    # 'min_weight_fraction_leaf': trial.suggest_float('RandomForest_min_weight_fraction_leaf', 0.0001, 0.1),
                    # 'max_features': trial.suggest_float('RandomForest_max_features', 0.1,1.0),
                    # 'max_leaf_nodes': trial.suggest_int('RandomForest_max_leaf_nodes', 1, 100),
                    # 'min_impurity_decrease': trial.suggest_float('RandomForest_min_impurity_decrease', 0.0001, 0.1),
                    # 'bootstrap': trial.suggest_categorical('RandomForest_bootstrap', [True, False]),
                    # 'oob_score': trial.suggest_categorical('RandomForest_oob_score', [True, False]),
                    # 'n_jobs': trial.suggest_int('RandomForest_n_jobs', 1, 100),
                    'random_state': trial.suggest_int('RandomForest_random_state', 1, 100),
                    # 'verbose': trial.suggest_int('RandomForest_verbose', 0, 100),
                    # 'warm_start': trial.suggest_categorical('RandomForest_warm_start', [True, False]),
                    # 'ccp_alpha': trial.suggest_float('RandomForest_ccp_alpha', 0.0001, 0.1),
                    # 'max_samples': (trial.suggest_float('RandomForest_max_samples', 0.1, 1.0) if trial.params['RandomForest_bootstrap'] else None)
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
                    # 'n_jobs': trial.suggest_int('KNN_n_jobs', 1, 100),
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
        for func_info in selection_summary_5_tab2[3]:
            X_train, X_test,y_train,y_test= functions_["Preprocess"][func_info][0](X_train, X_test,y_train,y_test,
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
            print(e)
            # if isReg:
            #     return 100000000
            # else:
            #     return 0

    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    # optuna.logging.enable_default_handler()
    if selected_metric in ["accuracy", "precision", "recall",'r2','r']:
        direction ='maximize'
    else:
        # 否则，默认为最小化目标，例如MAE、MSE或R2
        direction = 'minimize'
    study = optuna.create_study(direction=direction)
    study.optimize(objective,n_trials=chose_n_trails)
    print(str(temp_list_to_database[study.best_trial.number]))

    
    train_test_scatter(study.best_trial.user_attrs['y_train'],study.best_trial.user_attrs['y_train_pred'],study.best_trial.user_attrs['y_test'],study.best_trial.user_attrs['y_pred'],category=save_name,save = save)
    
            
    
    # if selected_metric in reg_metric:
    #     st.write(f"最优{selected_metric}",max_result)
    #     st.write('reg_metric: ', best_trial_mertic )
    #     # st.write("具体参数:  ",str(temp_list_to_database[best_trial_num]))
    #     st.write("具体参数： ", temp_list_to_database[best_trial_num])

    # else:
    #     st.write(f"最优{selected_metric}",max_result)
    #     st.write('["accuracy", "precision", "recall"]: ', best_trial_mertic )
    #     st.write("具体参数:  ",str(temp_list_to_database[best_trial_num]))
    #     st.write("具体参数： ", temp_list_to_database[best_trial_num])

    # if True:
    #     st.session_state.model_param_by_optuna = temp_list_to_database[best_trial_num]

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



from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

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



def run_regression_optuna_v3(data_name,X = None,y=None ,data_splited = None, model='PLS',split = 'SPXY',test_size = 0.3, n_trials=200,object = None,cv = None,save_dir = None,each_class_mae=False,only_train_and_val_set=False):

    # 新增功能 data_splited 输入训练集验证集和测试集

    '''
    -----
    params:
    -----
        data_name: 数据集名称
        X: 特征数据
        y: 标签数据
        data_splited: 划分好训练验证测试数据的字典，{'X_train':X_train,'X_val':X_val,'X_test':X_test,'y_train':y_train,'y_val':y_val,'y_test':y_test}
        model: 选择模型，{"PLS","SVR","RFreg","LR"}
        split: 选择数据集划分方式，{"SPXY","Random"}
        n_trials: 选择优化次数
        object: 选择优化目标，{"R2","MAE","RMSE","Pearsonr"}
        cv: 交叉验证方式,填整数就是k折交叉验证，填None不做交叉验证
        save_dir: 保存结果的名称
        each_class_mae: 是否计算每个类别的mae,画再散点图上
        
    -----
    return:

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
            X_train,X_val,y_train,y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
        else:
            assert False, "split_name error"
    elif X is not None and y is not None and only_train_and_val_set == True:
        if split == 'SPXY':
            X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size, 'SPXY')
            X_val, y_val = X_test,y_test
        elif split == "Random":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            X_val, y_val = X_test,y_test
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
                    assert False, "object_name error"
                
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










from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

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








        
if __name__ == "__main__":
    ##### save_data_to_csv   test_case ##### 
    # mycsv = save_data_to_csv(r"D:\Desktop\AllTheCodesYouNeed\useful\NIR\code&note\1B-20240206重建光谱的预处理\data\test.csv",columns=['test'])
    # mycsv.put(["test"])
    pass


        
    
    

    


























# """
# --------
# load_data
# ---------
#     data_name 



# model_train_and_eval
#     模型训练和评估，
#     参数：
#         model_name,   { "LR","RFreg","SVR","FS_eval_LR_model"}
#         X_train,
#         X_test,
#         y_train,
#         y_test,
#         pred_data = False,
#         **kwargs

#                 model_name = "LR" 时，参数：**kwargs 里面包含:
#                 model_name = "RFreg" 时，参数：**kwargs 里面包含: n_estimators:{int，可选,默认4}
#                 model_name = "SVR" 时，参数：**kwargs 里面包含: C:{float，可选,默认1.0}，epsilon:{float，可选,默认0.1}，kernel:{str，可选,默认'rbf'}，gamma:{str，可选,默认'auto'}
#                 model_name = "FS_eval_LR_model" 用于特征选择时的评价，固定model是线性模型，会画两个图，并保存在指定路径，参数：**kwargs 里面包含: importance:{list，重要性列表}，title:{str，可选,默认None}，save_path:{str，可选,默认None},n_features:{int，可选,len(importance)//2}
                


# precessing_data
#     数据预处理
#     参数：
#         pre_name=None,   {"SG","SNV","MSC","LN","delete_by_y_value"}
#         X=None,
#         y=None,
#         **kwargs

#                 pre_name = "SG" 时，参数：**kwargs 里面包含: window_length:{int，可选,默认14}，polyorder:{int，可选,默认1}，deriv:{int，可选,默认0}
#                 pre_name = "SNV" 时，参数：**kwargs 里面包含:
#                 pre_name = "MSC" 时，参数：**kwargs 里面包含:
#                 pre_name = "LN" 时，线性归一化，返回X，参数：**kwargs 里面包含:
#                 pre_name = "delete_by_y_value" 时，参数：**kwargs 里面包含: min_y:{int，可选,默认None}，max_y:{int，可选,默认None}
# features_selection
#     特征选择
#     参数：
#         fs_name,    {"RF"}
#         X,
#         y,
#         **kw

# tarin
#     训练
#     参数：
#         group,  {"SG+RF+LR"}
#         X,
#         y,

# feature_import_draw
#     画出特征重要性图
#     参数：
#         draw_name,  {"absorb_and_importance","importance"}
#         input,
#         figsize = (20,10),
#         title=None,
#         **kwargs

#                 draw_name = "absorb_and_importance" 时，画吸收率和重要性在一张图上，参数：**kwargs 里面包含:save_path:{str，可选,默认None,表示不保存}
#                 draw_name = "importance" 时，只画重要性图，参数：**kwargs 里面包含:wave:{list,波长列表，相当于X轴坐标},nameList:{strList，可选，如果是单个志愿者分析的时候最好有,默认为range(len(input))}，save_path:{str，可选,默认为None}，legend:{bool，可选,默认为False}
# """
# # 判断是否安装成功
# def hello_world():
#     print("hello_world")





# def model_train_and_eval(model_name,X_train,X_test,y_train,y_test,pred_data = False,**kwargs):
#     """
#     param:
#         kwargs：是参数字典至少要包含五个参数：
#             model_name:["LR"]
#             X_train: 训练数据
#             y_train:
#             X_test:
#             y_test: 
#     """
#     import numpy as np
#     from api.ML_in_NIR.ML_eval import ML_eval,ML_eval_no_model_arg

# ##LR#######################################################################
#     if model_name == "LR":
#         from api.ML_in_NIR.OLS_or_LinearRegression import LR
#         LR = LR(X_train,y_train)
#         eval =ML_eval(model=LR.fitting(),test_data=(np.column_stack((X_test,y_test))))
#         NI = eval.numerical_index()
#         y_train_ans = LR.fitting().predict(X_train)
#         y_test_ans = LR.fitting().predict(X_test)
#         if pred_data:
#             # example: pred_data = True
#             return (NI),y_train_ans,y_test_ans
#         return NI
    
# ##RFreg#######################################################################
#     if model_name == "RFreg":
#         from sklearn.ensemble import RandomForestRegressor
#         RFreg = RandomForestRegressor(**kwargs)
#         RFreg.fit(X_train,y_train)
#         y_pred = RFreg.predict(X_test)
#         eval = ML_eval_no_model_arg(X_test,y_test,y_pred)
#         NI = eval.numerical_index()
#         return NI

# ##SVR######################################################################
#     if model_name == "SVR":
#         from sklearn.svm import SVR
#         regr = SVR(**kwargs)
#         regr.fit(X_train, y_train)
#         y_pred = regr.predict(X_test)
#         eval =ML_eval_no_model_arg(X_test=X_test,y_test=y_test,y_pred=y_pred)
#         NI = eval.numerical_index()
#         y_train_ans = regr.predict(X_train)
#         y_test_ans = regr.predict(X_test)
#         if pred_data:
#             # example: pred_data = True
#             return (NI),y_train_ans,y_test_ans
#         return NI
# ## FS_eval_LR_model ######################################################################
#     if model_name == "FS_eval_LR_model":
#         # kwargs
#         """
#         importance:
#         title:optional
#         save_path:optional        
#         """
#         import json
#         with open("wave.json",'r') as f:
#             wavelegths = json.load(f)

#         # kwargs 
#         importance = kwargs.get("importance")
#         title = kwargs.get("title")
#         n_features = kwargs.get('n_features',len(importance)//2)

#         save_path = kwargs.get("save_path") # example: save_path = "D:\test\
#         from datetime import datetime
#         now = datetime.now()
#         now_time_str = now.strftime("%Y-%m-%d_%H_%M_%S")
#         feature_import_draw(draw_name='importance', input=importance,figsize=(20,10),title=title,wave = wavelegths, save_path=save_path +"_importance_" +now_time_str+".png")
#         feature_import_draw(input=importance,title=title,save_path=save_path +title +now_time_str+".png")
#         # 画特征选择的散点图，看哪些特征被选择了
#         def features_selection_scatter():
#             import matplotlib.pyplot as plt
#             wavelegths_ = np.array(wavelegths)
#             plt.figure(figsize=(20,10))
#             plt.title(title+"features_selection_scatter")
#             plt.plot(wavelegths,importance)
#             plt.scatter(wavelegths_[importance.argsort()[::-1][:n_features]],importance[importance.argsort()[::-1][:n_features]],s=100,c='r',marker='*')
#             plt.show()
#         features_selection_scatter()
#         indices = importance.argsort()[::-1][:n_features]
#         X_train_reduced = X_train[:,indices]
#         X_test_reduced = X_test[:,indices]
#         from sklearn.linear_model import LinearRegression
#         linreg = LinearRegression()
#         linreg.fit(X_train_reduced, y_train)
#         print('R-squared:', linreg.score(X_test_reduced, y_test))
#         y_train_ans = linreg.predict(X_train_reduced)
#         y_test_ans = linreg.predict(X_test_reduced)
#         draw(draw_name="train_test_scatter",y_train = y_train,y_test=y_test,y_train_ans=y_train_ans,y_test_ans=y_test_ans,title = title,save_path = save_path +"_train_test_scatter_" +now_time_str+".png")
        




# def draw(draw_name = None,**kwargs):
#     if draw_name == "curve":
#         from api.draw.curve import draw_curve
#         draw = draw_curve(**kwargs) 
#         draw.draw()
#     if draw_name == "train_test_scatter":
#         from api.draw.scatter import scatter
#         scatter().train_test_scatter(**kwargs) # kwargs contain {y_train,y_test,y_train_ans,y_test_ans}

        

        
    

    

    

# def load_data(data_name=None,**kwargs):
#     '''这个函数用来加载数据
#     -------
#     Parameters:
#     ---------
#         - data_name : str
#             {11_7_draw,alcohol_1111 | get_volunteer_name }
#         - kwargs : dict
#             {file_path,name | file_path}
#     ---------
#     Returns:
#     ---------
#         - X : ndarray
#             NIR spectral data
#         - y : ndarray
#             alcohol content
#     '''

#     if data_name == "alcohol_1111":
#         def alcohol_1111(file_path,name=None):
#             '''加载11月11日的酒精数据 Loading Alcohol data for Nov. 11 
#             -------
#             Parameters:
#             ---------
#                 - file_path : str 
#                 - name : str  
#                     {voluntee name}
#             ---------
#             Returns:
#             ---------
#                 - X : ndarray
#                     NIR spectral data
#                 - y : ndarray
#                     alcohol content
            
#             '''
#             import numpy as np
#             import pandas as pd
#             data = pd.read_csv(file_path).values
#             if name is None:
#                 X = data[:,:1899]
#                 y = data[:,1899].reshape(-1,1)
#             else:
#                 index_row = data[:,1902] == name
#                 X = data[index_row,:1899]
#                 y = data[index_row,1899].reshape(-1,1)
#             return X,y
#         return alcohol_1111(**kwargs)
    
#     if data_name == "get_volunteer_name":
#         def get_volunteer_name(file_path):
#             '''获取志愿者的名字
#             -------
#             Parameters:
#             ---------
#                 - file_path : str 
#             ---------
#             Returns:
#             ---------
#                 - name : list
#                     volunteer name
#             '''
#             import pandas as pd
#             import numpy as np
#             data = pd.read_csv(file_path).values
#             name = data[:,1902]
#             name = np.unique(name)
#             return name
#         return get_volunteer_name(file_path=kwargs.get("file_path"))
            


    

# def precessing_data(pre_name = None,X = None,y=None,**kwargs):

#     assert pre_name is not None , "pre_name = None"
#     assert X is not None , "X = none"
# ##SG#### 多项式平滑处理 ##########################################################################
#     if pre_name == "SG":
#         '''
#         param:
#             "window_length" : 窗口大小
#             "polyorder" : 多项式项数
#             "deriv" : 求解迭代次数
#         '''
#         from api.preprocessing.SG import SG
#         if not kwargs.get("window_length"):
#             print("没有输入window_length 默认14")
#         if not kwargs.get("polyorder") :
#              print("polyorder 默认1")
#         if not kwargs.get("deriv"):
#              print("deriv 默认0")

#         window_length = kwargs.get("window_length",14)
#         polyorder = kwargs.get("polyorder",1)
#         deriv = kwargs.get("deriv",0)
#         sg_X = SG(X).fit(window_length=window_length, polyorder=polyorder, deriv=deriv)
#         return sg_X

# ##SNV#### SNV处理 ############################################################################
#     if pre_name == "SNV":  
#         from api.preprocessing.SNV import SNV
#         snv = SNV(absorbances=X)
#         return snv.getdata()
    
# ##MSC#### 均值中心化处理  #####################################################################
#     if pre_name == "MSC":
#         from api.preprocessing.MSC import MSC
#         msc_data = MSC(X).get_data()
#         return msc_data
    
# ##Linear normalization## 线性归一化  #########################################################
#     if pre_name =='LN':
#         import numpy as np
#         X = (X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))
        
#         return X
    
# ##delete data by y_value #  把y值在一定范围内的数据删除  ######################################
#     if pre_name == "delete_by_y_value":
#         # 大于min_y的数据删除，小于max_y的数据删除
#         import numpy as np
#         min_y = kwargs.get("min_y")
#         max_y = kwargs.get("max_y")
#         if not kwargs.get("min_y"):
#             print("没有输入min_y")
#             delete_index = np.where((y<max_y))
#         elif not kwargs.get("max_y"):
#             print("没有输入max_y")
#             delete_index = np.where((y>min_y))
#         else:
#             delete_index = np.where((y>min_y) & (y<max_y))


#         X = np.delete(X,delete_index,axis=0)
#         y = np.delete(y,delete_index,axis=0)
#         return X,y

# ##RNV ######################################################################################
#     if pre_name == "RNV":
#         import numpy as np
#         from typing import Union,List
#         def RNV(X: Union[np.ndarray,List],percent=25) -> np.ndarray:
#             """RNV preprocessing method for NIR spectral analysis(robust normal variate)
#                 RNV : z = [x - precentile(x)]/std[x <= percentile(x)]
#                 .. where precentile(x) is the percentile in dataset x,
#                   which defaults to the 25th percentile according to the paper's prompts,
#                     but may be set to 10 depending on the situation.
#             -------
            
#             Parameters
#             ----------
#             - X : ndarray or list
#                 Spectral absorbance data, generally two-dimensional,
#                 you can also enter one-dimensional, other dimensions will report errors. 
#             - percent : int from 0-100 ,default = 25
#                 see np.percentile for detial.

#             Returns
#             ---------
#             - X_RNV : ndarray
#                 NIR spectral data arfter RNV processing

#             Examples
#             -------
#             >>> X = np.array([[1,2,3,4,5,6],[3,9,9,100,9,9]])
#             >>> RNV(X,percent=50)
#             array([[-3, -1,  0,  0,  1,  3],
#                     [-2,  0,  0, 37,  0,  0]])
#             """
#             assert isinstance(X,np.ndarray) or isinstance(X,list),"Variable X is of wrong type, must be ndarray or list"
#             if isinstance(X,list):
#                 X = np.array(X)
#             X_RNV = np.zeros_like(X)
#             if X.ndim == 2:
#                 for i in range(X.shape[0]):
#                     percentile_value = np.percentile(X[i],percent,method="median_unbiased")
#                     X_RNV[i] = (X[i]-percentile_value)/np.std(X[i][X[i]<=percentile_value])
#             elif X.ndim == 1:
#                 percentile_value = np.percentile(X,percent,method="median_unbiased")
#                 X_RNV = (X-percentile_value)/np.std(X[X<=percentile_value])
#             else :
#                 assert False,"Variable X dimension error"
#             return X_RNV
#         RNV_precent = kwargs.get("precent",25)
#         return RNV(X,RNV_precent)

            
        





# def features_selection(fs_name="RF",X=None,y=None,**kw):
    
#     if fs_name == "RF":
#         if type(y) == None:
#             assert "缺少数据y"
#         min_num = min(y)
#         max_num = max(y)
#         newy = (y-min_num)//((max_num-min_num)/2)
#         from api.features_selection.random_forest import random_forest
#         if not kw.get('n_estimators'):
#             print("没有输入n_estimators，默认4")
#         newX = random_forest(X,newy,n_estimators=kw.get("n_estimators",4)).fitting()
#         return newX




# def tarin(group = None,X = None,y=None):
#     from sklearn.model_selection import train_test_split
#     if type(X)==None or type(y) ==None:
#         assert("缺少数据")
#     if not group:
#         X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.5,random_state=0)
#         return model_train_and_eval(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_name= "LR")
    

# ##SG+RF+LR######################################################################################################
#     if group == "SG+RF+LR":
#         sg_X = precessing_data({
#             "pre_name":"SG",
#             "X":X
#         })
#         newX = features_selection(fs_name="RF",X=sg_X,y=y,kw ={"n_estimators":4})
#         X_train,X_test,y_train,y_test = train_test_split(newX,y,train_size=0.5,random_state=0)
#         return model_train_and_eval(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_name="LR")
        

# def feature_import_draw(draw_name= "absorb_and_importance",input = None,figsize = (20,10),title='features_importance',**kwargs):


# ## absorb_and_importance #######################################################################################################################
# # 吸收率和重要性在一张图上
#     if draw_name == "absorb_and_importance":
#         import numpy as np
#         import matplotlib.pyplot as plt
#         wave,abs = load_data("11_7_draw")  # 加载光波数据和吸收率数据
        

#         # 线性归一化
#         abs  =  (abs - np.min(abs)) / (np.max(abs) - np.min(abs))  #吸收率数据归一化
#         input = (input - np.min(input)) / (np.max(input) - np.min(input))  #外部输入的重要性数据归一化

#         # 光谱数据和外部输入的重要性数据合并可视化 
#         abs = np.row_stack([abs,input]) # 合并
        
#         plt.figure(figsize = figsize)
#         plt.title(title)
#         # 解决中文显示问题
#         plt.rcParams['font.sans-serif'] = ['SimHei']
#         plt.rcParams['axes.unicode_minus'] = False

#         for i in range(len(abs)):
#         # 画出每个吸光度曲线
#             absorbance = abs[i] # 吸光度
#             plt.plot(wave,absorbance) # 绘制曲线
#         plt.xlabel('Wavelength (nm)') # x轴标题
#         # plt.ylabel('Absorbance') # y轴标题

#         # if save_path:
#         save_path = kwargs.get("save_path")
#         if save_path:
#             plt.savefig(save_path)

#         plt.show() # 显示图像
#         return 
    
# ## importance #######################################################################################################################
# # 只画重要性图
#     if draw_name == "importance":
#         import matplotlib.pyplot as plt
#         # wave,_ = load_data("11_7_draw")  # 加载光波数据和吸收率数据
#         wave = kwargs.get("wave")

#         if input is None:
#             print("没有输入数据")
#             return
        
#         # set title
#         if not figsize:
#             print("没有输入figsize")
#             plt.figure(figsize=(20,10))
#         else:
#             plt.figure(figsize=figsize)

#         # set title
#         if not title:
#             print("没有输入title")
#             plt.title("importance")
#         else:
#             plt.title(title)

#         # 解决中文显示问题
#         plt.rcParams['font.sans-serif'] = ['SimHei']
#         plt.rcParams['axes.unicode_minus'] = False

#         # set nameList
#         if kwargs.get("nameList") :
#             nameList = kwargs.get("nameList")
#         else:
#             if  input.ndim == 2:
#                 nameList = [i for i in range(len(input))]

#         # plot
#         if input.ndim == 2:
#             for i in range(len(input)):
#                 if wave is None:
#                     plt.plot(input[i],label = list(nameList)[i])
#                 else:
#                     plt.plot(wave,input[i],label = list(nameList)[i])
#         if input.ndim == 1:
#             if wave is None:
#                 plt.plot(input)
#             else:
#                 plt.plot(wave,input)
#         else:
#             print("输入数据维度不对")
        
#         # set legend
#         if kwargs.get("legend") == True:
#             plt.legend()

#         # 如果需要保存图片
#         if kwargs.get("save_path"):
#             plt.savefig(kwargs.get("save_path"))

#         plt.show()

        


# if __name__ == "__main__":
#     wave,abs = load_data("11_7_draw")
#     print(abs)

#     draw(wavelengths=wave,absorb=abs,draw_name="curve")
