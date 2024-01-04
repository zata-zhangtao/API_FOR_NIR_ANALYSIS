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
"""
from typing import Union


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
