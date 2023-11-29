'''绘图模块包含了绘制光谱吸收图的函数
---------
Functions:
---------
    - spectral_absorption(X,y,class="all_samples"): 画光谱吸收图
    - Sample_spectral_ranking(X,y,class="all_samples"): 画光谱吸收排序图
    - Numerical_distribution(ndarr,class="all_samples",feat_ = None): 画数值分布图
    - RankingMAE_and_Spearman_between_X_and_Y(X,y,class = "all_samples"): 画X和Y之间的排名MAE和spearman相关系数
    - train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,class = "all_samples"): 画训练集和测试集的散点图
---------

'''
import pandas as pd
wavelengths = pd.read_csv(r"C:\Users\zata\AppData\Local\Programs\Python\Python310\Lib\site-packages\nirapi\Alcohol.csv").columns[:1899].values.astype("float")

def spectral_absorption(X,y,category="all_samples",wavelengths = None):
    '''画光谱吸收图
    -------
    Parameters:
    ---------
        - X : ndarray 光谱数据
        - y : ndarray 物质含量
        - class :表示数据是属于谁的，默认是所有人的，如果是某个人的，就填写志愿者的名字
        - wavelengths : ndarray 波长
    ---------
    Returns:
    ---------
    Modify:
    -------
        - 2023-11-28 : 增加了Wavelengths参数，可以传入波长数据，如果不传入，就默认加载1899维的波长数据

    '''
    import numpy as np
    import pandas as pd 
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    
    # 读取数据
    absorbance = X
###### 2023-11-28 增加了Wavelengths参数，可以传入波长数据，如果不传入，就默认加载1899维的波长数据 begin
    if wavelengths is None: 
        wavelengths = pd.read_csv(r"C:\Users\zata\AppData\Local\Programs\Python\Python310\Lib\site-packages\nirapi\Alcohol.csv").columns[:1899].values
###############################3 end
    fig = go.Figure()
    for i in range(len(absorbance)):
        fig.add_trace(go.Scatter(x=wavelengths, y=absorbance[i,:],name=str(y[i])))
    fig.update_layout(title=category+" Spectra Absorbance Curve",
                        xaxis_title="Wavelength(nm)",
                        yaxis_title="Absorbance")
    fig.show()

def Sample_spectral_ranking(X,y,category="all_samples"):
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
    
    global wavelengths

    absorbance = X
    
    ranked = X.argsort(axis=0).argsort(axis=0)

    
    # Plotly plotting
    fig = go.Figure()
    for i in range(ranked.shape[0]):
        fig.add_trace(go.Scatter(x=wavelengths, y=ranked[i,:], name=str(y[i])))
    
    fig.update_xaxes(title_text="Wavelength")
    fig.update_yaxes(title_text="Rank")
    fig.update_layout(title=str(category)+" Spectra Absorbance Ranking Curve")
    fig.show()

def Numerical_distribution(ndarr,category="all_samples",feat_ = None):
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
    fig1 = px.bar(df, x='vals', y='counts', color='feats', barmode='group')
    fig1.show()

def RankingMAE_and_Spearman_between_X_and_Y(X,y,category = "all_samples",vertical_line:list = None,RankingMAE = True,Spearman = True):
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
    """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    global wavelengths

    
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


    if Spearman: # 画spearman
        coffs_spearman = []
        from scipy.stats import spearmanr
        for i in range(len(wavelengths)):
            coeff, p_value  = spearmanr(X[:,i], y)
            coffs_spearman.append(coeff)
        if RankingMAE:
            plt.subplot(1,2,2)
        plt.plot(wavelengths, coffs_spearman)
        plt.xlabel("Wavelengths")
        plt.ylabel("spearman"+"(abs_max:{},index:{})".format(round(max(np.abs(coffs_spearman)),3),np.argmax(np.abs(coffs_spearman))))
    # 11-28 增加了画垂直线的功能 begin
        if vertical_line is not None:  
            for i in vertical_line:
                from nirapi.load_data import get_wave_accroding_feat_index
                plt.vlines(get_wave_accroding_feat_index([i]),min(coffs_spearman),max(coffs_spearman),colors = "r",linestyles = "dashed")
            # plt.legend()
    # 11-28 增加了画垂直线的功能 end
        plt.title(category+"spearman Between Absorbance and y_value,samples_num:"+str(len(y_ranks)))


    plt.show()

def train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,category = "all_samples"):
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
    plt.show()