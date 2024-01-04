from typing import Union
'''绘图模块包含了绘制光谱吸收图的函数
---------
Functions:
---------
    - spectral_absorption(X,y,class="all_samples"): 画光谱吸收图
    - Sample_spectral_ranking(X,y,class="all_samples"): 画光谱吸收排序图
    - Numerical_distribution(ndarr,class="all_samples",feat_ = None): 画数值分布图
    - RankingMAE_and_Spearman_between_X_and_Y(X,y,class = "all_samples"): 画X和Y之间的排名MAE和spearman相关系数
    - train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,class = "all_samples"): 画训练集和测试集的散点图
    - Comparison_of_data_before_and_after(before_X,before_y,after_X,after_y,class = "all_samples"): 画数据处理前后的spearman图
    -     
---------

'''
import pandas as pd
# wavelengths = pd.read_csv(r"C:\Users\zata\AppData\Local\Programs\Python\Python310\Lib\site-packages\nirapi\Alcohol.csv").columns[:1899].values.astype("float")
wavelengths = pd.read_csv(r"C:\Users\zata\AppData\Local\Programs\Python\Python310\Lib\site-packages\nirapi\Alcohol.csv").columns[:1899].values.astype("float")
def spectral_absorption(X,y,category="all_samples",wave = None,plt_show = False):
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
    if wave is None: 
        global wavelengths
    else:
        wavelengths = wave
###############################3 end
        

    ####begin 是否用matplotlib画图 modify 2023-12-14
    if plt_show:
        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # matplotlib plotting
        plt.figure(figsize=(15,5))
        for i in range(len(absorbance)):
            plt.plot(wavelengths,absorbance[i,:],label=str(y[i]))
        plt.xlabel('Wavelength')
        plt.ylabel('Absorbance')
        plt.legend()
        plt.title(category+' Spectra Absorbance Curve')
        plt.show()
        return
    ####end 是否用matplotlib画图 modify 2023-12-14




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
                                            draw = True
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
