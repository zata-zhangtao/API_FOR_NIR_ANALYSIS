from typing import Union
'''绘图模块包含了绘制光谱吸收图的函数
---------
Functions:
---------
    - spectral_absorption(X,y,class="all_samples",wave = None,plt_show = False): 画光谱吸收图
    - Sample_spectral_ranking(X,y,class="all_samples"): 画光谱吸收排序图
    - Numerical_distribution(ndarr,class="all_samples",feat_ = None): 画数值分布图
    - RankingMAE_and_Spearman_between_X_and_Y(X,y,class = "all_samples"): 画X和Y之间的排名MAE和spearman相关系数
    - Comparison_of_data_before_and_after(before_X,before_y,after_X,after_y,class = "all_samples"): 画数据处理前后的spearman图
    - train_test_scatter(y_train,y_train_pred,y_test,y_test_pred,class = "all_samples"): 画训练集和测试集的散点图
    - classification_report_plot(y_true,y_pred): 分类任务中画分类报告图
    - train_and_test_pred_plot(y_train,y_test,y_pred_train,y_pred_test,data_name): 画回归任务中的散点图，用plotly画
    - train_val_and_test_pred_plot(y_train,y_val,y_test,y_pred_train,y_pred_val,y_pred_test, data_name="",save_dir=None,each_class_mae=True,content = "")  # 画训练验证测试集
    - pred_plot(y_test,y_pred_test, data_name="",save_dir=None,each_class_mae=True,content = "") # 画真实值和预测值的散点图
    - plot_multiclass_line(data_name,X,y,save_dir=None): 根据Y的值设置不同颜色画折线图
    - plot_correlation_graph(X,y): 画spearman 和 pearson相关系数图
    - plot_mean(data_name,X_name,y_name, data_X, data_y, scale=True,draw_y=True, save_dir=None) # 画平均值图

---------
Example:
---------
    
    

---------

'''
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
# wavelengths = pd.read_csv(r"C:\Users\zata\AppData\Local\Programs\Python\Python310\Lib\site-packages\nirapi\Alcohol.csv").columns[:1899].values.astype("float")
# wavelengths = pd.read_csv(r"C:\Users\zata\A ppData\Local\Programs\Python\Python310\Lib\site-packages\nirapi\Alcohol.csv").columns[:1899].values.astype("float")
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
        wavelengths = [i for i in range(X.shape[1])]
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
            plt.plot(None,absorbance[i,:],label=str(y[i]))
        plt.xlabel('Wavelength')
        plt.ylabel('Absorbance')
        plt.legend()
        plt.title(category+' Spectra Absorbance Curve')
        plt.show()
        return
    ####end 是否用matplotlib画图 modify 2023-12-14




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
