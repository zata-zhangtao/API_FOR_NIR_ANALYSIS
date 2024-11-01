'''这个文件中的函数用来进行数据分析
--------
Functions:
----------
    - plot_correlation_graph(X,y): 画spearman 和 pearson相关系数图
    - plot_mean(data_name,X_name,y_name, data_X, data_y, scale=True,draw_y=True, save_dir=None) # 画平均值图

'''
import numpy as np
import plotly.graph_objs as go
from scipy.stats import pearsonr, spearmanr


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