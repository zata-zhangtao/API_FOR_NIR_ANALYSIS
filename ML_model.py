############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
# 首先是用于自动调参的代码
############################################################################################################################################################################################
# 为了方便自动调参，将一些常用的函数封装起来，方便调用，要求有统一的输入输出格式
import copy
import numpy as np
import pandas as pd
# import streamlit as st
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from obspy.signal.detrend import polynomial
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from scipy.signal import savgol_filter
# from pybaselines.whittaker import iarpls, airpls, derpsalsa
# 不做任何处理
def return_inputs(*args):
    return args





# 前处理   要求输入为 X, y, **params  输出为 X_new, y_new
def mahalanobis(X, y,threshold=95):
    """
    -----
    param
    -----
    X: array-like, shape (n_samples, n_features)
    y: array-like, shape (n_samples,)
    threshold: int, default 95
    -----
    return
    -----
    mahal_x: array-like, shape (n_samples, n_features)
    y: array-like, shape (n_samples,)
    -----
    """

    mahal_X = np.asarray(X)
    x_mu = mahal_X - np.mean(mahal_X, axis=0)
    cov = np.cov(mahal_X.T)
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left_term, x_mu.T)
    d = mahal.diagonal()
    threshold = np.percentile(d, threshold)
    mahal_x = mahal_X[d < threshold]

    return mahal_x, y[d < threshold]



# 数据集划分  要求输入为 X, y, **params  输出为 X_train, X_test, y_train, y_test
def random_split(x, y, test_size=0.3, random_seed=42):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_seed)

    return x_train, x_test, y_train, y_test

def custom_train_test_split(x, y, test_size, method = 'KS'):
    if method == 'KS':
        distance = cdist(x, x)

    if method == 'SPXY':
        y = np.expand_dims(y, axis=-1)

        distance_x = cdist(x, x)
        print(y.shape)
        distance_y = cdist(y, y)

        distance_x /= distance_x.max()
        distance_y /= distance_y.max()

        distance = distance_x + distance_y
    if method == 'ManualDivision':
        X_train = x[:int(len(x)*test_size)]
        y_train = y[:int(len(y)*test_size)]
        X_test = x[int(len(x)*test_size):]
        y_test = y[int(len(y)*test_size):]
        return X_train, X_test, y_train, y_test
        

        
        

    def max_min_distance_split(distance, train_size):
        i_train = []
        i_test = [i for i in range(distance.shape[0])]

        first_2_points = np.unravel_index(np.argmax(distance), distance.shape)

        i_train.append(first_2_points[0])
        i_train.append(first_2_points[1])

        i_test.remove(first_2_points[0])
        i_test.remove(first_2_points[1])

        for _ in range(train_size - 2):
            max_min_dist_idx = np.argmax(np.min(distance[i_train, :], axis=0))

            i_train.append(max_min_dist_idx)
            i_test.remove(max_min_dist_idx)

        return i_train, i_test

    if 0 < test_size < 1:
        test_size = int(x.shape[0] * test_size)
    index_train, index_test = max_min_distance_split(distance, x.shape[0] - test_size)
    x_train, x_test, y_train, y_test = x[index_train], x[index_test], y[index_train], y[index_test]

    return x_train, x_test, y_train.reshape(-1,), y_test.reshape(-1,)



# 预处理  要求输入为X_train, X_test, y_train, y_test, **params  输出为 X_train, X_test, y_train, y_test
def mean_centering(X_train, X_test, y_train, y_test, axis=None):
    """
    return the mean-centering of the 2D array x.

    :param x: shape (n_samples, n_features)
    :param axis == None: element-wise mean-centering
           axis == 0   :  column-wise mean-centering
           axis == 1   :     row-wise mean-centering
    """

    if axis not in [None, 0, 1]:
        raise ValueError('Unexpected axis value.')

    if axis == None:
        x_mean = np.mean(X_train, axis=axis)
        return X_train - x_mean, X_test - x_mean, y_train, y_test
    if axis == 0:
        x_mean = np.expand_dims(np.mean(X_train, axis=axis), axis=axis)
        return X_train - x_mean, X_test - x_mean, y_train, y_test
    if axis == 1:
        x_train_mean = np.expand_dims(np.mean(X_train, axis=axis), axis=axis)
        x_test_mean = np.expand_dims(np.mean(X_test, axis=axis), axis=axis)
        return X_train-x_train_mean , X_test-x_test_mean , y_train,y_test



def normalization(X_train, X_test, y_train, y_test,  axis=None, min=0, max=1):
    """
    return the normalization of the 2D array X such that each element is in [0, 1]
    scaling each feature to a given range,
    The transformation is given by:
    x_std = (x - x.min(axis)) / (x.max(axis) - x.min(axis))
    x_scaled = x_std * (max - min) + min

    :param x: shape (n_samples, n_features)
    :param axis == None: element-wise normalization
           axis == 0   :  column-wise normalization
           axis == 1   :     row-wise normalization
    """

    if axis not in [None, 0, 1]:
        raise ValueError('Unexpected axis value.')

    if axis == None:
        x_min = np.min(X_train, axis=axis)
        x_max = np.max(X_train, axis=axis)
        return (X_train - x_min) / (x_max - x_min) * (max - min) + min, (X_test - x_min) / (x_max - x_min) * (max - min) + min, y_train, y_test


    if axis != None:
        if axis == 0:

            x_min = np.expand_dims(np.min(X_train, axis=axis),axis=axis)
            x_max = np.expand_dims(np.max(X_train, axis=axis),axis=axis)
            return (X_train - x_min) / (x_max - x_min) * (max - min) + min, (X_test - x_min) / (x_max - x_min) * (max - min) + min, y_train, y_test
        if axis == 1:
            x_train_min = np.expand_dims(np.min(X_train, axis=axis),axis=axis)
            x_train_max = np.expand_dims(np.max(X_train, axis=axis),axis=axis)
            x_test_min = np.expand_dims(np.min(X_test, axis=axis),axis=axis)
            x_test_max = np.expand_dims(np.max(X_test, axis=axis),axis=axis)
            return (X_train - x_train_min) / (x_train_max - x_train_min) * (max - min) + min, (X_test - x_test_min) / (x_test_max - x_test_min) * (max - min) + min, y_train, y_test


def standardization(X_train, X_test, y_train, y_test, axis=None):
    """
    return the standardization of the 2D array X such that
    each element is subtracted the mean and divided by the standard deviation.

    :param x: shape (n_samples, n_features)
    :param axis == None: element-wise standardization
           axis == 0   :  column-wise standardization
           axis == 1   :     row-wise standardization
    """

    if axis not in [None, 0, 1]:
        raise ValueError('Unexpected axis value.')

    x_mean = np.mean(X_train, axis=axis)
    x_std = np.std(X_train, axis=axis)

    if axis == None:
        return (X_train - x_mean) / x_std, (X_test - x_mean) / x_std, y_train, y_test
    if axis != None:
        if axis == 0:
            x_mean = np.expand_dims(x_mean, axis=axis)
            x_std = np.expand_dims(x_std, axis=axis)
            return (X_train - x_mean) / x_std, (X_test - x_mean) / x_std, y_train, y_test
        if axis == 1:
            x_train_mean = np.expand_dims(np.mean(X_train, axis=axis),axis=axis)
            x_train_std = np.expand_dims(np.std(X_train, axis=axis),axis=axis)
            x_test_mean = np.expand_dims(np.mean(X_test, axis=axis),axis=axis)
            x_test_std = np.expand_dims(np.std(X_test, axis=axis),axis=axis)
            return (X_train - x_train_mean) / x_train_std, (X_test - x_test_mean) / x_test_std, y_train, y_test


def poly_detrend(X_train, X_test, y_train, y_test, poly_order=2):
    """
    Polynomial De-trending
    Removes a polynomial trend from the spectrum.

    :param x: shape (n_samples, n_features)
    :param poly_order: The order of the polynomial to fit, [2, 4]
    """




    x_det = np.zeros_like(X_train)
    for i in range(X_train.shape[0]):
        x_det[i] = polynomial(X_train[i].copy(), order=poly_order, plot=False)

    X_test_det = np.zeros_like(X_test)
    for i in range(X_test.shape[0]):
        X_test_det[i] = polynomial(X_test[i].copy(), order=poly_order, plot=False)

    return x_det,X_test_det, y_train, y_test

def remove_baseline(X_train2,X_test2,y_train2,y_test2):
    for i in range(X_train2.shape[0]):
        X_train2[i] = X_train2[i] - np.mean(X_train2[i])
    for i in range(X_test2.shape[0]):
        X_test2[i] = X_test2[i] - np.mean(X_test2[i])
    return X_train2, X_test2, y_train2, y_test2



def snv(X_train, X_test, y_train, y_test):
    """
    Standard Normal Variate (SNV)
    x_snv == standardization(x, axis=1) == StandardScaler().fit_transform(x.T).T

    :param x: shape (n_samples, n_features).
    """

    return standardization(X_train, X_test, y_train, y_test, axis=1)

def savgol(X_train, X_test, y_train, y_test, window_len=11, poly=1, deriv=1,delta=1.0,axis=-1,mode='interp',cval=0.0):
    """
    Savitzky-Golay (SG)

    :param x: shape (n_samples, n_features)
    :param window_len: The length of the filter window
    :param poly: The order of the polynomial used to fit the samples
    :param deriv: The order of the derivative to compute
    """
    X_train_sg = savgol_filter(X_train, window_len, poly, deriv,delta,axis,mode,cval)
    X_test_sg = savgol_filter(X_test, window_len, poly, deriv,delta,axis,mode,cval)

    return X_train_sg, X_test_sg, y_train, y_test

def rnv(X_train, X_test, y_train, y_test, percent=25):
    """
    Robust Normal Variate transformation
    z = [x - precentile(x)]/std[x <= percentile(x)]
    where precentile(x) is the percentile in dataset x,
    which defaults to the 25th percentile according to the paper's prompts,
    but may be set to 10 depending on the situation.

    :param X_train: raw spectrum training data, shape (n_train_samples, n_features)
    :param X_test: raw spectrum testing data, shape (n_test_samples, n_features)
    :param y_train: training labels
    :param y_test: testing labels
    :param percent: int from 0-100, default=25; see np.percentile for detail.
    :return: X_train_rnv, X_test_rnv, y_train, y_test
    """

    def apply_rnv(x):
        percentile_value = np.percentile(x, percent, method="median_unbiased")
        return (x - percentile_value) / np.std(x[x <= percentile_value])

    X_train_rnv = np.apply_along_axis(apply_rnv, 1, X_train)
    X_test_rnv = np.apply_along_axis(apply_rnv, 1, X_test)

    return X_train_rnv, X_test_rnv, y_train, y_test

def msc(X_train, X_test, y_train, y_test, mean_center=True, reference=None):
    """
    Multiplicative Scatter Correction.

    :param x: shape (n_samples, n_features)
    :param mean_center: mean-centering each row of the 2D array X if true
    :param reference: optional reference spectrum
    """

    if mean_center:
        X_train = np.array([row - row.mean() for row in X_train])

    x_msc = np.empty(X_train.shape)
    x_ref = np.mean(X_train, axis=0) if reference is None else reference

    for i in range(X_train.shape[0]):
        a, b = np.polyfit(x_ref, X_train[i], 1)
        x_msc[i] = (X_train[i] - b) / a

    if mean_center:
        X_test = np.array([row - row.mean() for row in X_test])
    X_test_msc = np.empty(X_test.shape)
    X_test_ref = np.mean(X_test, axis=0) if reference is None else reference
    for i in range(X_test.shape[0]):
        a, b = np.polyfit(X_test_ref, X_test[i], 1)
        X_test_msc[i] = (X_test[i] - b) / a

    return x_msc, X_test_msc, y_train, y_test

def d1(X_train, X_test, y_train, y_test):
    """ First derivative
    :param X_train: raw spectrum training data, shape (n_train_samples, n_features)
    :param X_test: raw spectrum testing data, shape (n_test_samples, n_features)
    :param y_train: training labels
    :param y_test: testing labels
    :return: X_train_d1, X_test_d1, y_train, y_test
    """
    n_train, p_train = X_train.shape
    n_test, p_test = X_test.shape

    X_train_d1 = np.ones((n_train, p_train - 1))
    X_test_d1 = np.ones((n_test, p_test - 1))

    for i in range(n_train):
        X_train_d1[i] = np.diff(X_train[i])

    for i in range(n_test):
        X_test_d1[i] = np.diff(X_test[i])

    return X_train_d1, X_test_d1, y_train, y_test

def d2(X_train, X_test, y_train, y_test):
    """ Second derivative
    :param X_train: raw spectrum training data, shape (n_train_samples, n_features)
    :param X_test: raw spectrum testing data, shape (n_test_samples, n_features)
    :param y_train: training labels
    :param y_test: testing labels
    :return: X_train_d2, X_test_d2, y_train, y_test
    """
    def calculate_second_derivative(data):
        if isinstance(data, pd.DataFrame):
            data = data.values
        temp2 = (pd.DataFrame(data)).diff(axis=1)
        temp3 = np.delete(temp2.values, 0, axis=1)
        temp4 = (pd.DataFrame(temp3)).diff(axis=1)
        spec_D2 = np.delete(temp4.values, 0, axis=1)
        return spec_D2

    X_train_d2 = calculate_second_derivative(X_train)
    X_test_d2 = calculate_second_derivative(X_test)

    return X_train_d2, X_test_d2, y_train, y_test

def move_avg(X_train, X_test, y_train, y_test, window_size=11):
    """滑动平均滤波
       :param X_train: raw spectrum training data, shape (n_train_samples, n_features)
       :param X_test: raw spectrum testing data, shape (n_test_samples, n_features)
       :param y_train: training labels
       :param y_test: testing labels
       :param window_size: int, odd number
       :return: X_train_ma, X_test_ma, y_train, y_test
    """
    if window_size % 2 == 0:
        raise ValueError('预处理方法move_avg的window_size必须是奇数')
    def apply_move_avg(x):
        x_ma = copy.deepcopy(x)
        for i in range(x.shape[0]):
            out0 = np.convolve(x_ma[i], np.ones(window_size, dtype=int), 'valid') / window_size
            r = np.arange(1, window_size - 1, 2)
            start = np.cumsum(x_ma[i, :window_size - 1])[::2] / r
            stop = (np.cumsum(x_ma[i, :-window_size:-1])[::2] / r)[::-1]
            x_ma[i] = np.concatenate((start, out0, stop))
        return x_ma

    X_train_ma = apply_move_avg(X_train)
    X_test_ma = apply_move_avg(X_test)

    return X_train_ma, X_test_ma, y_train, y_test

def baseline_iarpls(X_train, X_test, y_train, y_test, lam=1000):
    """Baseline correction using IARPLS
    :param X_train: raw spectrum training data, shape (n_train_samples, n_features)
    :param X_test: raw spectrum testing data, shape (n_test_samples, n_features)
    :param y_train: training labels
    :param y_test: testing labels
    :param lam: Lambda parameter for IARPLS
    :return: X_train_iarpls, X_test_iarpls, y_train, y_test
    """

    def apply_iarpls(x):
        x_iarpls = copy.deepcopy(x)
        for k in range(x.shape[0]):
            baseline, _ = iarpls(x_iarpls[k, :], lam)
            x_iarpls[k, :] = x_iarpls[k, :] - baseline
        return x_iarpls

    X_train_iarpls = apply_iarpls(X_train)
    X_test_iarpls = apply_iarpls(X_test)

    return X_train_iarpls, X_test_iarpls, y_train, y_test


def baseline_airpls(X_train, X_test, y_train, y_test, lam=1000):
    """Baseline correction using AIRPLS
    :param X_train: raw spectrum training data, shape (n_train_samples, n_features)
    :param X_test: raw spectrum testing data, shape (n_test_samples, n_features)
    :param y_train: training labels
    :param y_test: testing labels
    :param lam: Lambda parameter for AIRPLS
    :return: X_train_airpls, X_test_airpls, y_train, y_test
    """

    def apply_airpls(x):
        x_airpls = copy.deepcopy(x)
        for k in range(x.shape[0]):
            baseline, _ = airpls(x_airpls[k, :], lam)
            x_airpls[k, :] = x_airpls[k, :] - baseline
        return x_airpls

    X_train_airpls = apply_airpls(X_train)
    X_test_airpls = apply_airpls(X_test)

    return X_train_airpls, X_test_airpls, y_train, y_test

import copy

def baseline_derpsalsa(X_train, X_test, y_train, y_test, lam=1000):
    """Baseline correction using DERPSALSA
    :param X_train: raw spectrum training data, shape (n_train_samples, n_features)
    :param X_test: raw spectrum testing data, shape (n_test_samples, n_features)
    :param y_train: training labels
    :param y_test: testing labels
    :param lam: Lambda parameter for DERPSALSA
    :return: X_train_derpsalsa, X_test_derpsalsa, y_train, y_test
    """
    def apply_derpsalsa(x):
        x_derpsalsa = copy.deepcopy(x)
        for k in range(x.shape[0]):
            baseline, _ = derpsalsa(x_derpsalsa[k, :], lam)
            x_derpsalsa[k, :] = x_derpsalsa[k, :] - baseline
        return x_derpsalsa

    X_train_derpsalsa = apply_derpsalsa(X_train)
    X_test_derpsalsa = apply_derpsalsa(X_test)

    return X_train_derpsalsa, X_test_derpsalsa, y_train, y_test












# 特征选择  要求输入为 X_train, X_test, y_train, y_test, **params  输出为 X_train, X_test, y_train, y_test




# 去除方差较大的特征并进行最大最小归一化
def remove_high_variance_and_normalize(X_train, X_test, y_train, y_test, remove_feat_ratio):
    # 计算X_train各特征的方差
    variances = np.var(X_train, axis=0)

    # 根据方差对特征索引进行排序（从小到大）
    sorted_indices = np.argsort(variances)

    # 计算要移除的特征数量
    num_features_to_remove = int(X_train.shape[1] * remove_feat_ratio)

    # 确定要移除的特征索引（取方差较大的那部分）
    features_to_remove = sorted_indices[-num_features_to_remove:]

    # 去除X_train和X_test中的指定特征
    X_train = np.delete(X_train, features_to_remove, axis=1)
    X_test = np.delete(X_test, features_to_remove, axis=1)

    # 对剩余特征进行最大最小归一化
    for i in range(X_train.shape[1]):
        min_val = np.min(X_train[:, i])
        max_val = np.max(X_train[:, i])
        X_train[:, i] = (X_train[:, i] - min_val) / (max_val - min_val)
        X_test[:, i] = (X_test[:, i] - min_val) / (max_val - min_val)

    return X_train, X_test, y_train, y_test




#随机选择
def random_select(X_train, X_test, y_train, y_test,min_features=1, max_features=20, random_seed=42,mylog = None):
    """
    Randomly select a subset of features

    :param x: shape (n_samples, n_features)
    :param min_features: int, minimum number of features to select
    :param max_features: int, maximum number of features to select
    :param random_seed: int, random seed
    """

    if not 0 < min_features <= max_features <= X_train.shape[1]:
        raise ValueError('Unexpected min_features or max_features value.')
    n = np.random.randint(min_features, max_features )
    num_columns = X_train.shape[1]
    # 随机选择 n 个不重复的列索引
    selected_columns = np.random.choice(num_columns, n, replace=False)

    selected_columns = sorted(selected_columns)

    selected_columns = np.array([49, 53, 81, 92, 105, 109, 132, 190, 201, 220, 258, 283, 319, 322, 376, 380, 396, 397])




    # 从 data_X 中选择这些列
    X_train = X_train[:, selected_columns]
    X_test = X_test[:, selected_columns]
    mylog.info(selected_columns)

    return X_train, X_test, y_train, y_test






# @st.cache_data
def cars(x,X_test, y, y_test,n_sample_runs=50, pls_components=20, n_cv_folds=5):
    samples_ratio = 0.8
    n_samples, n_wavelengths = x.shape

    # prepare for edf_schedule
    u = np.power((n_wavelengths / 2), (1 / (n_sample_runs - 1)))
    k = (1 / (n_sample_runs - 1)) * np.log(n_wavelengths / 2)

    n_fit_samples = np.round(n_samples * samples_ratio)
    b2 = np.arange(n_wavelengths)
    x_copy = copy.deepcopy(x)
    idx_with_x = np.vstack((np.array(b2).reshape(1, -1), x))

    wave_data = []
    wave_num_list = []
    RMSECV = []
    selection_ratios = []
    for i in range(1, n_sample_runs + 1):

        # edf schedule
        selection_ratios.append(u * np.exp(-1 * k * i))
        wave_num = int(np.round(selection_ratios[i - 1] * n_wavelengths))
        wave_num_list = np.hstack((wave_num_list, wave_num))

        fitting_samples_index = np.random.choice(np.arange(n_samples), size=int(n_fit_samples), replace=False)
        wavelength_index = b2[0:wave_num].reshape(1, -1)[0]

        x_pls_fit = x_copy[np.ix_(list(fitting_samples_index), list(wavelength_index))]
        y_pls_fit = y[fitting_samples_index]
        x_copy = x_copy[:, wavelength_index]

        idx_with_x = idx_with_x[:, wavelength_index]
        d = idx_with_x[0, :].reshape(1, -1)

        if (n_wavelengths - wave_num) > 0:
            d = np.hstack((d, np.full((1, (n_wavelengths - wave_num)), -1)))

        if len(wave_data) == 0:
            wave_data = d
        else:
            wave_data = np.vstack((wave_data, d.reshape(1, -1)))

        if wave_num < pls_components:
            pls_components = wave_num

        pls = PLSRegression(n_components=pls_components)
        pls.fit(x_pls_fit, y_pls_fit)
        beta = pls.coef_
        b = np.abs(beta)
        b2 = np.argsort(-b).squeeze()

        def pc_cross_validation(x, y, pc, cv):
            kf = KFold(n_splits=cv, shuffle=True)
            RMSECV = []
            for i in range(pc):
                RMSE = []
                for train_index, test_index in kf.split(x):
                    x_train, x_test = x[train_index], x[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    pls = PLSRegression(n_components=i + 1)
                    pls.fit(x_train, y_train)
                    y_predict = pls.predict(x_test)
                    RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
                RMSE_mean = np.mean(RMSE)
                RMSECV.append(RMSE_mean)
            rindex = np.argmin(RMSECV)
            RMSE_mean_min = RMSECV[rindex]
            return RMSECV, rindex, RMSE_mean_min

        _, rindex, rmse_min = pc_cross_validation(x_pls_fit, y_pls_fit, pls_components, n_cv_folds)
        RMSECV.append(rmse_min)

    wavelengths_set = []
    for i in range(wave_data.shape[0]):
        wd = wave_data[i, :]
        wd_ones = np.ones((len(wd)))
        for j in range(len(wd)):
            ind = np.where(wd == j)
            if len(ind[0]) == 0:
                wd_ones[j] = 0
            else:
                wd_ones[j] = wd[ind[0]]
        if len(wavelengths_set) == 0:
            wavelengths_set = copy.deepcopy(wd_ones)
        else:
            wavelengths_set = np.vstack((wavelengths_set, wd_ones.reshape(1, -1)))

    min_idx = np.argmin(RMSECV)
    optimal = wavelengths_set[min_idx, :]
    opt_wavelengths = np.where(optimal != 0)[0]

    return x[:,opt_wavelengths], X_test[:,opt_wavelengths], y, y_test

def spa(x,X_test,y_train,y_test ,i_init, method=0, mean_center=True):
    """
    Successive Projections Algorithm
    https://www.sciencedirect.com/science/article/pii/S0169743901001198
    https://www.sciencedirect.com/science/article/pii/S0165993612002750

    :param x:
    :param i_init: the index of the initial wavelength to be selected
    :param method: 0 or 1 (method 1 is extremely slow in matrix inversion)
    :param mean_center: mean center the columns of the 2D array X if true
    :return:
    """

    if mean_center: x,X_test,y_train,y_test = mean_centering(x,X_test,y_train,y_test, axis=0)

    m = x.shape[0]
    n = x.shape[1]

    if not 0 <= i_init < n: raise ValueError('Unexpected i_init value.')
    if not method in [0, 1]: raise ValueError('Unexpected method value.')

    x_copy = np.array(x)
    x_orth = [x[:, i_init]]

    n_select = [np.linalg.norm(x_orth[0])]
    i_select = [i_init]
    i_remain = [i for i in range(n) if i != i_init]
    for _ in range(min(m, n) - 1):

        n_max = -1
        i_max = -1

        x_trans = np.array(x_orth).T

        for j in i_remain:

            if method == 0:

                xi = x_trans[:, -1]
                xj = x_copy[:, j]

                proj = xj - xi * np.dot(xi, xj) / np.dot(xi, xi)
                norm = np.linalg.norm(proj)

            else:

                x = x_copy[:, j]

                proj = (np.identity(m) - x_trans @ np.linalg.inv(x_trans.T @ x_trans) @ x_trans.T) @ x
                norm = x.T @ proj

            x_copy[:, j] = proj
            if norm > n_max: n_max, i_max = norm, j

        x_orth.append(x_copy[:, i_max])

        n_select.append(n_max)
        i_select.append(i_max)
        i_remain.remove(i_max)

    return x[:,i_select], X_test[:, i_select], y_train, y_test


def corr_coefficient(x, x_test, y, y_test, threshold):
    from sklearn.feature_selection import f_regression

    coef = f_regression(x, y)[0]
    # 计算保留特征的数量
    num_features_to_keep = int(len(coef) * threshold)
    # 找到绝对值最大的前 num_features_to_keep 个特征的索引
    i_select = np.argsort(np.abs(coef))[-num_features_to_keep:]

    X_train = x[:, i_select]
    X_test = x_test[:, i_select]

    return X_train, X_test, y, y_test

def anova(x,x_test,y,y_test, threshold):
    # task_ = 'reg'
    # if "isReg" in st.session_state:
    #     pass
    # if "isReg" in st.session_state and st.session_state.isReg == True:
    #     task_ = 'reg'
    # elif "isReg" in st.session_state and st.session_state.isReg:
    #     task_ = 'cls'
    # else:
    task_ = 'reg' if np.unique(y).shape[0] > 10  else  'cls'
    from sklearn.feature_selection import SelectKBest, f_regression, f_classif
    features_score = pd.DataFrame()
    if task_ == 'reg':
        fs = SelectKBest(score_func=f_regression, k=x.shape[1])
        fit = fs.fit(x, y)
    elif task_ == 'cls':
        fs = SelectKBest(score_func=f_classif, k=x.shape[1])
        fit = fs.fit(x, y)

    def normalization_(x, axis=None, min=0, max=1):
            """
            return the normalization of the 2D array X such that each element is in [0, 1]
            scaling each feature to a given range,
            The transformation is given by:
            x_std = (x - x.min(axis)) / (x.max(axis) - x.min(axis))
            x_scaled = x_std * (max - min) + min

            :param x: shape (n_samples, n_features)
            :param axis == None: element-wise normalization
                   axis == 0   :  column-wise normalization
                   axis == 1   :     row-wise normalization
            """

            if axis not in [None, 0, 1]:
                raise ValueError('Unexpected axis value.')

            x_min = np.min(x, axis=axis)
            x_max = np.max(x, axis=axis)

            if axis != None:
                x_min = np.expand_dims(x_min, axis=axis)
                x_max = np.expand_dims(x_max, axis=axis)

            x_std = (x - x_min) / (x_max - x_min)
            x_scaled = x_std * (max - min) + min

            return x_scaled

    num_features_to_keep = int(len(fit.scores_) * threshold)
    # i_select = np.where(normalization_(np.abs(fit.scores_)) > threshold)
    i_select = np.argsort(np.abs(normalization_(fit.scores_)))[-num_features_to_keep:]

    # X_train = x[:, i_select[0]]
    # X_test = x_test[:, i_select[0]]


    X_train = x[:, i_select]
    X_test = x_test[:, i_select]
    return X_train, X_test, y, y_test

def fipls(x,x_test,y,y_test, n_intervals, interval_width, n_comp):
    from auswahl import FiPLS
    """
    https://auswahl.readthedocs.io/en/latest/?badge=latest
    """
    pls = PLSRegression(n_components=n_comp)
    selector = FiPLS(n_intervals_to_select=n_intervals, interval_width=interval_width, n_cv_folds=5)
    select_intervals = selector.fit(x, y).get_support()
    i_select = [i for i, value in enumerate(select_intervals) if value]
    X_train = x[:, i_select]
    X_test = x_test[:, i_select]
    return  X_train, X_test, y, y_test








# 降维  要求输入为 X_train, X_test, y_train, y_test, **params  输出为 X_train, X_test, y_train, y_test
def pca(X_train, X_test, y_train, y_test, n_components=2,random_state=42):
    """
    Principal Component Analysis (PCA)

    :param x: shape (n_samples, n_features)
    :param n_components: The number of principal components to retain
    :param scale: scale the principal components by the standard deviation of the corresponding eigenvalue
    """

    pca = PCA(n_components=n_components,random_state=random_state)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca, y_train, y_test






# 模型  要求输入为 X_train, X_test, y_train, y_test, **params  输出为 y_train, y_test, y_train_pred, y_test_pred

def LR(X_train,X_test, y_train,y_test):

    # 线性回归模型
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)

    return y_train,y_test, y_train_pred,y_test_pred

def SVR(x_train, x_test, y_train, y_test,kernel = 'rbf',C = 1.0,epsilon = 0.1,degree = 3,gamma = 'scale'):
    from sklearn.svm import SVR
    svr = SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree, gamma=gamma, max_iter=100000)
    svr.fit(x_train, y_train)
    y_train_pred = svr.predict(x_train)
    y_test_pred = svr.predict(x_test)
    return y_train, y_test, y_train_pred, y_test_pred

def PLSR(x_train, x_test, y_train, y_test, n_components=2,scale=True):
    from sklearn.cross_decomposition import PLSRegression
    plsr = PLSRegression(n_components=n_components, scale=scale)
    plsr.fit(x_train, y_train)
    y_train_pred = plsr.predict(x_train)
    y_test_pred = plsr.predict(x_test)
    return y_train, y_test, y_train_pred, y_test_pred

def bayes(x_train, x_test, y_train, y_test, n_iter=300, tol=1.0e-3, alpha_1=1.0e-6, alpha_2=1.0e-6, lambda_1=1.0e-6,
          lambda_2=1.0e-6, compute_score=False, fit_intercept=True):
    from sklearn.linear_model import BayesianRidge
    bayes = BayesianRidge(tol=tol, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1,
                          lambda_2=lambda_2,
                          compute_score=compute_score, fit_intercept=fit_intercept)
    bayes.fit(x_train, y_train)
    y_train_pred = bayes.predict(x_train)
    y_test_pred = bayes.predict(x_test)
    return y_train, y_test, y_train_pred, y_test_pred


def RFR(x_train, x_test, y_train, y_test, n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2,
        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
        min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None,
        random_state=None, verbose=0, warm_start=False,**kwargs):
    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                 bootstrap=bootstrap, oob_score=oob_score,
                                n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,**kwargs)
    rfr.fit(x_train, y_train)
    y_train_pred = rfr.predict(x_train)
    y_test_pred = rfr.predict(x_test)
    return y_train, y_test, y_train_pred, y_test_pred


def BayesianRidge(x_train, x_test, y_train, y_test, alpha_1=1.0, alpha_2=1.0, lambda_1=1.0,
                  lambda_2=1.0, tol=0.001, fit_intercept=True,
                  copy_X=True, verbose=False, **kwargs):
    from sklearn.linear_model import BayesianRidge
    br = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1,
                       lambda_2=lambda_2, tol=tol, fit_intercept=fit_intercept,
                      copy_X=copy_X, verbose=verbose, **kwargs)
    br.fit(x_train, y_train)
    y_train_pred = br.predict(x_train)
    y_test_pred = br.predict(x_test)
    return y_train, y_test, y_train_pred, y_test_pred





#######################################################################################################################################
##########################################################分类模型 #####################################################################
#######################################################################################################################################
def logr(x_train, x_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    logr = LogisticRegression()
    logr.fit(x_train, y_train)
    y_train_pred = logr.predict(x_train)
    y_test_pred = logr.predict(x_test)
    return y_train, y_test, y_train_pred, y_test_pred


def SVM(X_train, X_test,y_train, y_test, kernel='linear', C=1.0, gamma='scale', degree=3,random_state = 42,
                           coef0=0.0):

    """
    训练和评估SVM模型

    参数:
    - X_train: 训练集特征
    - y_train: 训练集标签
    - X_test: 测试集特征
    - y_test: 测试集标签
    - kernel: SVM核函数，默认为线性核
    - C: 正则化参数，默认为1.0
    - gamma: 核函数的尺度参数，默认为'scale'，也可以是具体的数值
    - degree: 多项式核的次数，默认为3
    - coef0: 核函数的独立项，默认为0.0

    返回:
    - y_train: 训练集真实标签
    - y_test: 测试集真实标签
    - y_train_pred: 训练集预测标签
    - y_test_pred: 测试集预测标签
    """
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 创建SVM模型
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0, random_state=random_state, max_iter=100000)

    # 训练模型
    svm_model.fit(X_train, y_train)

    # 在训练集上进行预测
    y_train_pred = svm_model.predict(X_train)

    # 在测试集上进行预测
    y_test_pred = svm_model.predict(X_test)

    return y_train, y_test, y_train_pred, y_test_pred






def DT(X_train, X_test, y_train, y_test, criterion='gini', splitter='best',
                        min_samples_split=2, min_samples_leaf=1, random_state = 42):
    from sklearn import tree
    from sklearn.metrics import accuracy_score
    # 创建决策树分类器
    classifier = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, min_samples_split=min_samples_split,
                                              min_samples_leaf=min_samples_leaf,random_state=random_state)

    # 训练模型
    classifier.fit(X_train, y_train)

    # 使用模型进行训练集和测试集的预测
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)


    return  y_train, y_test, y_train_pred, y_test_pred



def RandomForest(X_train, X_test, y_train, y_test, n_estimators=100, criterion='gini', min_samples_split=2, min_samples_leaf=1, random_state=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # 创建随机森林分类器
    classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf, random_state=random_state)

    # 训练模型
    classifier.fit(X_train, y_train)

    # 使用模型进行训练集和测试集的预测
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    return y_train, y_test, y_train_pred, y_test_pred


def KNN(X_train, X_test, y_train, y_test, n_neighbors=5, weights='uniform', algorithm='auto', p=2, metric='minkowski', leaf_size=30):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    # 创建K近邻分类器
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p, metric=metric, leaf_size=leaf_size)

    # 训练模型
    classifier.fit(X_train, y_train)

    # 使用模型进行训练集和测试集的预测
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    return y_train, y_test, y_train_pred, y_test_pred

def CustomNaiveBayes(X_train, X_test, y_train, y_test, classifier_type='gaussian', alpha=1.0, binarize=0.0):
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    from sklearn.metrics import accuracy_score

    if classifier_type == 'gaussian':
        classifier = GaussianNB(var_smoothing=alpha)
    elif classifier_type == 'multinomial':
        classifier = MultinomialNB(alpha=alpha)
    elif classifier_type == 'bernoulli':
        classifier = BernoulliNB(alpha=alpha, binarize=binarize)
    else:
        raise ValueError("Invalid classifier type. Supported types: 'gaussian', 'multinomial', 'bernoulli'.")

    # 训练模型
    classifier.fit(X_train, y_train)

    # 使用模型进行训练集和测试集的预测
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    return y_train, y_test, y_train_pred, y_test_pred


def GradientBoostingTree(X_train, X_test, y_train, y_test, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score

    # 创建梯度提升树分类器
    classifier = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)

    # 训练模型
    classifier.fit(X_train, y_train)

    # 使用模型进行训练集和测试集的预测
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    return y_train, y_test, y_train_pred, y_test_pred


def XGBoost(X_train, X_test, y_train, y_test, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, colsample_bytree=1.0, random_state=42):
    import xgboost as xgb
    from sklearn.metrics import accuracy_score

    # 创建XGBoost分类器
    classifier = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                                   subsample=subsample, colsample_bytree=colsample_bytree, random_state=random_state)

    # 训练模型
    classifier.fit(X_train, y_train)

    # 使用模型进行训练集和测试集的预测
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    return y_train, y_test, y_train_pred, y_test_pred






########################################################################################################################################################################
##### 把上面的所有函数都构建成管道#####

























if __name__ == '__main__':
    # Generate synthetic data
    np.random.seed(0)
    n_samples = 100
    n_features = 50

    X_train = np.random.rand(n_samples, n_features) * 10
    X_test = np.random.rand(n_samples, n_features) * 10

    # Adding a quadratic trend
    t = np.linspace(0, 1, n_features)
    for i in range(n_samples):
        X_train[i] += 5 * t**2
        X_test[i] += 5 * t**2

    y_train = np.random.randint(0, 2, n_samples)
    y_test = np.random.randint(0, 2, n_samples)

    # Applying the de-trending function
    X_train_det, X_test_det, y_train, y_test = poly_detrend(X_train, X_test, y_train, y_test, poly_order=2)