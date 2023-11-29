"""For data preprocessing
-------
Functions:
----------
    - SNV(X) >> ndarray like X  # Standard Normal Variate (SNV)
    - MC(X, axis=None) >> ndarray like X  # mean-centering
    - normalization(X, axis=None) >> ndarray like X  # normalization
    

"""

import numpy as np
def MC(X, axis=None):
    import numpy as np
    '''
    return the mean-centering of the 2D array X
    axis == None: element-wise mean-centering
    axis == 0   :  column-wise mean-centering
    axis == 1   :     row-wise mean-centering
    '''
    
    if axis not in [None, 0, 1]: raise ValueError('Unexpected axis value.')
    
    X_mean = np.mean(X, axis=axis)
    
    if axis != None: X_mean = np.expand_dims(X_mean, axis=axis)
    
    return X - X_mean

def normalization(X, axis=None):
    
    '''
    --------
    return the normalization of the 2D array X such that each element is in [0, 1]
    axis == None: element-wise normalization
    axis == 0   :  column-wise normalization
    axis == 1   :     row-wise normalization
    '''
    
    if axis not in [None, 0, 1]: raise ValueError('Unexpected axis value.')
    
    X_min = np.min(X, axis=axis)
    X_max = np.max(X, axis=axis)
    
    if axis != None:
        
        X_min = np.expand_dims(X_min, axis=axis)
        X_max = np.expand_dims(X_max, axis=axis)
    
    return (X - X_min) / (X_max - X_min)

def SNV(X):
    '''
    Standard Normal Variate (SNV)
    '''
    x = X
    x_snv = np.zeros_like(x)
    for i in range(x.shape[0]):
        x_snv[i] = (x[i] - np.mean(x[i])) / np.std(x[i])
    
    return x_snv

def SG(X,window_length=14, polyorder=1,deriv = 0):
    """ SG 对X进行处理，不会用到y
    -------
    Parameters:
    -------
        window_length : int ,窗口长度
        polyorder : int ,多项式拟合的阶次
    -------
    return : 
    -------
        SG处理后的ndarrary,数据格式和X一样

    """
    from scipy.signal import savgol_filter
    SG_data = savgol_filter(x=X,window_length=window_length, polyorder=polyorder,deriv=deriv)
    return SG_data