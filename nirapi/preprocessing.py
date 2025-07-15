"""For data preprocessing
-------
Functions:
----------
    - MC(X, axis=None) >> ndarray like X  # mean-centering
    - normalization(X, axis=None) >> ndarray like X  # normalization
    - remove_baseline_drift(X) >> ndarray like X  # remove baseline drift
    - SNV(X) >> ndarray like X  # Standard Normal Variate (SNV)
    - normalization(X, axis=None) >> ndarray like X  # normalization
    - SG(X,window_length=14, polyorder=1,deriv = 0)
    - MSC(X, mean_center=True, reference=None)
    - weighted_SNV(X,weighed_martix = None,epsilon = 1e-5,Ns_times = 100,Nw_times = 100,draw_weighed_martix = False)
    - RNV(X, percent=25)
    

"""



import numpy as np
from scipy.spatial.distance import mahalanobis

def remove_top_20_percent_mahalanobis(X):
    """
    remove_top_20_percent_mahalanobis
    --------
    Parameters:
    --------
        X : ndarray, shape (n_samples, n_features)
            The data to be processed.
            
    Returns:
    --------
        X_cleaned : ndarray, shape (n_samples, n_features)
            The data after removing the top 20% with the largest Mahalanobis distance.
        distances : ndarray, shape (n_samples,)
            The Mahalanobis distances of each point.
    """
    
    # 计算每列的均值
    mean_X = np.mean(X, axis=0)
    
    # 计算协方差矩阵及其逆矩阵
    cov_matrix = np.cov(X, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    # 计算每个点的马氏距离
    distances = np.array([mahalanobis(x, mean_X, inv_cov_matrix) for x in X])
    
    # 找到马氏距离最大的20%的点
    cutoff_index = int(len(distances) * 0.2)
    # 按距离值从大到小排序并找到前20%的索引
    outlier_indices = np.argsort(distances)[-cutoff_index:]
    
    # 创建一个新的数组，将这些索引的值设为 NaN
    X_cleaned = X.copy()
    X_cleaned[outlier_indices, :] = np.nan
    
    return X_cleaned, distances



import numpy as np

def remove_outliers(X, threshold=3, axis=None):
    """
    remove_outliers
    --------
    Parameters:
    --------
        X : ndarray, shape (n_samples, n_features)
            The data to be processed.
        threshold : float, optional, default=3
            The Z-score threshold to identify outliers. Points with Z-score 
            greater than the threshold will be considered outliers.
        axis : int or None, optional
            The axis along which to calculate Z-scores. If None, it will flatten the array.
            axis == 0 : column-wise outlier detection
            axis == 1 : row-wise outlier detection
            axis == None: element-wise outlier detection
            
    Returns:
    --------
        X_cleaned : ndarray, shape (n_samples, n_features)
            The data after removing outliers. The outliers are replaced with NaN.
    """
    
    if axis not in [None, 0, 1]:
        raise ValueError('Unexpected axis value.')
    
    mean_X = np.mean(X, axis=axis)
    std_X = np.std(X, axis=axis)
    
    if axis is not None:
        mean_X = np.expand_dims(mean_X, axis=axis)
        std_X = np.expand_dims(std_X, axis=axis)
    
    Z_scores = (X - mean_X) / std_X
    
    # 将 Z-score 超过阈值的值设为 NaN
    X_cleaned = np.where(np.abs(Z_scores) > threshold, np.nan, X)
    
    return X_cleaned




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

def remove_baseline_drift(X):
    """remove_baseline_drift
    --------
    Parameters:
    --------
        X : ndarray like, shape (n_samples, n_features)
            The data to be processed.
    --------
    Returns:
    --------
        X_remove_baseline_drift : ndarray like, shape (n_samples, n_features)
            The data after remove_baseline_drift processing.
    
    """
    X_remove_baseline_drift = X - np.mean(X,axis=1).reshape(-1,1)
    return X_remove_baseline_drift

def VSN(X,
                 weighed_martix = None,
                 epsilon = 1e-5,
                 Ns_times = 100,
                 Nw_times = 100,
                 draw_weighed_martix = False,
                 tqdm = False,
                 ):
    """weighted SNV: VSN: Variable sorting for normalization ---- https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/cem.3164
    --------
    Parameters:
    --------
        X : ndarray like, shape (n_samples, n_features)
            The data to be processed.
        weighed_martix : list, shape (n_features,)
            The weighed martix to be used in VSN.
        epsilon : float, default = 1e-5
            The threshold of VSN.
        Ns_times : int, default = 100
            The number of times to repeat the process of selecting two samples.
        Nw_times : int, default = 100
            The number of times to repeat the process of selecting two wavelengths.
        draw_weighed_martix : bool, default = False 
            If True, draw the weighed_martix plot.
        tqdm : bool, default = False
            If True, use tqdm to show the progress bar.

    --------
    Returns:
    --------
        X_VSN : ndarray like, shape (n_samples, n_features)
            The data after VSN processing.
    
    """
    ### change X shape if X.ndim == 1
    if X.ndim == 1:
        X = np.array([X])


    ### calculate weighed_martix if weighed_martix is None
    if weighed_martix is None:
        weighed_martix = [0]*X.shape[1] # init weighed_martix

        # If there is only one row, then weighed_martix will have all the values 1, indicating a traditional SNV
        if X.shape[0] == 1:
            weighed_martix = np.array([1]*X.shape[1]) 
        else:
            epsilon = epsilon # epsilon , which is the threshold of VSN
            Sz = [] # set Sz，which holds the selected i and j wavelengths and is finally used to calculate weighted_martix

            # Repeat Ns times
            if tqdm:
                from tqdm import tqdm
                Ns_times_tqdm = tqdm(range(Ns_times))
            else:
                Ns_times_tqdm = range(Ns_times)
            for Ns in Ns_times_tqdm:
                samples = np.random.choice(np.arange(0, X.shape[0]), size=2, replace=False) # random select two samples
                max_E_ab = 0 # max E(a,b) 
                i_correspond_to_max_E_ab = 0 
                j_correspond_to_max_E_ab = 0
                Nw = 0 # Nw times 


                # Repeat Nw times
                while Nw < Nw_times:
                    Nw += 1
                    wavelengths = np.random.choice(np.arange(0, X.shape[1]), size=2, replace=False) # random select two wavelengths
                    # must X[samples[0], wavelengths[1]] != X[samples[1], wavelengths[1]]:
                    if X[samples[0], wavelengths[1]] - X[samples[1], wavelengths[1]] == 0:
                        Nw -= 1
                        continue
                    
                    
                    #### two equations: X_1_i = a*X_1_j + b, X_2_i = a*X_2_j + b 
                    #### calculate a and b by {X_1_i, X_2_i} and {X_1_j, X_2_j}
                    a = (X[samples[0], wavelengths[0]] - X[samples[1], wavelengths[0]]) / (X[samples[0], wavelengths[1]] - X[samples[1], wavelengths[1]])
                    b = X[samples[0], wavelengths[0]] - a * X[samples[0], wavelengths[1]]


                    #### calculate the expect value of E(a,b) = \sum_{k = 1}^p \sigma(a,b,k) ,witch p is the number of all wavelengths(X.shape[1])),
                    #### and \sigma(a,b,k) = 1 if |X_1_k - a*X_2_k - b| < epsilon, else \sigma(a,b,k) = 0
                    E_ab = 0
                    for k in range(X.shape[1]):
                        if abs(X[samples[0], k] - a * X[samples[0], wavelengths[1]] - b) < epsilon:
                            E_ab += 1


                    # update max_E_ab
                    if E_ab > max_E_ab:
                        max_E_ab = E_ab
                        i_correspond_to_max_E_ab = wavelengths[0]
                        j_correspond_to_max_E_ab = wavelengths[1]
                Sz.append([i_correspond_to_max_E_ab, j_correspond_to_max_E_ab])
            # calculate weighted_martix
            for i in range(len(Sz)):
                weighed_martix[Sz[i][0]] += 1
                weighed_martix[Sz[i][1]] += 1
            weighed_martix = np.array(weighed_martix) / (len(Sz))
            # print(weighed_martix)
                    
    
    ### draw weighed_martix plot
    import matplotlib.pyplot as plt
    if draw_weighed_martix:
        plt.plot(weighed_martix)
        plt.title("weighed_martix")
        plt.show()


    ### calculate X_VSN 
    X_VSN = np.zeros_like(X)
    for i in range(X.shape[0]):
        Xc = X[i] - np.mean(X[i]*weighed_martix)
        X_VSN[i] = (X[i] - np.mean(X[i]*weighed_martix)) / np.std(Xc*weighed_martix)
        # print("mean: ",np.mean(X[i]*weighed_martix),"std: ",np.std(Xc*weighed_martix))
    return X_VSN



def SNV(X,
        replace_wave:list = None,
        ):
    '''
    Standard Normal Variate (SNV)
    --------
    Parameters:
    --------
        X : ndarray like, shape (n_samples, n_features)
            The data to be processed.
        replace_wave : list, default = None
            用其他波段的数据替换掉SNV中原本所有的波段
    --------
    Returns:
    --------
        X_snv : ndarray like, shape (n_samples, n_features)
            The data after SNV processing.
    
    '''
    if X.ndim == 1:
        X = np.array([X])
    if replace_wave is not None:
        X_base_line = X[:,replace_wave[0]:replace_wave[1]] * 5
    else:
        X_base_line = X
    x = X
    x_snv = np.zeros_like(x)
    # print(X_base_line.shape)
    # print(X_base_line)
    for i in range(x.shape[0]):
        x_snv[i] = (x[i] - np.mean(X_base_line[i])) / np.std(X_base_line[i])
        # x_snv[i] = (x[i] - np.mean(X_base_line[i]))
        # print(np.std(X_base_line[i]))
    
    return x_snv

def weighted_SNV(X,
                 weighed_martix = None,
                 epsilon = 1e-5,
                 Ns_times = 100,
                 Nw_times = 100,
                 draw_weighed_martix = False,
                 ):
    """weighted SNV: VSN: Variable sorting for normalization ---- https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/cem.3164
    --------
    Parameters:
    --------
        X : ndarray like, shape (n_samples, n_features)
            The data to be processed.
        weighed_martix : list, shape (n_features,)
            The weighed martix to be used in VSN.
        epsilon : float, default = 1e-5
            The threshold of VSN.
        Ns_times : int, default = 100
            The number of times to repeat the process of selecting two samples.
        Nw_times : int, default = 100
            The number of times to repeat the process of selecting two wavelengths.
        draw_weighed_martix : bool, default = False 
            If True, draw the weighed_martix plot.

    --------
    Returns:
    --------
        X_weighted_SNV : ndarray like, shape (n_samples, n_features)
            The data after VSN processing.
    
    """
    ### change X shape if X.ndim == 1
    if X.ndim == 1:
        X = np.array([X])


    ### calculate weighed_martix if weighed_martix is None
    if weighed_martix is None:
        weighed_martix = [0]*X.shape[1] # init weighed_martix

        # If there is only one row, then weighed_martix will have all the values 1, indicating a traditional SNV
        if X.shape[0] == 1:
            weighed_martix = np.array([1]*X.shape[1]) 
        else:
            epsilon = epsilon # epsilon , which is the threshold of VSN
            Sz = [] # set Sz，which holds the selected i and j wavelengths and is finally used to calculate weighted_martix

            # Repeat Ns times
            from tqdm import tqdm
            Ns_times_tqdm = tqdm(range(Ns_times))
            for Ns in Ns_times_tqdm:
                samples = np.random.choice(np.arange(0, X.shape[0]), size=2, replace=False) # random select two samples
                max_E_ab = 0 # max E(a,b) 
                i_correspond_to_max_E_ab = 0 
                j_correspond_to_max_E_ab = 0
                Nw = 0 # Nw times 


                # Repeat Nw times
                while Nw < Nw_times:
                    Nw += 1
                    wavelengths = np.random.choice(np.arange(0, X.shape[1]), size=2, replace=False) # random select two wavelengths
                    # must X[samples[0], wavelengths[1]] != X[samples[1], wavelengths[1]]:
                    if X[samples[0], wavelengths[1]] - X[samples[1], wavelengths[1]] == 0:
                        Nw -= 1
                        continue
                    
                    
                    #### two equations: X_1_i = a*X_1_j + b, X_2_i = a*X_2_j + b 
                    #### calculate a and b by {X_1_i, X_2_i} and {X_1_j, X_2_j}
                    a = (X[samples[0], wavelengths[0]] - X[samples[1], wavelengths[0]]) / (X[samples[0], wavelengths[1]] - X[samples[1], wavelengths[1]])
                    b = X[samples[0], wavelengths[0]] - a * X[samples[0], wavelengths[1]]


                    #### calculate the expect value of E(a,b) = \sum_{k = 1}^p \sigma(a,b,k) ,witch p is the number of all wavelengths(X.shape[1])),
                    #### and \sigma(a,b,k) = 1 if |X_1_k - a*X_2_k - b| < epsilon, else \sigma(a,b,k) = 0
                    E_ab = 0
                    for k in range(X.shape[1]):
                        if abs(X[samples[0], k] - a * X[samples[0], wavelengths[1]] - b) < epsilon:
                            E_ab += 1


                    # update max_E_ab
                    if E_ab > max_E_ab:
                        max_E_ab = E_ab
                        i_correspond_to_max_E_ab = wavelengths[0]
                        j_correspond_to_max_E_ab = wavelengths[1]
                Sz.append([i_correspond_to_max_E_ab, j_correspond_to_max_E_ab])
            # calculate weighted_martix
            for i in range(len(Sz)):
                weighed_martix[Sz[i][0]] += 1
                weighed_martix[Sz[i][1]] += 1
            weighed_martix = np.array(weighed_martix) / (len(Sz))
            # print(weighed_martix)
                    
    
    ### draw weighed_martix plot
    import matplotlib.pyplot as plt
    if draw_weighed_martix:
        plt.plot(weighed_martix)
        plt.title("weighed_martix")
        plt.show()


    ### calculate X_weighted_SNV   
    X_weighted_SNV = np.zeros_like(X)
    for i in range(X.shape[0]):
        Xc = X[i] - np.mean(X[i]*weighed_martix)
        X_weighted_SNV[i] = (X[i] - np.mean(X[i]*weighed_martix)) / np.std(Xc*weighed_martix)
        # print("mean: ",np.mean(X[i]*weighed_martix),"std: ",np.std(Xc*weighed_martix))
    return X_weighted_SNV

def RNV(X, percent=25):
    
    '''
    Robust Normal Variate 
    RNV : z = [x - precentile(x)]/std[x <= percentile(x)]
    .. where precentile(x) is the percentile in dataset x,
        which defaults to the 25th percentile according to the paper's prompts,
        but may be set to 10 depending on the situation.
        code author : Tao Zhang
    ----------
    Parameters
    ----------
    - X : ndarray or list
        Spectral absorbance data, generally two-dimensional,
        you can also enter one-dimensional, other dimensions will report errors. 
    - percent : int from 0-100 ,default = 25
        see np.percentile for detial.
    '''
    
    assert isinstance(X,np.ndarray) or isinstance(X,list),"Variable X is of wrong type, must be ndarray or list"
    
    if isinstance(X,list):
        X = np.array(X)
    
    X_RNV = np.zeros_like(X)
    
    if X.ndim == 2:        
        for i in range(X.shape[0]):
            percentile_value = np.percentile(X[i],percent,method="median_unbiased")
            X_RNV[i] = (X[i]-percentile_value)/np.std(X[i][X[i]<=percentile_value])
    elif X.ndim == 1:
        percentile_value = np.percentile(X,percent,method="median_unbiased")
        X_RNV = (X-percentile_value)/np.std(X[X<=percentile_value])
    else :
        assert False,"Variable X dimension error"
        
    return X_RNV

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

def MSC(X, mean_center=True, reference=None):
        # Find the idealzed spectum for all spectra
        ideal_spectrum = np.mean(X,axis=0)
        # perform a unicariate linear regressing of the spectrum of each sample against the 
        # average spectrum,solving a least squares problem to obtain the baseline shift for each sample.
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        coef = []
        bate = []
        X_MSC = np.zeros_like(X)
        for i in range( X.shape[0]):
            reg.fit(ideal_spectrum.reshape(-1,1),X[i,:])
            # Correct the spectrum of each sample by subtracting the obtained baseline shift and then dividing by the offset to get the corrected spectrum
            X_MSC[i,:] = (X[i,:]-reg.intercept_)/reg.coef_
        return X_MSC

def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    def WhittakerSmooth(x,w,lambda_,differences=1):
        '''
        只能一个一个样本的处理，不能一次处理多个
        Penalized least squares algorithm for background fitting
        
        input
            x: input data (i.e. chromatogram of spectrum)
            w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
            lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
            differences: integer indicating the order of the difference of penalties
        
        output
            the fitted background vector
        '''
        X=np.matrix(x)
        m=X.size
        E=eye(m,format='csc')
        for i in range(differences):
            E=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
        W=diags(w,0,shape=(m,m))
        A=csc_matrix(W+(lambda_*E.T*E))
        B=csc_matrix(W*X.T)
        background=spsolve(A,B)
        return np.array(background)
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return x-z