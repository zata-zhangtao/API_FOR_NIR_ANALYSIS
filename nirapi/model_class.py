import optuna
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
from nirapi.draw import *
from sklearn.ensemble import GradientBoostingRegressor  # 修改为梯度提升树


########################################################################################
#### pipeline


# 自定义转换器类
class SNVTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return SNV(X) # 假设 SNV 操作对数据进行预处理
    






class NormalizeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        normalized_data = np.zeros_like(X)
        for i in range(len(X)):
            sample = X[i]
            min_val = np.min(sample)
            max_val = np.max(sample)
            if max_val == min_val:
                normalized_data[i] = np.zeros_like(sample)
            else:
                normalized_data[i] = (sample - min_val) / (max_val - min_val)
        return normalized_data
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

class RemoveHighMeanFeatureByRatio(BaseEstimator, TransformerMixin):
    def __init__(self, ratio_threshold=0.8):
        if not 0 <= ratio_threshold <= 1:
            raise ValueError("ratio_threshold must be between 0 and 1")
        self.ratio_threshold = ratio_threshold
        self.features_to_remove = None

    def fit(self, X, y=None):
        feature_means = np.mean(np.abs(X), axis=0)
        n_features = X.shape[1]
        n_features_to_keep = int(np.ceil(n_features * self.ratio_threshold))
        n_features_to_remove = n_features - n_features_to_keep
        if n_features_to_remove > 0:
            self.features_to_remove = np.argsort(feature_means)[-n_features_to_remove:]
        else:
            self.features_to_remove = np.array([])
        return self

    def transform(self, X):
        if len(self.features_to_remove) > 0:
            return np.delete(X, self.features_to_remove, axis=1)
        return X

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



def classify_alcohol_model_0923(X_train, X_val, y_train, y_val, n_trials=50):
    # Splitting data into train and validation sets

    def objective(trial):
        # Define the hyperparameters to be tuned
        C = trial.suggest_loguniform('C', 1e-3, 1e2)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        
        if kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 5)
        else:
            degree = 3
        
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        
        # Create a pipeline model
        model = Pipeline([
            ('SNV', SNVTransformer()),
            ('svc', SVC(C=C, kernel=kernel, degree=degree, gamma=gamma))
        ])
        
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        return accuracy_score(y_val, y_val_pred)

    # Create an Optuna study object and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print(f"Best Parameters: {study.best_params}")
    print(f"Best Validation Accuracy: {study.best_value}")

    # Retrain the model with the best parameters on the full training set
    best_params = study.best_params
    best_degree = best_params.get('degree', 3)

    best_model = Pipeline([
        ('SNV', SNVTransformer()),
        ('svc', SVC(C=best_params['C'], kernel=best_params['kernel'], degree=best_degree, gamma=best_params['gamma']))
    ])
    
    best_model.fit(X_train, y_train)
    return best_model, study.best_params, study.best_value




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline as SklearnPipeline

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import scipy.linalg





###################################################################################################################################################
#################                                   2024-10-12
###################################################################################################################################################
class EmscScaler(object):
    def __init__(self, order=1):
        self.order = order
        self._mx = None

    def mlr(self, x, y):
        """Multiple linear regression fit of the columns of matrix x
        (dependent variables) to constituent vector y (independent variables)

        order -     order of a smoothing polynomial, which can be included
                    in the set of independent variables. If order is
                    not specified, no background will be included.
        b -         fit coeffs
        f -         fit result (m x 1 column vector)
        r -         residual   (m x 1 column vector)
        """

        if self.order > 0:
            s = np.ones((len(y), 1))
            for j in range(self.order):
                s = np.concatenate((s, (np.arange(0, 1 + (1.0 / (len(y) - 1)), 1.0 / (len(y) - 1)) ** j).reshape(-1,1)[0:len(y)]),1)
            X = np.concatenate((x.reshape(-1,1), s), 1)
        else:
            X = x

        # calc fit b=fit coefficients
        b = np.dot(np.dot(scipy.linalg.pinv(np.dot(X.T, X)), X.T), y)
        f = np.dot(X, b)
        r = y - f

        return b, f, r

    def fit(self, X, y=None):
        """fit to X (get average spectrum), y is a passthrough for pipeline compatibility"""
        self._mx = np.mean(X, axis=0)

    def transform(self, X, y=None, copy=None):
        if type(self._mx) == type(None):
            print("EMSC not fit yet. run .fit method on reference spectra")
        else:
            # do fitting
            corr = np.zeros(X.shape)
            for i in range(len(X)):
                b, f, r = self.mlr(self._mx, X[i, :])
                corr[i, :] = np.reshape((r / b[0]) + self._mx, (corr.shape[1],))
            return corr

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class SpectraPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, emsc_order=2, X_ref=None, ranges=None):
        if ranges == None:
            raise ValueError("ranges can not be None!")
        self.ranges = ranges
        self.emsc_order = emsc_order
        self.emsc_scalers = [EmscScaler(order=emsc_order) for _ in range(len(ranges))]
        self.X_ref = X_ref

    def fit(self, X, y=None):
        X_ref = self.X_ref
        if X_ref is None:
            X_ref = X.copy()
        ranges = self.ranges

        # Fit EmscScaler for each segment
        for i, (start, end) in enumerate(ranges):
            self.emsc_scalers[i].fit(X_ref[:, start:end])

        return self

    def transform(self, X, y=None):
        # Define the column ranges for each segment
        ranges = self.ranges

        # Transform each segment
        transformed_segments = []
        for i, (start, end) in enumerate(ranges):
            segment = X[:, start:end]
            transformed_segment = self.emsc_scalers[i].transform(segment)
            # transformed_segment = savgol(transformed_segment, 65, poly=2, deriv=0)
            # transformed_segment = SBL_Batch(transformed_segment)
            transformed_segments.append(transformed_segment)
            print('Band %d Done...' % (i+1))

        # Concatenate all transformed segments
        return np.concatenate(transformed_segments, axis=1)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    

class CustomPipeline(SklearnPipeline):
    def fit(self, X, y=None, X_ref=None, **fit_params):
        if X_ref is not None:
            fit_params['emsc_preprocessor__X_ref'] = X_ref
        return super().fit(X, y, **fit_params)

    def fit_transform(self, X, y=None, X_ref=None, **fit_params):
        if X_ref is not None:
            fit_params['emsc_preprocessor__X_ref'] = X_ref
        return super().fit_transform(X, y, **fit_params)













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

# 自定义转换器类
class SNVTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    # 如果没有需要拟合的内容，fit 直接返回 self
    def fit(self, X, y=None):
        return self

    # transform 应用自定义的预处理操作
    def transform(self, X):
        return SNV(X)  # 假设 SNV 操作对数据进行预处理


class EmscScaler(object):
    def __init__(self, order=1):
        self.order = order
        self._mx = None

    def mlr(self, x, y):
        """Multiple linear regression fit of the columns of matrix x
        (dependent variables) to constituent vector y (independent variables)

        order -     order of a smoothing polynomial, which can be included
                    in the set of independent variables. If order is
                    not specified, no background will be included.
        b -         fit coeffs
        f -         fit result (m x 1 column vector)
        r -         residual   (m x 1 column vector)
        """

        if self.order > 0:
            s = np.ones((len(y), 1))
            for j in range(self.order):
                s = np.concatenate(
                    (s, (np.arange(0, 1 + (1.0 / (len(y) - 1)), 1.0 / (len(y) - 1)) ** j).reshape(-1, 1)[0:len(y)]), 1)
            X = np.concatenate((x.reshape(-1, 1), s), 1)
        else:
            X = x

        # calc fit b=fit coefficients
        b = np.dot(np.dot(scipy.linalg.pinv(np.dot(X.T, X)), X.T), y)
        f = np.dot(X, b)
        r = y - f

        return b, f, r

    def fit(self, X, y=None):
        """fit to X (get average spectrum), y is a passthrough for pipeline compatibility"""
        X = np.array([row - row.mean() for row in X])
        self._mx = np.mean(X, axis=0)

    def transform(self, X, y=None, copy=None):
        if type(self._mx) == type(None):
            print("EMSC not fit yet. run .fit method on reference spectra")
        else:
            # do fitting
            corr = np.zeros(X.shape)
            for i in range(len(X)):
                b, f, r = self.mlr(self._mx, X[i, :])
                corr[i, :] = np.reshape((r / b[0]) + self._mx, (corr.shape[1],))
            return corr

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


