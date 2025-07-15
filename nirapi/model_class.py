"""
Model classes and transformers for NIR spectroscopy data processing.

This module provides custom transformers and model classes for NIR spectroscopy
data preprocessing and analysis, including SNV, EMSC, and optimization functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg
import optuna
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from nirapi.draw import *

__all__ = [
    'SNVTransformer',
    'NormalizeTransformer', 
    'RemoveHighMeanFeatureByRatio',
    'EmscScaler',
    'SpectraPreprocessor',
    'CustomPipeline',
    'SNV',
    'classify_alcohol_model_0923'
]


# === Core Preprocessing Functions ===

def SNV(X, replace_wave=None):
    """
    Standard Normal Variate (SNV) preprocessing.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The spectral data to be processed.
    replace_wave : list, optional
        Range [start, end] to use for baseline calculation instead of full spectrum.
        
    Returns
    -------
    X_snv : ndarray, shape (n_samples, n_features)
        The SNV-processed spectral data.
        
    Raises
    ------
    TypeError
        If X is not array-like
    ValueError
        If X is empty or has invalid dimensions
    """
    # Input validation
    if X is None:
        raise ValueError("X cannot be None")
    
    try:
        X = np.asarray(X)
    except (TypeError, ValueError):
        raise TypeError("X must be array-like")
    
    if X.size == 0:
        raise ValueError("X cannot be empty")
    
    if X.ndim == 1:
        X = np.array([X])
    elif X.ndim > 2:
        raise ValueError("X must be 1D or 2D array")
    
    # Validate replace_wave if provided
    if replace_wave is not None:
        if not isinstance(replace_wave, (list, tuple, np.ndarray)):
            raise TypeError("replace_wave must be a list, tuple, or array")
        
        if len(replace_wave) != 2:
            raise ValueError("replace_wave must contain exactly 2 elements [start, end]")
        
        start, end = replace_wave
        if not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)):
            raise TypeError("replace_wave elements must be integers")
        
        if start < 0 or end < 0:
            raise ValueError("replace_wave indices must be non-negative")
        
        if start >= end:
            raise ValueError("replace_wave start index must be less than end index")
        
        if end > X.shape[1]:
            raise ValueError(f"replace_wave end index ({end}) exceeds number of features ({X.shape[1]})")
    
    try:
        if replace_wave is not None:
            X_base_line = X[:, replace_wave[0]:replace_wave[1]] * 5
        else:
            X_base_line = X
        
        x = X
        x_snv = np.zeros_like(x)
        
        for i in range(x.shape[0]):
            mean_val = np.mean(X_base_line[i])
            std_val = np.std(X_base_line[i])
            
            if std_val == 0:
                # Handle zero standard deviation case
                x_snv[i] = np.zeros_like(x[i])
            else:
                x_snv[i] = (x[i] - mean_val) / std_val
        
        return x_snv
        
    except Exception as e:
        raise RuntimeError(f"Error during SNV processing: {str(e)}")


# === Custom Transformer Classes ===

class SNVTransformer(BaseEstimator, TransformerMixin):
    """
    Standard Normal Variate transformer for sklearn pipelines.
    """
    
    def __init__(self, replace_wave=None):
        self.replace_wave = replace_wave

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return SNV(X, self.replace_wave)


class NormalizeTransformer(BaseEstimator, TransformerMixin):
    """
    Min-Max normalization transformer for individual samples.
    """
    
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
    """
    Remove features with highest mean values based on ratio threshold.
    """
    
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
        if self.features_to_remove is not None and len(self.features_to_remove) > 0:
            return np.delete(X, self.features_to_remove, axis=1)
        return X


class EmscScaler(BaseEstimator, TransformerMixin):
    """
    Extended Multiplicative Signal Correction (EMSC) scaler.
    """
    
    def __init__(self, order=1):
        self.order = order
        self._mx = None

    def mlr(self, x, y):
        """
        Multiple linear regression fit.
        
        Parameters
        ----------
        x : array-like
            Independent variables matrix
        y : array-like
            Dependent variable vector
            
        Returns
        -------
        b : ndarray
            Fit coefficients
        f : ndarray
            Fit result
        r : ndarray
            Residual
        """
        if self.order > 0:
            s = np.ones((len(y), 1))
            for j in range(self.order):
                s = np.concatenate((s, (np.arange(0, 1 + (1.0 / (len(y) - 1)), 
                                                 1.0 / (len(y) - 1)) ** j).reshape(-1, 1)[0:len(y)]), 1)
            X = np.concatenate((x.reshape(-1, 1), s), 1)
        else:
            X = x

        # Calculate fit coefficients
        b = np.dot(np.dot(scipy.linalg.pinv(np.dot(X.T, X)), X.T), y)
        f = np.dot(X, b)
        r = y - f

        return b, f, r

    def fit(self, X, y=None):
        """Fit to X (get average spectrum)."""
        X = np.array([row - row.mean() for row in X])
        self._mx = np.mean(X, axis=0)
        return self

    def transform(self, X, y=None):
        """Apply EMSC transformation."""
        if self._mx is None:
            raise ValueError("EMSC not fit yet. Run .fit method on reference spectra")
        
        corr = np.zeros(X.shape)
        for i in range(len(X)):
            b, f, r = self.mlr(self._mx, X[i, :])
            # Ensure b is an array and has at least one element
            b = np.asarray(b)
            if len(b) == 0:
                raise ValueError("MLR coefficients are empty")
            corr[i, :] = np.reshape((r / b[0]) + self._mx, (corr.shape[1],))
        return corr

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class SpectraPreprocessor(BaseEstimator, TransformerMixin):
    """
    Spectral preprocessor for segmented EMSC processing.
    """
    
    def __init__(self, emsc_order=2, X_ref=None, ranges=None):
        if ranges is None:
            raise ValueError("ranges cannot be None!")
        self.ranges = ranges
        self.emsc_order = emsc_order
        self.emsc_scalers = [EmscScaler(order=emsc_order) for _ in range(len(ranges))]
        self.X_ref = X_ref

    def fit(self, X, y=None):
        X_ref = self.X_ref if self.X_ref is not None else X.copy()
        
        # Fit EmscScaler for each segment
        for i, (start, end) in enumerate(self.ranges):
            self.emsc_scalers[i].fit(X_ref[:, start:end])
        
        return self

    def transform(self, X, y=None):
        # Transform each segment
        transformed_segments = []
        for i, (start, end) in enumerate(self.ranges):
            segment = X[:, start:end]
            transformed_segment = self.emsc_scalers[i].transform(segment)
            transformed_segments.append(transformed_segment)
            print(f'Band {i+1} Done...')

        # Concatenate all transformed segments
        return np.concatenate(transformed_segments, axis=1)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class CustomPipeline(Pipeline):
    """
    Custom pipeline with additional X_ref parameter support.
    """
    
    def fit(self, X, y=None, X_ref=None, **fit_params):
        if X_ref is not None:
            fit_params['emsc_preprocessor__X_ref'] = X_ref
        return super().fit(X, y, **fit_params)

    def fit_transform(self, X, y=None, X_ref=None, **fit_params):
        if X_ref is not None:
            fit_params['emsc_preprocessor__X_ref'] = X_ref
        return super().fit_transform(X, y, **fit_params)


# === Model Optimization Functions ===

def classify_alcohol_model_0923(X_train, X_val, y_train, y_val, n_trials=50):
    """
    Optimize SVM classifier for alcohol classification using Optuna.
    
    Parameters
    ----------
    X_train, X_val : array-like
        Training and validation features
    y_train, y_val : array-like
        Training and validation labels
    n_trials : int, default=50
        Number of optimization trials
        
    Returns
    -------
    best_model : Pipeline
        Best model found during optimization
    best_params : dict
        Best hyperparameters
    best_value : float
        Best validation accuracy
    """
    def objective(trial):
        # Define hyperparameters to be tuned
        C = trial.suggest_loguniform('C', 1e-3, 1e2)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        
        if kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 5)
        else:
            degree = 3
        
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        
        # Create pipeline model
        model = Pipeline([
            ('SNV', SNVTransformer()),
            ('svc', SVC(C=C, kernel=kernel, degree=degree, gamma=gamma))
        ])
        
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        return accuracy_score(y_val, y_val_pred)

    # Create Optuna study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print(f"Best Parameters: {study.best_params}")
    print(f"Best Validation Accuracy: {study.best_value}")

    # Retrain model with best parameters
    best_params = study.best_params
    best_degree = best_params.get('degree', 3)

    best_model = Pipeline([
        ('SNV', SNVTransformer()),
        ('svc', SVC(C=best_params['C'], kernel=best_params['kernel'], 
                   degree=best_degree, gamma=best_params['gamma']))
    ])
    
    best_model.fit(X_train, y_train)
    return best_model, study.best_params, study.best_value