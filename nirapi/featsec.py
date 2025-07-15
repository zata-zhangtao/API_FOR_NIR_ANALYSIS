"""
Feature selection module for NIR spectroscopy.

This module provides various feature selection methods including:
- Filter methods (correlation-based, chi-squared, etc.)
- Wrapper methods (recursive feature elimination, etc.)
- Embedded methods (regularization-based, etc.)

Classes:
    - filter: Filter-based feature selection methods
    - wrapper: Wrapper-based feature selection methods
"""

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score
import optuna

__all__ = [
    'filter',
    'wrapper'
]
class filter:
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    def pearson(self,output_categories = "feat",threshold = 0.75):
        """ ! The output is abs of pearson coefficient if  importance is selected
        -----
        params:
        -----
            output_categories: "feat" "importance"  [ps: The output is features or importance of features]
            threshold: if "feat" is seleted , select fetatures number accroding threshold
        -----
        
        """
        X_train = self.X_train
        X_test = self.X_test
        pearson_importance = []
        for i in range(X_train.shape[1]):  # 遍历所有特征
            corr = np.corrcoef(X_train[:, i], self.y_train)[0, 1]
            pearson_importance.append(corr)
        pearson_importance = abs(np.array(pearson_importance)) 
        if output_categories == "importance":
            return pearson_importance
        elif output_categories == "feat":
            threshold_num = int(X_train.shape[1]*threshold)
            min_value_of_importance = sorted(pearson_importance)[threshold_num-1 if threshold_num >0 else 0]
            Index = np.where(pearson_importance >= min_value_of_importance)
            X_train_new = X_train[:, Index]
            X_test_new = X_test[:, Index]
            return X_train_new,X_test_new,self.y_train,self.y_test
        else:
            raise("似乎参数有错")
    def spearman(self,output_categories = "feat",threshold = 0.75):
        """
        -----
        params:
        -----
            output_categories: "feat" "importance"  [ps: The output is features or importance of features]
            threshold: if "feat" is seleted , select fetatures number accroding threshold

        """
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        spearman_importance = []
        for i in range(X_train.shape[1]):  # 遍历所有特征
            corr, _ = spearmanr(X_train[:, i], y_train)
            spearman_importance.append(corr)
        spearman_importance = abs(np.array(spearman_importance))
        if output_categories == "importance":
            return spearman_importance
        elif output_categories =="feat":
            threshold_num = int(X_train.shape[1]*threshold)
            min_value_of_importance = sorted(spearman_importance)[threshold_num-1 if threshold_num >0 else 0]
            Index = np.where(spearman_importance >= min_value_of_importance)
            X_train_new = X_train[:, Index]
            X_test_new = X_test[:, Index]
            return X_train_new,X_test_new,self.y_train,self.y_test
        else:
            raise("似乎参数有错")
    def chi_2(self,output_categories = "feat",threshold = 0.75):
        """ ! The chi-square test is only applicable to classification tasks.
        -----
        params:
        -----
            output_categories: "feat" "importance"  [ps: The output is features or importance of features]
            threshold: if "feat" is seleted , select fetatures number accroding threshold

        """
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        chi2_importance,_ = chi2(X_train, y_train)
        if output_categories == "importance":
            return chi2_importance
        elif output_categories =="feat":
            threshold_num = int(X_train.shape[1]*threshold)
            min_value_of_importance = sorted(chi2_importance)[threshold_num-1 if threshold_num >0 else 0]
            Index = np.where(chi2_importance >= min_value_of_importance)
            X_train_new = X_train[:, Index]
            X_test_new = X_test[:, Index]
            return X_train_new,X_test_new,self.y_train,self.y_test
        else:
            raise("似乎参数有错")
class wrapper:
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    def random_forest(self,Classify=True,output_categories = "feat",threshold = 0.75):
        """
        -----
        params:
        -----
            Classify: if it is classification task , then "true"
        """
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        X_train,X_vaild,y_train,y_vaild = train_test_split(X_train,y_train,test_size=0.33,random_state=42)

        if Classify:
            fau = RandomForestClassifier
        else:
            fau = RandomForestRegressor
        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 100, 1000)
            max_depth = trial.suggest_int("max_depth", 2, 50)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            max_features = trial.suggest_float("max_features", 0.001,1.0, log=True)
            
            clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf, max_features=max_features)
            clf.fit(X_train, y_train)
            y_vaild_pred = clf.predict(X_vaild)
            acc = accuracy_score(y_vaild_pred,y_vaild)
            return acc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        last  = fau(**study.best_params)
        last.fit(X_train, y_train)
        if output_categories == "importance":
            return last.feature_importances_
        elif output_categories == "feat":  
            threshold_num = int(X_train.shape[1]*threshold)
            min_value_of_importance = sorted(last.feature_importances_)[threshold_num-1 if threshold_num >0 else 0]
            Index = np.where(last.feature_importances_ >= min_value_of_importance)
            X_train_new = X_train[:, Index]
            X_test_new = X_test[:, Index]
            return X_train_new,X_test_new,self.y_train,self.y_test


if __name__ == "__main__":
    from nirapi.load_data import *
    X,y = get_waterContent_11_07()
    y = y.ravel()
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    wp = wrapper(X_train,X_test,y_train,y_test)
    wp.random_forest(Classify = False)