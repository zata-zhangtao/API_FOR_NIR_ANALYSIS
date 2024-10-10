from nirapi.preprocessing import SNV
import optuna
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nirapi.preprocessing import SNV
from sklearn.base import TransformerMixin, BaseEstimator
# 自定义转换器类
class SNVTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return SNV(X) # 假设 SNV 操作对数据进行预处理
    




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
