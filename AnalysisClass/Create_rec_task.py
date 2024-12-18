import numpy as np
import json
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class BaseClass:
    def __init__(self):
        pass

    def fit(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def transform(self):
        raise NotImplementedError("Subclass must implement abstract method")
        



class SpectralDictionaryMapper(BaseEstimator, TransformerMixin):
    """
    A transformer that learns dictionaries for prototype and FT data and creates a mapping between them.
    
    Parameters
    ----------
    n_components : int, default=100
        Number of dictionary atoms to learn
    n_nonzero_coefs : int, default=10
        Number of nonzero coefficients in sparse coding
    alpha : float, default=1
        Sparsity controlling parameter in dictionary learning
    max_iter : int, default=1000
        Maximum number of iterations for dictionary learning
    random_state : int, default=42
        Random state for reproducibility
    """
    
    def __init__(self, n_components=100, n_nonzero_coefs=10, alpha=1, max_iter=1000, random_state=42):
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        
        # These will be learned during fitting
        self.dictionary_ft_ = None
        self.dictionary_proto_ = None
        self.mapping_matrix_ = None
    
    def fit(self, X, y):
        """
        Fit the dictionaries and learn the mapping between them.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_proto)
            The prototype spectral data
        y : array-like of shape (n_samples, n_features_ft)
            The FT spectral data
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Train dictionary for prototype data
        dict_learner_proto = DictionaryLearning(
            n_components=self.n_components,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        dict_learner_proto.fit(X)
        self.dictionary_proto_ = dict_learner_proto.components_
        
        # Train dictionary for FT data
        dict_learner_ft = DictionaryLearning(
            n_components=self.n_components,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        dict_learner_ft.fit(y)
        self.dictionary_ft_ = dict_learner_ft.components_
        
        # Calculate sparse codes for both datasets
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)
        
        # Get sparse codes for prototype data
        proto_coefs = []
        for i in range(X.shape[0]):
            omp.fit(self.dictionary_proto_.T, X[i])
            proto_coefs.append(omp.coef_)
        proto_coefs = np.array(proto_coefs)
        
        # Get sparse codes for FT data
        ft_coefs = []
        for i in range(y.shape[0]):
            omp.fit(self.dictionary_ft_.T, y[i])
            ft_coefs.append(omp.coef_)
        ft_coefs = np.array(ft_coefs)
        
        # Learn mapping matrix
        self.mapping_matrix_ = np.linalg.lstsq(proto_coefs, ft_coefs, rcond=None)[0]
        
        return self
    
    def transform(self, X):
        """
        Transform prototype data to FT data space using learned dictionaries and mapping.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_proto)
            The prototype spectral data to transform
            
        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features_ft)
            The transformed data in FT space
        """
        # Check if fit has been called
        if self.dictionary_proto_ is None or self.dictionary_ft_ is None or self.mapping_matrix_ is None:
            raise ValueError("This SpectralDictionaryMapper instance is not fitted yet. "
                           "Call 'fit' with appropriate arguments before using this transformer.")
        
        # Calculate sparse codes for input prototype data
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)
        proto_coefs = []
        
        for i in range(X.shape[0]):
            omp.fit(self.dictionary_proto_.T, X[i])
            proto_coefs.append(omp.coef_)
        proto_coefs = np.array(proto_coefs)
        
        # Map to FT coefficient space
        predicted_ft_coefs = np.dot(proto_coefs, self.mapping_matrix_)
        
        # Reconstruct FT data
        reconstructed_ft = np.dot(predicted_ft_coefs, self.dictionary_ft_)
        
        return reconstructed_ft
    
    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_proto)
            The prototype spectral data to transform
        y : array-like of shape (n_samples, n_features_ft)
            The FT spectral data
            
        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features_ft)
            The transformed data in FT space
        """
        return self.fit(X, y).transform(X)
    
    
    
    def save_model(self, filepath):
        """
        保存模型参数到文件
        
        Parameters
        ----------
        filepath : str
            保存模型的文件路径（.npz格式）
        """
        if not self.dictionary_proto_ is None:


            np.savez(filepath,
                    dictionary_proto=self.dictionary_proto_,
                    dictionary_ft=self.dictionary_ft_,
                    mapping_matrix=self.mapping_matrix_,
                    n_components=self.n_components,
                    n_nonzero_coefs=self.n_nonzero_coefs,
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    random_state=self.random_state)
        else:
            raise ValueError("模型还未训练，请先调用fit方法")

    def load_model(self, filepath):
        """
        从文件加载模型参数
        
        Parameters
        ----------
        filepath : str
            模型文件的路径（.npz格式）
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到模型文件: {filepath}")
            
        loaded = np.load(filepath)
        self.dictionary_proto_ = loaded['dictionary_proto']
        self.dictionary_ft_ = loaded['dictionary_ft']
        self.mapping_matrix_ = loaded['mapping_matrix']
        self.n_components = int(loaded['n_components'])
        self.n_nonzero_coefs = int(loaded['n_nonzero_coefs'])
        self.alpha = float(loaded['alpha'])
        self.max_iter = int(loaded['max_iter'])
        self.random_state = int(loaded['random_state'])




class SpectrumTransformerByNN(BaseEstimator, TransformerMixin):
    class MappingDataset(Dataset):
        def __init__(self, input_data, target_data):
            self.input_data = torch.FloatTensor(input_data)
            self.target_data = torch.FloatTensor(target_data)
        
        def __len__(self):
            return len(self.input_data)
        
        def __getitem__(self, idx):
            return self.input_data[idx], self.target_data[idx]

    class DimensionMapper(nn.Module):
        def __init__(self, input_dim, hidden_dim=1200, output_dim=None):
            super(SpectrumTransformerByNN.DimensionMapper, self).__init__()

            if output_dim is None:
                output_dim = input_dim
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, x):
            return self.model(x)
            
    def __init__(self, epochs=100, batch_size=64, learning_rate=0.001, hidden_dim=1200, model_path=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.normalization_params = None
        
    def save_model(self, filepath):
        """
        保存模型参数和标准化参数到文件
        
        Parameters
        ----------
        filepath : str
            保存模型的文件路径
        """
        if self.model is None:
            raise ValueError("模型还未训练，请先调用fit方法")
            
        # 保存模型状态
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'prototype_mean': self.prototype_mean,
            'prototype_std': self.prototype_std,
            'ft_mean': self.ft_mean,
            'ft_std': self.ft_std,
            'hidden_dim': self.hidden_dim,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }
        torch.save(model_state, filepath)
        
    def load_model(self, filepath):
        """
        从文件加载模型参数和标准化参数
        
        Parameters
        ----------
        filepath : str
            模型文件的路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到模型文件: {filepath}")
            
        # 加载模型状态
        model_state = torch.load(filepath)
        
        # 恢复标准化参数
        self.prototype_mean = model_state['prototype_mean']
        self.prototype_std = model_state['prototype_std']
        self.ft_mean = model_state['ft_mean']
        self.ft_std = model_state['ft_std']
        
        # 恢复模型超参数
        self.hidden_dim = model_state['hidden_dim']
        self.epochs = model_state['epochs']
        self.batch_size = model_state['batch_size']
        self.learning_rate = model_state['learning_rate']
        
        # 重新创建模型并加载参数
        input_dim = self.prototype_mean.shape[0]
        output_dim = self.ft_mean.shape[0]
        self.model = self.DimensionMapper(input_dim=input_dim, hidden_dim=self.hidden_dim, output_dim=output_dim).to(self.device)
        self.model.load_state_dict(model_state['model_state_dict'])
        
    def fit(self, X, y):
        """
        Fit the transformer to the training data
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input PD spectra
        y : array-like of shape (n_samples, n_target_features)
            Target FT spectra
        """
        # 计算标准化参数
        self.prototype_mean = np.mean(X, axis=0)
        self.prototype_std = np.std(X, axis=0)
        self.ft_mean = np.mean(y, axis=0)
        self.ft_std = np.std(y, axis=0)
        
        # 标准化数据
        X_normalized = (X - self.prototype_mean) / (self.prototype_std + 1e-8)
        y_normalized = (y - self.ft_mean) / (self.ft_std + 1e-8)
        
        # 准备数据集
        dataset = self.MappingDataset(X_normalized, y_normalized)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 初始化或加载模型
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)
        else:
            self.model = self.DimensionMapper(input_dim=input_dim, hidden_dim=self.hidden_dim, output_dim=output_dim).to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # 训练循环
            for epoch in range(self.epochs):
                total_loss = 0
                for batch_input, batch_target in dataloader:
                    batch_input = batch_input.to(self.device)
                    batch_target = batch_target.to(self.device)
                    
                    outputs = self.model(batch_input)
                    loss = criterion(outputs, batch_target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(dataloader):.6f}')
            
            # 保存模型
            if self.model_path:
                self.save_model(self.model_path)
        
        return self
    
    def transform(self, X):
        """
        Transform the input data using the fitted transformer
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input PD spectra to transform
            
        Returns:
        --------
        array-like of shape (n_samples, n_target_features)
            Transformed FT spectra
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        
        # 标准化输入数据
        X_normalized = (X - self.prototype_mean) / (self.prototype_std + 1e-8)
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        
        # 推理
        self.model.eval()
        with torch.no_grad():
            output_normalized = self.model(X_tensor)
        
        # 反标准化输出数据
        output_data = output_normalized.cpu().numpy() * (self.ft_std + 1e-8) + self.ft_mean
        
        return output_data
    


if __name__ == "__main__":
    n_samples = 100
    n_features_proto = 800
    n_features_ft = 614
    
    # 生成原始PD光谱数据(模拟高斯分布)
    prototypedata = np.random.normal(0, 1, (n_samples, n_features_proto))
    
    # 生成对应的FT光谱数据(添加一些非线性变换)
    FTdata = np.sin(prototypedata[:, :n_features_ft]) + 0.5 * np.random.normal(0, 0.1, (n_samples, n_features_ft))
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(prototypedata, FTdata, test_size=0.2, random_state=42)


    selected_function = '神经网络' # '字典学习' or '神经网络'

    if selected_function == '神经网络':
        # 创建和训练转换器
        transformer = SpectrumTransformerByNN()
        transformer.fit(X_train, y_train)

        # 转换测试数据
        new_data = transformer.transform(X_test)

        
    if selected_function == '字典学习':
        transformer = SpectralDictionaryMapper()
        transformer.fit(X_train, y_train)
        new_data = transformer.transform(X_test)

    # 画图
    plt.plot(new_data[0], label='PD Reconstructed Spectrum', color='blue')
    plt.plot(y_test[0], label='Original Spectrum', color='black')
    plt.legend()
    plt.show()