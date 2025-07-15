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
    



class conv_MatMul_recon(BaseEstimator,TransformerMixin):
    """直接乘以传输矩阵得到所谓的恢复光谱，然后样品除以光源光谱 
    效果似乎不好

    
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    
    S21 = pd.read_csv( os.path.join(current_file_dir, "S21_6波段.csv")).values
    wavelength_merged = np.linspace(1245,1750,1200)
    band_point_num = 200
    def __int__(self):
        pass
    def fit(self):
        pass
    def transform(self,pd_sample,pd_source):
        """ pd维度为(n_sample,1200)或者(1200)
        
        """

            


        def rec(PD):
            rec_PD_list = []
            for i in range(self.S21.shape[1]):
                N = self.band_point_num
                conv_matrix = np.zeros((N, N))
                for j in range(N):
                    conv_matrix[j] = np.roll(self.S21[:,i], j)
                s21_m  = conv_matrix
                pd = PD[i*self.band_point_num:(i+1)*self.band_point_num]
                rec_PD_list.append(np.dot(pd.reshape(1,200),np.linalg.inv(s21_m)).reshape(-1))
            return np.concatenate(rec_PD_list)
        

        if pd_sample.ndim == 2:
            rec_pd_sample_list = []
            rec_pd_source_list = []
            for i in range(pd_sample.shape[0]):
                # rec_list.append(rec(pd_sample[i])*20/rec(pd_source[i]))
                rec_pd_sample_list.append(rec(pd_sample[i]))
                rec_pd_source_list.append(rec(pd_source[i]))
            return np.array(rec_pd_sample_list),np.array(rec_pd_source_list)
                

        rec_PD_sample_list = rec(pd_sample)
        rec_PD_source_list = rec(pd_source)
        # print(rec_PD_sample_list.shape)
        # print(rec_PD_sample_list)
        # return rec_PD_sample_list*20/rec_PD_source_list
        return rec_PD_sample_list,rec_PD_source_list

        



    

class conv_DFT_recon(BaseEstimator,TransformerMixin):
    S21_lp_nums = [1, 1, 1, 1, 1, 1]

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_file_dir)
    
    S21 = pd.read_csv( os.path.join(current_file_dir, "S21_6波段.csv")).values
    shifter =[110, 87, 80, 110, 95, 100]
    band_point_num = 200
    wavelength = [
                np.linspace(1245, 1410, 200),  # 第1波段: 1245-1410nm
                np.linspace(1320, 1490, 200),  # 第2波段: 1320-1490nm
                np.linspace(1390, 1555, 200),  # 第3波段: 1390-1555nm
                np.linspace(1485, 1655, 200),  # 第4波段: 1485-1655nm
                np.linspace(1525, 1675, 200),  # 第5波段: 1525-1675nm
                np.linspace(1545, 1750, 200),  # 第6波段: 1545-1750nm
            ]

    wavelength_merged = np.linspace(1245,1750,1200)
    def __init__(self):
        pass
        
    def fit(self):
        return 
    def transform(self,pd_sample,pd_source):
        """进行重建
        """
        def rec(PD):
            rec_PD_list = []
            for i in range(self.S21.shape[1]):
                pd = PD[i*self.band_point_num:(i+1)*self.band_point_num]
                s21 = self.S21[:,i]
                lp_num = self.S21_lp_nums[i]

                pd_fft = np.fft.fft(pd)
                s21_fft = np.fft.fft(s21)
                lp = np.zeros(pd.shape[0])
                lp[0] = 1
                lp[1 : lp_num + 1] = 1
                lp[-lp_num:] = 1
                pd_fft_lp = pd_fft * lp
                recon = np.fft.ifft(pd_fft_lp / s21_fft)
                recon = np.where(np.real(recon) < 0, 0, recon)
                recon = np.abs(recon)
                recon = np.flip(recon)
                recon = np.roll(recon, self.shifter[i])
                rec_PD_list.append(recon)
            return np.concatenate(rec_PD_list)
        
        if pd_sample.ndim == 2:
            rec_pd_sample_list = []
            rec_pd_source_list = []
            for i in range(pd_sample.shape[0]):
                # rec_list.append(rec(pd_sample[i])*20/rec(pd_source[i]))
                rec_pd_sample_list.append(rec(pd_sample[i]))
                rec_pd_source_list.append(rec(pd_source[i]))
            return np.array(rec_pd_sample_list),np.array(rec_pd_source_list)
        rec_PD_sample_list = rec(pd_sample)
        rec_PD_source_list = rec(pd_source)
        # return rec_PD_sample_list/rec_PD_source_list
        return rec_PD_sample_list,rec_PD_source_list

    def fit_transform(self,pd_sample,pd_source):
        return self.transform(pd_sample,pd_source)

    def transform_merge(self, pd_sample, pd_source):
        """
        重建之后把相同波段的光谱强度进行合并。

        Args:
            pd_sample:  输入样本数据 (Pandas DataFrame).
            pd_source:  源数据 (Pandas DataFrame).

        Returns:
            sorted_intensities: 合并后的排序好的光谱强度值 (Numpy array).
        """

        recon = self.transform(pd_sample, pd_source)

        band_points = [self.band_point_num for i in range(len(self.S21_lp_nums))]
        band_indices = np.cumsum([0] + band_points[:-1])

        indices = [
            np.arange(start, start + points)
            for start, points in zip(band_indices, band_points)
        ]

        # 使用 numpy 数组直接累加，避免字典的开销
        # 首先确定最终数组的大小，即所有波长的数量
        all_wavelengths = np.concatenate(self.wavelength)
        unique_wavelengths = np.unique(all_wavelengths)  # 去重，确保每个波长只计算一次
        sorted_wavelengths = np.sort(unique_wavelengths) #排序波长
        num_wavelengths = len(sorted_wavelengths)
        sorted_intensities = np.zeros(num_wavelengths)

        # 创建一个波长到索引的映射，方便快速查找
        wavelength_to_index = {wl: i for i, wl in enumerate(sorted_wavelengths)}

        # 遍历每个波段的数据，累加强度值
        for i, idx in enumerate(indices):
            wavelengths = self.wavelength[i]
            intensities = recon[idx]

            # 确保波长和强度数组长度一致
            if len(wavelengths) != len(intensities):
                raise ValueError("波长和强度数组长度不一致！")

            # 累加强度值
            for j, wl in enumerate(wavelengths):
                index = wavelength_to_index[wl]  # 找到波长对应的索引
                sorted_intensities[index] += intensities[j]

        print("Sorted Wavelengths:", sorted_wavelengths.shape)
        print("Sorted Intensities:", sorted_intensities.shape)

        return sorted_intensities

    def transform_cut_inter_smooth(self,pd_sample,pd_source,left_b = [40, 50, 55, 50, 70, 70],right_b = [140, 130, 165, 140, 130, 130] ):
        """重建之后进行剪切，把两边的数据切掉
        """

        spectral_data = self.transform(pd_sample,pd_source)
        
        valid_wavelengths = []
        valid_spectral = []
        for i, points in enumerate([self.band_point_num for _ in left_b]):
            valid_spectral_data = spectral_data[i * points : (i + 1) * points]
            valid_spectral_data = valid_spectral_data[left_b[i] : right_b[i]]
            valid_spectral.append(valid_spectral_data)

            full_wavelengths = np.concatenate(self.wavelength)
            valid_wavelength = full_wavelengths[i * points : (i + 1) * points]
            valid_wavelength = valid_wavelength[left_b[i] : right_b[i]]
            valid_wavelengths.append(valid_wavelength)

        spectral = np.concatenate(valid_spectral)
        wavelength = np.concatenate(valid_wavelengths)



        sort_indices = np.argsort(wavelength)
        wavelength = wavelength[sort_indices]
        spectral = spectral[sort_indices]

        from scipy.interpolate import interp1d
        f = interp1d(wavelength,spectral,kind="linear",fill_value="extrapolate")
        spectral_interpolated = f(self.wavelength_merged)

        # smooth
        from scipy.signal import savgol_filter

        spectral_smooth = savgol_filter(spectral_interpolated,window_length=30,polyorder=3)

        return spectral_smooth




class FermentPeelVectorReLU(BaseEstimator,TransformerMixin):
    """先用字典学习学到一个字典，然后再用一个可学习的向量去学习在字典之上其他光谱的变换规律，注意是向量
    """
    rec_dic_class =  SpectralDictionaryMapper()
    rec_dft_class = conv_DFT_recon()
    wavelength_merged = np.linspace(1245,1750,1200)

    def __init__(self, W=None, enable_dic=True, enable_dft=True, dic_params=None, dft_params=None):
        # Initialize W as a learnable parameter matrix
        if W is None :
            self.W = None
        else:
            self.W = W 
            
        # Initialize optimizer
        self.optimizer = torch.optim.Adam([self.W]) if self.W is not None else None
        self.enable_dic = enable_dic
        self.enable_dft = enable_dft
        
        # Initialize sub-transformers
        self.dic_params = dic_params  # Store as attribute
        self.dft_params = dft_params  # Store as attribute
        self.rec_dic_class = SpectralDictionaryMapper(**(dic_params or {}))
        self.rec_dft_class = conv_DFT_recon(**(dft_params or {}))
        # self.rec_dic_class = SpectralDictionaryMapper(**(dic_params or {}))
        # self.rec_dft_class = conv_DFT_recon(**(dft_params or {}))
        
        # Merged wavelength range


        
    def fit(self, pd_samples, ft_spectra,epochs=100):
        """
        Fit the transformer to the training data.
        
        Parameters
        ----------
        pd_samples : array-like of shape (n_samples, n_features_proto)
            The prototype spectral data.
        ft_spectra : array-like of shape (n_samples, n_features_ft)
            The FT spectral data.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """

        # Convert data to tensors
        pd_samples = torch.tensor(pd_samples, dtype=torch.float32)
        ft_spectra = torch.tensor(ft_spectra, dtype=torch.float32)


        # Get dimensions
        n_samples, n_features_proto = pd_samples.shape
        n_features_ft = ft_spectra.shape[1]
        self.n_features_ft = n_features_ft
        from scipy.interpolate import interp1d
        # Interpolate pd_samples to match ft_spectra feature size
        pd_samples_interp = np.zeros((n_samples, n_features_ft))
        for i in range(n_samples):
            # Create interpolation function
            x_old = np.linspace(0, 1, n_features_proto)
            x_new = np.linspace(0, 1, n_features_ft)
            interp_func = interp1d(x_old, pd_samples[i], kind='linear')
            pd_samples_interp[i] = interp_func(x_new)
        
        # Convert interpolated data back to tensor
        pd_samples_interp = torch.tensor(pd_samples_interp, dtype=torch.float32)


        pd_samples = pd_samples_interp


        # Add a constant vector to pd_samples (假设常数向量为可学习的参数)
        if not hasattr(self, 'bias'):
            self.bias = torch.nn.Parameter(torch.zeros(pd_samples.shape[1], requires_grad=True))
        # pd_samples_adjusted = pd_samples + self.bias  # 每个样本加上同一个常数向量
            
        if self.enable_dic:
            self.rec_dic_class.fit(pd_samples, ft_spectra)

        # Initialize W as a learnable parameter matrix
        if self.W is None:
            # self.W = torch.nn.Parameter(torch.randn(pd_samples.shape[1], ft_spectra.shape[1], requires_grad=True))
            self.W = torch.nn.Parameter(torch.ones(ft_spectra.shape[1], requires_grad=True))
        else:
            self.W = self.W 
            
        # Initialize optimizer
        self.optimizer = torch.optim.Adam([self.W]) if self.W is not None else None
        # if self.enable_dft:
        #     self.rec_dft_class.fit()  # Assuming DFT recon doesn't need fitting
        
        
        
        # Training loop for W
        for epoch in range(epochs):


            for i in range(pd_samples.shape[0]):
                self.optimizer.zero_grad()
                

                # Forward pass
                pd_samples_i = pd_samples[i].unsqueeze(0)
                pd_sample_i = pd_samples_i+self.bias  # 使用调整后的样本


                
                # Forward pass
                rec_dic = self.rec_dic_class.transform(pd_samples_i).squeeze(0) if self.enable_dic else 0
                # rec_dft = self.rec_dft_class.transform_cut_inter_smooth(pd_samples, pd_samples)
                if isinstance(rec_dic, np.ndarray):  # Check if rec_dic is a NumPy array
                    rec_dic = torch.tensor(rec_dic, dtype=torch.float32)
                elif rec_dic == 0:  # Handle the case when enable_dic is False
                    rec_dic = torch.zeros_like(ft_spectra[i])
                # Apply W as a linear transformation
                # reconstructed = rec_dic + torch.matmul(pd_samples[i].unsqueeze(0), self.W)
                
                reconstructed = rec_dic + pd_samples[i] * self.W

                reconstructed = torch.relu(reconstructed)
                # Calculate loss (mean squared error)
                loss = torch.mean((reconstructed - ft_spectra[i]) ** 2)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

        return self
        

    def transform(self, pd_sample):
        """
        Transform the input data using the fitted transformer.
        
        Parameters
        ----------
        pd_sample : array-like of shape ( n_features_proto)
            The prototype spectral data to transform.

            
        Returns
        -------
        reconstructed_data : array-like of shape (n_samples, n_features_ft)
            The reconstructed FT spectral data.
        """

        # pd_sample = torch.tensor(pd_sample, dtype=torch.float32)
        # pd_source = torch.tensor(pd_source, dtype=torch.float32)


        def rec(pd_sample):
            rec_dic = 0
            n_features_proto = pd_sample.shape[0]
            n_features_ft = self.n_features_ft
            from scipy.interpolate import interp1d
            # Interpolate pd_samples to match ft_spectra feature size
            x_old = np.linspace(0, 1, n_features_proto)
            x_new = np.linspace(0, 1, n_features_ft)
            interp_func = interp1d(x_old, pd_sample, kind='linear')
            pd_sample_interp = interp_func(x_new)
            # Convert interpolated data back to tensor
            pd_sample = torch.tensor(pd_sample_interp, dtype=torch.float32)
            if self.enable_dic:
                rec_dic = self.rec_dic_class.transform(pd_sample.unsqueeze(0))
            # if self.enable_dft:
            #     rec_dft = self.rec_dft_class.transform_cut_inter_smooth(pd_sample, pd_source)
            
            # Combine results with weights
            if isinstance(rec_dic, np.ndarray):  # Check if rec_dic is a NumPy array
                rec_dic = torch.tensor(rec_dic, dtype=torch.float32)

            # reconstructed_data = rec_dic + torch.matmul(pd_sample.unsqueeze(0), self.W)
            reconstructed_data = rec_dic + pd_sample.unsqueeze(0) * self.W
            return reconstructed_data.detach().numpy()

        if pd_sample.ndim == 2:
            rec_pd_sample_list = []
            for i in range(pd_sample.shape[0]):
                # rec_list.append(rec(pd_sample[i])*20/rec(pd_source[i]))
                rec_pd_sample_list.append(rec(pd_sample[i]))
            return np.array(rec_pd_sample_list)
        return rec(pd_sample)



    def fit_transform(self, pd_samples, ft_spectra, pd_sample):
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        pd_samples : array-like of shape (n_samples, n_features_proto)
            The prototype spectral data.
        ft_spectra : array-like of shape (n_samples, n_features_ft)
            The FT spectral data.
        pd_sample : array-like of shape (n_samples, n_features_proto)
            The prototype spectral data to transform.
        pd_source : array-like of shape (n_samples, n_features_proto)
            The source prototype spectral data.
            
        Returns
        -------
        reconstructed_data : array-like of shape (n_samples, n_features_ft)
            The reconstructed FT spectral data.
        """
        self.fit(pd_samples, ft_spectra)
        return self.transform(pd_sample, pd_source)

    def save_model(self, filepath):
        """
        Save the model parameters to a file.
        
        Parameters
        ----------
        filepath : str
            The file path to save the model.
        """
        model_state = {
            'W': self.W,
            'enable_dic': self.enable_dic,
            'enable_dft': self.enable_dft,
            'rec_dic_class': self.rec_dic_class,
            'rec_dft_class': self.rec_dft_class
        }
        torch.save(model_state, filepath)
    
    def load_model(self, filepath):
        """
        Load the model parameters from a file.
        
        Parameters
        ----------
        filepath : str
            The file path to load the model from.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_state = torch.load(filepath)
        self.W = model_state['W']
        self.enable_dic = model_state['enable_dic']
        self.enable_dft = model_state['enable_dft']
        self.rec_dic_class = model_state['rec_dic_class']
        self.rec_dft_class = model_state['rec_dft_class']


class conv_FullDictDotMat(BaseEstimator,TransformerMixin):
    rec_dic_class = SpectralDictionaryMapper()
    rec_dft_class = conv_DFT_recon()
    wavelength_merged = np.linspace(1245,1750,1200)

    def __init__(self, W=None, enable_dic=True, enable_dft=True, dic_params=None, dft_params=None):
        # Initialize W as a learnable parameter matrix
        if W is None :
            self.W = None
        else:
            self.W = W 
            
        # Initialize optimizer
        self.optimizer = torch.optim.Adam([self.W]) if self.W is not None else None
        self.enable_dic = enable_dic
        self.enable_dft = enable_dft
        
        # Initialize sub-transformers
        self.dic_params = dic_params  # Store as attribute
        self.dft_params = dft_params  # Store as attribute
        self.rec_dic_class = SpectralDictionaryMapper(**(dic_params or {}))
        self.rec_dft_class = conv_DFT_recon(**(dft_params or {}))
        # self.rec_dic_class = SpectralDictionaryMapper(**(dic_params or {}))
        # self.rec_dft_class = conv_DFT_recon(**(dft_params or {}))
        
        # Merged wavelength range


        
    def fit(self, pd_samples, ft_spectra,epochs=100):
        """
        Fit the transformer to the training data.
        
        Parameters
        ----------
        pd_samples : array-like of shape (n_samples, n_features_proto)
            The prototype spectral data.
        ft_spectra : array-like of shape (n_samples, n_features_ft)
            The FT spectral data.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """

        # Convert data to tensors
        pd_samples = torch.tensor(pd_samples, dtype=torch.float32)
        ft_spectra = torch.tensor(ft_spectra, dtype=torch.float32)


            
        if self.enable_dic:
            self.rec_dic_class.fit(pd_samples, ft_spectra)

        # Initialize W as a learnable parameter matrix
        if self.W is None:
            self.W = torch.nn.Parameter(torch.randn(pd_samples.shape[1], ft_spectra.shape[1], requires_grad=True))
        else:
            self.W = self.W 
            
        # Initialize optimizer
        self.optimizer = torch.optim.Adam([self.W]) if self.W is not None else None

        
        
        
        # Training loop for W
        for epoch in range(epochs):


            for i in range(pd_samples.shape[0]):
                self.optimizer.zero_grad()
                

                # Forward pass
                pd_samples_i = pd_samples[i].unsqueeze(0)


                
                # Forward pass
                rec_dic = self.rec_dic_class.transform(pd_samples_i).squeeze(0) if self.enable_dic else 0
                if isinstance(rec_dic, np.ndarray):  # Check if rec_dic is a NumPy array
                    rec_dic = torch.tensor(rec_dic, dtype=torch.float32)
                elif rec_dic == 0:  # Handle the case when enable_dic is False
                    rec_dic = torch.zeros_like(ft_spectra[i])
                # Apply W as a linear transformation
                reconstructed = rec_dic + torch.matmul(pd_samples[i].unsqueeze(0), self.W)

                # Calculate loss (mean squared error)
                loss = torch.mean((reconstructed - ft_spectra[i]) ** 2)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

        return self
        

    def transform(self, pd_sample):
        """
        Transform the input data using the fitted transformer.
        
        Parameters
        ----------
        pd_sample : array-like of shape (n_samples, n_features_proto)
            The prototype spectral data to transform.
        pd_source : array-like of shape (n_samples, n_features_proto)
            The source prototype spectral data.
            
        Returns
        -------
        reconstructed_data : array-like of shape (n_samples, n_features_ft)
            The reconstructed FT spectral data.
        """

        def rec(pd_sample):
            rec_dic = 0
            pd_sample = torch.tensor(pd_sample, dtype=torch.float32)
            # pd_source = torch.tensor(pd_source, dtype=torch.float32)
            
            if self.enable_dic:
                rec_dic = self.rec_dic_class.transform(pd_sample.unsqueeze(0))
            # Combine results with weights
            if isinstance(rec_dic, np.ndarray):  # Check if rec_dic is a NumPy array
                rec_dic = torch.tensor(rec_dic, dtype=torch.float32)


            reconstructed_data = rec_dic + torch.matmul(pd_sample.unsqueeze(0), self.W)
            return reconstructed_data.detach().numpy().reshape(-1)
        if pd_sample.ndim == 2:
            rec_pd_sample_list = []
            for i in range(pd_sample.shape[0]):
                rec_pd_sample_list.append(rec(pd_sample[i]))
            return np.array(rec_pd_sample_list)
        return rec(pd_sample)
    
    def fit_transform(self, pd_samples, ft_spectra, pd_sample):
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        pd_samples : array-like of shape (n_samples, n_features_proto)
            The prototype spectral data.
        ft_spectra : array-like of shape (n_samples, n_features_ft)
            The FT spectral data.
        pd_sample : array-like of shape (n_samples, n_features_proto)
            The prototype spectral data to transform.
        pd_source : array-like of shape (n_samples, n_features_proto)
            The source prototype spectral data.
            
        Returns
        -------
        reconstructed_data : array-like of shape (n_samples, n_features_ft)
            The reconstructed FT spectral data.
        """
        self.fit(pd_samples, ft_spectra)
        return self.transform(pd_sample, pd_source)

    def save_model(self, filepath):
        """
        Save the model parameters to a file.
        
        Parameters
        ----------
        filepath : str
            The file path to save the model.
        """
        model_state = {
            'W': self.W,
            'enable_dic': self.enable_dic,
            'enable_dft': self.enable_dft,
            'rec_dic_class': self.rec_dic_class,
            'rec_dft_class': self.rec_dft_class
        }
        torch.save(model_state, filepath)
    
    def load_model(self, filepath):
        """
        Load the model parameters from a file.
        
        Parameters
        ----------
        filepath : str
            The file path to load the model from.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_state = torch.load(filepath)
        self.W = model_state['W']
        self.enable_dic = model_state['enable_dic']
        self.enable_dft = model_state['enable_dft']
        self.rec_dic_class = model_state['rec_dic_class']
        self.rec_dft_class = model_state['rec_dft_class']



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from datetime import datetime
import os



class SpectrumModelEvaluator:
    def __init__(self, output_dir="evaluation_results"):
        """
        Initialize the evaluator with an output directory for saving results
        
        Parameters:
        -----------
        output_dir : str
            Directory where plots and summary document will be saved
        """
        self.output_dir = output_dir
        self.models = {}
        self.results = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def register_models(self):
        """Register all available models with default parameters"""
        self.models = {
            'SpectralDictionary': SpectralDictionaryMapper(),
            'SpectrumTransformerNN': SpectrumTransformerByNN(),
            # 'ConvMatMul': conv_MatMul_recon(),  # Requires source data
            'ConvDFT': conv_DFT_recon(),
            'FermentPeelVector': FermentPeelVectorReLU(),
            'ConvFullDict': conv_FullDictDotMat()
        }
        
    def evaluate_models(self, X_train, y_train, X_test, y_test=None):
        """
        Evaluate all registered models on the provided data
        
        Parameters:
        -----------
        X_train : array-like
            Training prototype spectral data
        y_train : array-like
            Training FT spectral data
        X_test : array-like
            Testing prototype spectral data
        y_test : array-like, optional
            Testing FT spectral data (if available)
        """
        self.register_models()
        self.results = {}
        
        for name, model in self.models.items():
            try:
                print(f"Evaluating {name}...")
                
                # Fit the model
                if name in ['ConvMatMul', 'ConvDFT']:
                    # These models require source data, we'll skip them for now
                    continue
                else:
                    model.fit(X_train, y_train)
                
                # Predict on test set
                y_pred = model.transform(X_test)
                
                # Calculate metrics if y_test is provided
                if y_test is not None:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    self.results[name] = {
                        'model': model,
                        'y_pred': y_pred,
                        'mse': mse,
                        'r2': r2
                    }
                else:
                    self.results[name] = {
                        'model': model,
                        'y_pred': y_pred
                    }
                
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                
    def generate_plots(self, X_test, y_test=None):
        """Generate comparison plots for all evaluated models"""
        plt.figure(figsize=(15, 10))
        
        if y_test is not None:
            # Plot 1: MSE Comparison
            plt.subplot(2, 1, 1)
            model_names = list(self.results.keys())
            mses = [self.results[name]['mse'] for name in model_names]
            plt.bar(model_names, mses)
            plt.title('Mean Squared Error Comparison')
            plt.ylabel('MSE')
            plt.xticks(rotation=45)
            
            # Plot 2: R2 Score Comparison
            plt.subplot(2, 1, 2)
            r2s = [self.results[name]['r2'] for name in model_names]
            plt.bar(model_names, r2s)
            plt.title('R2 Score Comparison')
            plt.ylabel('R2 Score')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(plot_path)
            plt.close()
            
            # Generate prediction vs actual plots
            for name in self.results:
                plt.figure(figsize=(10, 6))
                plt.plot(self.results[name]['y_pred'][0], label='Predicted')
                plt.plot(y_test[0], label='Actual')
                plt.title(f'{name} - Predicted vs Actual Spectrum')
                plt.legend()
                plt_path = os.path.join(self.output_dir, f'{name}_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                plt.savefig(plt_path)
                plt.close()
        else:
            # Plot transformed spectra when y_test is not available
            for name in self.results:
                plt.figure(figsize=(10, 6))
                plt.plot(X_test[0], label='Original X_test')
                plt.plot(self.results[name]['y_pred'][0], label='Transformed')
                plt.title(f'{name} - Original vs Transformed Spectrum')
                plt.legend()
                plt_path = os.path.join(self.output_dir, f'{name}_transformed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                plt.savefig(plt_path)
                plt.close()
                
            # Plot all transformed spectra together
            plt.figure(figsize=(15, 8))
            plt.plot(X_test[0], label='Original X_test', alpha=0.5)
            for name in self.results:
                plt.plot(self.results[name]['y_pred'][0], label=name, alpha=0.7)
            plt.title('All Models - Transformed Spectra Comparison')
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, f'all_transformed_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(plot_path)
            plt.close()
            
        return plot_path
    
    def generate_summary_doc(self, y_test=None):
        """Generate a summary document with results and recommendations"""
        doc_path = os.path.join(self.output_dir, f'evaluation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        with open(doc_path, 'w') as f:
            f.write("Spectral Model Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            
            if y_test is not None:
                f.write("Evaluation Metrics:\n")
                f.write("-" * 20 + "\n")
                for name in self.results:
                    f.write(f"Model: {name}\n")
                    f.write(f"MSE: {self.results[name]['mse']:.6f}\n")
                    f.write(f"R2 Score: {self.results[name]['r2']:.6f}\n")
                    f.write("\n")
                
                # Find best model based on R2 score
                best_model = max(self.results.items(), key=lambda x: x[1]['r2'])[0]
                f.write("Recommendation:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Best performing model: {best_model}\n")
                f.write(f"With R2 Score: {self.results[best_model]['r2']:.6f}\n")
                f.write(f"With MSE: {self.results[best_model]['mse']:.6f}\n")
            else:
                f.write("Note: No y_test provided, showing transformation results only\n")
                f.write("-" * 20 + "\n")
                for name in self.results:
                    f.write(f"Model: {name}\n")
                    f.write("Transformation completed successfully\n")
                    f.write("\n")
                f.write("Recommendation:\n")
                f.write("-" * 20 + "\n")
                f.write("Please examine the generated plots to visually assess transformation quality\n")
            
            f.write("\nNotes:\n")
            if y_test is not None:
                f.write("- Higher R2 Score (closer to 1) indicates better fit\n")
                f.write("- Lower MSE indicates better prediction accuracy\n")
            else:
                f.write("- Plots show original vs transformed spectra for visual comparison\n")
            
        return doc_path
    
    def run_evaluation(self, X_train, y_train, X_test, y_test=None):
        """
        Run complete evaluation pipeline
        
        Returns:
        --------
        dict: Results containing plot paths and summary document path
        """
        self.evaluate_models(X_train, y_train, X_test, y_test)
        plot_path = self.generate_plots(X_test, y_test)
        doc_path = self.generate_summary_doc(y_test)
        
        return {
            'plot_path': plot_path,
            'doc_path': doc_path,
            'results': self.results
        }




if __name__ == "__main__":


    # Generate dummy data for demonstration
    np.random.seed(42)
    X_train = np.random.rand(10, 1200)  # 100 samples, 1200 features
    y_train = np.random.rand(10, 1200)
    X_test = np.random.rand(2, 1200)
    y_test = np.random.rand(2, 1200)
    
    evaluator = SpectrumModelEvaluator_v2()
    results = evaluator.run_evaluation(X_train, y_train, X_test, None)
    
    print(f"Plots saved at: {results['plot_path']}")
    print(f"Summary document saved at: {results['doc_path']}")


    exit()
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