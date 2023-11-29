'''loading data
-------
Functions:
---------
    - get_data : X,y  加载光谱数据
    - alcohol_1111 : X,y  默认加载11月11日的酒精数据，也可以通过设置参数加载其他数据
    - get_volunteer_name : list(str) 默认加载11月11日的酒精数据的志愿者的名字
    - get_volunteer_data : dict  默认加载11月11日的酒精数据的志愿者的数据,返回字典
    - get_waterContent_11_07 : X,y 默认加载11月07日的皮肤水分数据
    - get_feat_index_accroding_wave : list(int) 根据波长范围，返回对应的索引
    - get_wave_accroding_feat_index : list(int) 根据索引，返回对应的波长
---------
Examples:
---------
    - 加载光谱数据  get_data()  不过提前要对数据做一下处理
    - 根据志愿者名字加载数据 get_volunteer_data()
    - 加载11月11日酒精数据  alcohol_1111()
    - 加载11月11日酒精数据志愿者名字 get_volunteer_name()
    - 加载11月7日皮肤水分数据 get_waterContent_11_07()
    - 加载11月7日皮肤水分每个志愿者的数据 data = get_volunteer_data(file_path=r"D:\\Desktop\\NIR spectroscopy\\dataset\\skin_moisture_11_07.csv",col_y=1899,col_name=1900)
    - 根据波长区间，返回对应的索引 get_feat_index_accroding_wave( wave_range:list,wavelengths = None)
    - 根据索引list，返回对应的波长 get_wave_accroding_feat_index(index:list,wavelengths = None)
'''

# wavelengths = pd.read_csv(r"C:\Users\zata\AppData\Local\Programs\Python\Python310\Lib\site-packages\nirapi\Alcohol.csv").columns[:1899].values.astype("float")
from typing import Union

def get_data(file_path:str,
                 name=None,
                 name_col=1902,
                 y_col=1899,
                 ):
    '''加载光谱数据
    -------
    Parameters:
    ---------
        - file_path : str 
        - name : str  volunteer name
        - name_col : int  volunteer name column
        - y_col : y值所在的列数
    ---------
    Returns:
    ---------
        - X : ndarray
            NIR spectral data
        - y : ndarray
            alcohol content
    
    '''
    import numpy as np
    import pandas as pd
    data = pd.read_csv(file_path).values
    if name is None:
        X = data[:,:1899]
        y = data[:,y_col].reshape(-1,1)
    else:
        index_row = data[:,name_col] == name
        X = data[index_row,:1899]
        y = data[index_row,y_col].reshape(-1,1)
    return X,y

def alcohol_1111(file_path = r"D:\Desktop\NIR spectroscopy\main\Features_Selection_Analysis\Alcohol.csv",
                 name=None,
                 name_col=1902,
                 y_col=1899,
                 ):
    '''加载11月11日的酒精数据 Loading Alcohol data for Nov. 11 
    -------
    Parameters:
    ---------
        - file_path : str 
        - name : str  volunteer name
        - name_col : int  volunteer name column
        - y_col : int  alcohol content column
    ---------
    Returns:
    ---------
        - X : ndarray
            NIR spectral data
        - y : ndarray
            alcohol content
    
    '''
    import numpy as np
    import pandas as pd
    data = pd.read_csv(file_path).values
    if name is None:
        X = data[:,:1899]
        y = data[:,y_col].reshape(-1,1)
    else:
        index_row = data[:,name_col] == name
        X = data[index_row,:1899]
        y = data[index_row,y_col].reshape(-1,1)
    return X,y

def get_volunteer_name(file_path = r"D:\Desktop\NIR spectroscopy\main\Features_Selection_Analysis\Alcohol.csv",
                       col = 1902):
    '''获取志愿者的名字
    -------
    Parameters:
    ---------
        - file_path : str 文件路径, default
        - col : int 列数 ,default 1902
    ---------
    Returns:
    ---------
        - name : list
            volunteer name
    '''
    import pandas as pd
    import numpy as np
    data = pd.read_csv(file_path).values
    name = data[:,col]
    name = np.unique(name)
    return name

def get_volunteer_data(file_path = r"D:\Desktop\NIR spectroscopy\main\Features_Selection_Analysis\Alcohol.csv",col_y = 1899,
col_name = 1902):
    '''获取志愿者的数据
    -------
    Parameters:
    ---------
        - file_path : str 文件路径, default
        - col_y : int or list(int) 列数 ,default 1899  酒精含量所在的列数,如果是list,则是列数范围，如[1899,1900],Y值的列数范围必须是连续的
        - col_name : int 列数 ,default 1902  志愿者名字所在的列数
    ---------
    Returns:
    ---------
        - dict : {volunteer_name:(X: ndarray
                            NIR spectral data,
                        y : ndarray
                            alcohol content
            )}
    '''
    import pandas as pd
    import numpy as np
    data = pd.read_csv(file_path).values
    name = data[:,col_name]
    name = np.unique(name)
    eacn_volunteer_data = {}
    for i in range(len(name)):
        index_row = data[:,col_name] == name[i]
        X = data[index_row,:1899]
        if isinstance(col_y,list):
            y = data[index_row,col_y[0]:col_y[1]+1]
        else:
            y = data[index_row,col_y].reshape(-1,1)
        eacn_volunteer_data[name[i]] = (X,y)
    return eacn_volunteer_data

def get_waterContent_11_07(file_path = r"D:\Desktop\NIR spectroscopy\api\load_data"):
    '''加载11月07日的皮肤水分数据
    -------
    Parameters:
    ---------
        - file_path : str 文件路径, default
    ---------
    Returns:
    ---------
        - X : ndarray
            NIR spectral data
        - y : ndarray
            water content
    '''
    import pandas as pd
    X = pd.read_csv(file_path+"\\Hydration_11_7_all_X.csv",index_col=0).values.astype(float)
    Y = pd.read_csv(file_path+"\\Hydration_11_7_all_Y.csv",index_col=0).values.astype(float)
    return X[:,:-1],Y

def get_feat_index_accroding_wave(wave_range:list,wavelengths = None):
    '''根据波长范围，返回对应的索引
    -------
    Parameters:
    ---------
        - wave_range : list 波长范围
        - wavelengths : ndarray 波长
    ---------
    Returns:
    ---------
        - index : list
            index of wave_range
    '''
    import pandas as pd
    if wavelengths is None:
        wavelengths = pd.read_csv(r"C:\Users\zata\AppData\Local\Programs\Python\Python310\Lib\site-packages\nirapi\Alcohol.csv").columns[:1899].values.astype("float")
    index = []
    for i in range(len(wavelengths)):
        if wavelengths[i] >= wave_range[0] and wavelengths[i] <= wave_range[1]:
            index.append(i)
    return index
    
def get_wave_accroding_feat_index(index:Union[list,int],wavelengths = None)->Union[list,int]:
    '''根据索引，返回对应的波长
    -------
    Parameters:
    ---------
        - index : list 索引
        - wavelengths : ndarray 波长 ,default None 默认加载1899维的波长数据，如果有其他波长数据，可以传入
    ---------
    Returns:
    ---------
        - wave : [list,int]
            wave of index
    '''
    import pandas as pd
    if wavelengths is None:
        wavelengths = pd.read_csv(r"C:\Users\zata\AppData\Local\Programs\Python\Python310\Lib\site-packages\nirapi\Alcohol.csv").columns[:1899].values.astype("float")
    if isinstance(index,int):
        return wavelengths[index]
    wave = []
    for i in range(len(index)):
        wave.append(wavelengths[index[i]])
    return wave
