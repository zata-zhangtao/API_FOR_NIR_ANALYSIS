'''
loading data
-------
Functions:
---------
    - split_date_time(date_time, X_train, X_val) : 划分得到数据集的时间戳列表
    - save_model(model, file_name=None) :  保存模型到指定文件
    - load_model(file_name) :  从指定文件加载模型
    - split_data_by_date(X , y , date_time,timestamp_split_point ) :  根据时间戳，分割数据集
    - split_date_time(date_time, start_timestamp = '2024-09-21',    end_timestamp = '2024-09-27 23:59:59') :  根据时间戳，分割日期索引
    - Filter_from_prototype_data_by_volunteer(file_path =r"C:\BaiduSyncdisk\code&note\0-data_analysis\0923酒精数据分析\data\MZI酒精数据_ALL.xlsx" , volunteer_name= None)
    - load_prototype_data(file_path):  加载样机的数据
    - get_data : X,y  加载光谱数据,输入X,y所在的列,如果有名字可以输入名字返回特定名字自愿者的数据
    - alcohol_1111 : X,y  默认加载11月11日的酒精数据，也可以通过设置参数加载其他数据
    - get_volunteer_name : list(str) 默认加载11月11日的酒精数据的志愿者的名字
    - get_volunteer_data : dict  默认加载11月11日的酒精数据的志愿者的数据,返回字典
    - get_waterContent_11_07 : X,y 默认加载11月07日的皮肤水分数据
    - get_feat_index_accroding_wave : list(int) 根据波长范围，返回对应的索引
    - get_wave_accroding_feat_index : list(int) 根据索引，返回对应的波长
    - get_file_list_include_name : list(str) 根据文件名所包含的字符串，返回文件列表
    - send_email_to_zhangtao() :  给我发邮件
    - Transforming_raw_xlsx_data_into_trainable_csv_data 把原始的采集的数据转成dataframe
    - save_dict_to_csv(data, csv_file, fill_value=None)
---------
Examples:
---------
    - 保存模型到指定文件 save_model(model, file_name="model.pkl")
    - 从指定文件加载模型 load_model("model.pkl")
    - 根据时间戳，分割数据集 返回X_train, X_val, X_test, y_train, y_val, y_test = split_data_by_date(X , y , date_time,['2024-09-27 23:59:59', '2024-09-29 23:59:59'])
    - 根据时间戳，返回索引
    - 根据志愿者的名字过滤样机的数据
    - 加载样机的数据 load_prototype_data("data.xlsx")
    - 加载光谱数据  get_data()  不过提前要对数据做一下处理
    - 根据志愿者名字加载数据 get_volunteer_data()
    - 加载11月11日酒精数据  alcohol_1111()
    - 加载11月11日酒精数据志愿者名字 get_volunteer_name()
    - 加载11月7日皮肤水分数据 get_waterContent_11_07()
    - 加载11月7日皮肤水分每个志愿者的数据 data = get_volunteer_data(file_path=r"D:\\Desktop\\NIR spectroscopy\\dataset\\skin_moisture_11_07.csv",col_y=1899,col_name=1900)
    - 根据波长区间，返回对应的索引 get_feat_index_accroding_wave( wave_range:list,wavelengths = None)
    - 根据索引list,返回对应的波长 get_wave_accroding_feat_index(index:list,wavelengths = None)
    - 把字典数据保存为csv文件
'''

# wavelengths = pd.read_csv(r"C:\Users\zata\AppData\Local\Programs\Python\Python310\Lib\site-packages\nirapi\Alcohol.csv").columns[:1899].values.astype("float")
from typing import Union
import pandas as pd
import numpy as np
import os
import datetime
import joblib







def get_date_time_array_for_train_val_test(date_time, X_train, X_val):
    """
    分割日期时间数组为训练、验证和测试集合的索引。

    参数:
    date_time: 完整的日期时间数组。
    len_train: 训练集的长度。
    len_val: 验证集的长度。

    返回:
    date_time_train: 对应训练集的日期时间数组部分。
    date_time_val: 对应验证集的日期时间数组部分。
    date_time_test: 对应剩余部分，通常用作测试集的日期时间数组部分。
    """
    len_train = len(X_train)
    len_val = len(X_val)
    date_time_train = date_time[:len_train]
    date_time_val = date_time[len_train:len_train + len_val]
    date_time_test = date_time[len_train + len_val:]
    
    return date_time_train, date_time_val, date_time_test



def save_model(model, file_name=None):
    """
    保存模型到指定文件。

    参数:
    model: 训练好的模型对象
    file_name: 保存模型的文件名（包括路径）
    """
    nowtime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if file_name is None:
        file_name = nowtime + "_model.pkl"
    
    # 保存模型
    joblib.dump(model, file_name)
    
    # 获取完整保存路径
    full_path = os.path.abspath(file_name)
    
    print(f"Model saved to {full_path}")


def load_model(file_name):
    """
    从指定文件加载模型。

    参数:
    file_name: 加载模型的文件名（包括路径）

    返回:
    返回加载的模型对象。
    """
    # 检查文件是否存在
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"No model file found at {file_name}")
    
    # 加载模型
    model = joblib.load(file_name)
    
    print(f"Model loaded from {file_name}")
    
    return model



def split_data_by_date_v2(X , y , date_time,timestamp_split_point,split_by_date=True):
    ''' 相较于v1版本增加了返回时间戳的功能,增加了随机划分功能， 但是随机划分需要在这里手动改test和val比例
    功能介绍： 根据时间戳，分割数据集
    example: timestamp_split_point= ['2024-09-27 23:59:59', '2024-09-29 23:59:59']
    '''
    if split_by_date == False:
        print("split by date is False, split by random")
        from sklearn.model_selection import train_test_split
        test_size=0.2
        val_size=0.25
        indices = range(len(X))
        indices_train_val, indices_test, _, _ = train_test_split(indices, y, test_size=test_size, random_state=42)
        # 分割测试集的时间戳
        date_time_test = date_time[indices_test]
        
        # 从剩余的数据中分割出验证集
        val_relative_size = val_size / (1 - test_size)  # 调整验证集的相对大小
        indices_train, indices_val, _, _ = train_test_split(indices_train_val, y[indices_train_val], test_size=val_relative_size, random_state=42)
        
        # 分割训练集和验证集的时间戳
        date_time_train = date_time[indices_train]
        date_time_val = date_time[indices_val]
        
        # 使用索引获取X和y的分割数据
        X_train, X_val, X_test = X[indices_train], X[indices_val], X[indices_test]
        y_train, y_val, y_test = y[indices_train], y[indices_val], y[indices_test]

        return X_train, X_val, X_test, y_train, y_val, y_test, date_time_train, date_time_val, date_time_test


    train_indice = split_date_time(date_time, start_timestamp = '1970-09-21',    end_timestamp =timestamp_split_point[0])
    val_indice = split_date_time(date_time, start_timestamp = timestamp_split_point[0],    end_timestamp = timestamp_split_point[1])
    test_indice = split_date_time(date_time, start_timestamp = timestamp_split_point[1],    end_timestamp = '2099-12-11 23:59:59')
    X_train = X[train_indice,:]
    X_val = X[val_indice,:]   
    X_test = X[test_indice, :]   
    y_train = y[train_indice]
    y_val = y[val_indice]   
    y_test = y[test_indice]   
    date_time_train = date_time[train_indice]
    date_time_val = date_time[val_indice]
    date_time_test = date_time[test_indice]
    
    return X_train, X_val, X_test, y_train, y_val, y_test,date_time_train,date_time_val,date_time_test


def split_data_by_date(X , y , date_time,timestamp_split_point ):
    '''
    example: timestamp_split_point= ['2024-09-27 23:59:59', '2024-09-29 23:59:59']
    '''

    train_indice = split_date_time(date_time, start_timestamp = '1970-09-21',    end_timestamp =timestamp_split_point[0])
    val_indice = split_date_time(date_time, start_timestamp = timestamp_split_point[0],    end_timestamp = timestamp_split_point[1])
    test_indice = split_date_time(date_time, start_timestamp = timestamp_split_point[1],    end_timestamp = '2099-12-11 23:59:59')
    X_train = X[train_indice,:]
    X_val = X[val_indice,:]   
    X_test = X[test_indice, :]   
    y_train = y[train_indice]
    y_val = y[val_indice]   
    y_test = y[test_indice]   
    return X_train, X_val, X_test, y_train, y_val, y_test

def split_date_time(date_time, start_timestamp = '2024-09-21',    end_timestamp = '2024-09-27 23:59:59'):
    ''' 注意，时间是精确到秒的，所以需要注意时间的格式
    date_time = ["2024-09-21 09:31:25", "2024-09-21 09:31:45", "2024-09-27 09:32:05"], start_timestamp = '2024-09-21',    end_timestamp = '2024-09-27 23:59:59'
    '''
    df = pd.DataFrame(date_time, columns=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # 筛选出9月28日之前的时间戳
    filtered_df = df[(df['timestamp'] >= start_timestamp) & (df['timestamp'] <= end_timestamp)]
    # 索引
    return  filtered_df.index.to_numpy()


def Filter_from_prototype_data_by_volunteer(data_list = None, file_path =r"C:\BaiduSyncdisk\code&note\0-data_analysis\0923酒精数据分析\data\MZI酒精数据_ALL.xlsx" , volunteer_name= None):
    '''
    返回指定志愿者的样机数据，如果志愿者名字是None，就返回所有志愿者数据
    -----------
    Params:
    ---------
        - file_path : str  文件路径
        - volunteer_name : str  志愿者名字

    ---------
    Returns:  Corrected_spectrum, label, date_time
    ---------
    '''
    if data_list is None:
        PD_Sample,PD_Source,PD_BG,Recon_Sample,Recon_Source,Corrected_spectrum,Biomark,Measured_Value,label,date_time,volunteer  = load_prototype_data(file_path,pos=None)
    else:
        PD_Sample,PD_Source,PD_BG,Recon_Sample,Recon_Source,Corrected_spectrum,Biomark,Measured_Value,label,date_time,volunteer  = data_list
    
    volunteer_names = np.unique(volunteer)

    if volunteer_name is not None:

        indices = np.where(volunteer == volunteer_name)[0]
        X = Corrected_spectrum[indices]
        date_time = Biomark[indices, 0]
        y = label[indices]
        print(X.shape, y.shape)
        return X, y, date_time
    elif volunteer_name is None:
        return Corrected_spectrum, label, date_time


def  load_prototype_data(file_path,pos=None):
    '''加载样机的数据
        -------
        Params:
        ---------
        - file_path : str  文件路径
        - pos : str  加载的位置，可以选择'PD Sample','PD Source','PD BG','Recon Sample','Recon Source','Corrected spectrum','Biomark','Measured_Value','y','date_time','volunteer'

    ---------
    Returns:  PD_Sample,PD_Source,PD_BG,Recon_Sample,Recon_Source,Corrected_spectrum,Biomark,Measured_Value,y,date_time,volunteer
    ---------
        - PD_Sample : ndarray
            PD Sample数据
        - PD Source : ndarray
            PD Source数据
        - PD BG : ndarray
            PD Background数据
        - Recon Sample : ndarray
            Recon Sample数据
        - Recon Source : ndarray
            Recon Source数据
        - Corrected spectrum : ndarray
            校正后的光谱数据
        - Biomark : ndarray
            生化数据
        - Measured_Value : ndarray
            实测值    
        - y : ndarray
            实测值
        - date_time : ndarray
            时间
        - volunteer : ndarray
            志愿者名字    
    '''

    # load data
    if pos is None:
        PD_Sample = pd.read_excel(file_path,header=0,sheet_name='PD Sample')
        PD_Source = pd.read_excel(file_path,header=0,sheet_name='PD Source')
        PD_BG = pd.read_excel(file_path,header=0,sheet_name='PD BG')
        Recon_Sample = pd.read_excel(file_path,header=0,sheet_name='Recon Sample')
        Recon_Source = pd.read_excel(file_path,header=0,sheet_name='Recon Source')
        Corrected_spectrum = pd.read_excel(file_path,header=0,sheet_name='Corrected spectrum')
        Biomark = pd.read_excel(file_path,header=0,sheet_name='Biomark')
        Measured_Value = pd.read_excel(file_path,header=0,sheet_name='Measured_Value')
        date_time = Biomark['时间']
        y = Measured_Value['实测值']
        if '志愿者' in Measured_Value.columns:
            volunteer = Measured_Value['志愿者'].values
        else:
            volunteer = None
        PD_Sample = PD_Sample.to_numpy()
        PD_Source = PD_Source.to_numpy()
        PD_BG = PD_BG.to_numpy()
        Recon_Sample = Recon_Sample.to_numpy()
        Recon_Source = Recon_Source.to_numpy()
        Corrected_spectrum = Corrected_spectrum.to_numpy()
        Biomark = Biomark.to_numpy()
        Measured_Value = Measured_Value.to_numpy()
        y = y.values
        date_time = date_time.values
        # volunteer = volunteer.values
        return PD_Sample,PD_Source,PD_BG,Recon_Sample,Recon_Source,Corrected_spectrum,Biomark,Measured_Value,y,date_time,volunteer
    elif pos == 'PD Sample':
        PD_Sample = pd.read_excel(file_path,header=0,sheet_name='PD Sample')
        return PD_Sample.to_numpy()
    elif pos == 'PD Source':
        PD_Source = pd.read_excel(file_path,header=0,sheet_name='PD Source')
        return PD_Source.to_numpy()
    elif pos == 'PD BG':
        PD_BG = pd.read_excel(file_path,header=0,sheet_name='PD BG')
        return PD_BG.to_numpy()
    elif pos == 'Recon Sample':
        Recon_Sample = pd.read_excel(file_path,header=0,sheet_name='Recon Sample')
        return Recon_Sample.to_numpy()
    elif pos == 'Recon Source':
        Recon_Source = pd.read_excel(file_path,header=0,sheet_name='Recon Source')
        return Recon_Source.to_numpy()
    elif pos == 'Corrected spectrum':
        Corrected_spectrum = pd.read_excel(file_path,header=0,sheet_name='Corrected spectrum')
        return Corrected_spectrum.to_numpy()
    elif pos == 'Biomark':
        Biomark = pd.read_excel(file_path,header=0,sheet_name='Biomark')
        return Biomark.to_numpy()
    elif pos == 'Measured_Value':
        Measured_Value = pd.read_excel(file_path,header=0,sheet_name='Measured_Value')
        return Measured_Value.to_numpy()
    elif pos == 'y':
        y = pd.read_excel(file_path,header=0,sheet_name='Measured_Value')['实测值']
        return y.values
    elif pos == 'date_time':
        date_time = pd.read_excel(file_path,header=0,sheet_name='Biomark')['时间']
        return date_time.values
    elif pos == 'volunteer':
        volunteer = pd.read_excel(file_path,header=0,sheet_name='Measured_Value')['志愿者']
        return volunteer.values
    else:
        print('pos参数输入错误')
        return None

def get_data(file_path:str,
                 name=None,
                 name_col=1902,
                 X_col = [0,1899],
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
    data = pd.read_csv(file_path)
    if name is None:
        X = data.iloc[:,X_col[0]:X_col[1]].to_numpy()
        y = data.iloc[:,y_col].to_numpy()
    else:
        index_row = data[:,name_col] == name
        X = data.iloc[index_row,X_col[0]:X_col[1]].to_numpy()
        y = data.iloc[index_row,y_col].to_numpy()
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

def get_volunteer_data(file_path = r"D:\Desktop\NIR spectroscopy\main\Features_Selection_Analysis\Alcohol.csv",col_y:Union[int,list] = 1899,
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
        - wavelengths : ndarray 波长 ，支持自己输入波长列表
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

def get_wave_list(file_path = None):
    """return wavelengths list
    -------
    Parameters:
    ---------
        - file_path : str, default = None
    ---------
    Returns:
    ---------
        - wavelengths : list
            wavelengths



    """
    import pandas as pd
    if file_path is None:
        wavelengths = pd.read_csv(r"C:\Users\zata\AppData\Local\Programs\Python\Python310\Lib\site-packages\nirapi\Alcohol.csv").columns[:1899].values.astype("float")
    else:
        wavelengths = pd.read_csv(file_path).columns.values.astype("float")
    return wavelengths

def get_file_list_include_name(file_path, name):
    """根据文件名，返回包含该名字的文件列表
    -------
    Parameters:
    ---------
        - file_path : str 文件路径
        - name : str 文件名
    ---------
    Returns:
    ---------
        - file_list : list
            file list
    ---------
    Example:
    ---------
        # a = get_file_list_include_name(r"file_path", ".py")
        # print(a)
    ---------
    """
    import os
    file_list = []
    for file in os.listdir(file_path):
        if file.find(name) != -1:
            file_list.append(file)
    return file_list

def send_email_to_zhangtao(content = "训练结束了",receivers = "1506739178@qq.com"):
    import smtplib
    from email.header import Header
    from email.mime.text import MIMEText
    def sendEmail(send_dict): 
        # 第三方 SMTP 服务
        mail_host = "smtp.163.com"      # SMTP服务器
        mail_user = "18305509246@163.com"               # 用户名
        mail_pass = "QDJUTEDFCRRTUBPY"            # 授权密码，非登录密码
        
        sender ="18305509246@163.com"   # 发件人邮箱(最好写全, 不然会失败)
        content = send_dict.content # 内容
        sender = send_dict.sender #你的邮箱账号如:18305509246@163.com
        receivers = send_dict.receivers #收件人邮箱
        title = send_dict.title # 主图
    
        message = MIMEText(content, 'plain', 'utf-8')  # 内容, 格式, 编码
        message['From'] = "{}".format(sender)
        message['To'] = receivers
        message['Subject'] = title
        try:
            smtpObj = smtplib.SMTP_SSL(mail_host, 465)  # 启用SSL发信, 端口一般是465
            smtpObj.login(mail_user, mail_pass)  # 登录验证
            smtpObj.sendmail(sender, receivers, message.as_string())  # 发送
            print("mail has been send successfully.")
        except smtplib.SMTPException as e:
            print(e)
    
    class MyDict(dict):
        def __getattribute__(self, key) :
            return self[key]
    send = MyDict(
        {
        "content":content,
        "title":"服务器",
        "receivers" :receivers,
        "sender":"18305509246@163.com",
        })
    sendEmail(send)

def Merge_all_csv(dirname = "data",include_name = ".csv"):

    import os
    import pandas as pd
    import time
    from datetime import datetime

    """打相同目录下的所有csv文件合并,
    -----
    params:
    -----
        dirname: 文件夹名称
    """
    all_files = os.listdir(dirname)
    file_list = []
    for file in all_files:
        if file.find(include_name) != -1:
            file_list.append(file)
    all_files = file_list
    all_files.sort()
    all_files = [os.path.join(dirname,i) for i in all_files]
    df = pd.concat([pd.read_csv(i) for i in all_files],axis = 0)
    now_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    df.to_csv(os.path.join(dirname, str(now_time)+'_all_results.csv'),index = False)
    # print("all_results.csv has been saved in {}".format(dirname))
    return df

def Transforming_raw_xlsx_data_into_trainable_csv_data(excel_path = "人体血糖手臂外侧",X_index = [i for i in range(1,1899+1)],y_index = [5],others1_index = [0],others2_index = [6]):
    # 2024-3-01
    """This function transforms raw xlsx data into a format suitable for training by merging specific columns from paired sheets.
    -----
    params:
    -----
        X_index (list): List of column indices for feature data.
        y_index (list): List of column indices for target variable.
        others1_index (list): List of additional column indices from the spectra sheet.
        others2_index (list): List of additional column indices from the physchem sheet.
    -----
    return (dataframe):
    -----

    """
    import pandas as pd


    # 使用pandas读取excel文件中的所有工作表名称
    xls = pd.ExcelFile(excel_path)

    # 获取所有工作表的名称
    sheet_names = xls.sheet_names

    # 准备一个字典来根据前缀对工作表名称进行分类
    spectra_sheets = {}
    physchem_sheets = {}

    # 遍历所有工作表名称，根据后缀分组
    for name in sheet_names:
        if name.endswith('_光谱'):
            prefix = name.split('_光谱')[0]
            spectra_sheets[prefix] = name
        elif name.endswith('_理化值'):
            prefix = name.split('_理化值')[0]
            physchem_sheets[prefix] = name

    # 准备一个列表来存储配对的工作表名称
    paired_sheets = []

    # 将两个字典中相同前缀的工作表名称配对
    for prefix in spectra_sheets:
        if prefix in physchem_sheets:
            paired_sheets.append((spectra_sheets[prefix], physchem_sheets[prefix]))

    # 现在paired_sheets列表中包含了所有配对的工作表名称
    ans_df = []
    for item in paired_sheets:
        volunteer_name = item[0].split('_')[0]
        spectra = pd.read_excel(excel_path,sheet_name=item[0])
        X = spectra.iloc[:,X_index]

        others1 = spectra.iloc[:,others1_index]
        biomark = pd.read_excel(excel_path,sheet_name=item[1])
        y = biomark.iloc[:,y_index]
        others2 = biomark.iloc[:,others2_index]
        others = pd.concat([others1,others2],axis=1)
        dataset_df = pd.concat([X,y,others],axis=1)
        dataset_df = dataset_df.assign(志愿者=volunteer_name)
        ans_df.append(dataset_df)
    return pd.concat(ans_df)

import csv
import numpy as np

def save_dict_to_csv(data, csv_file, fill_value=None):
    """
    将字典数据保存为CSV文件，填充不等长的列。

    参数:
    data (dict): 包含要保存的数据的字典，字典的每个键对应一个数组或列表。
    csv_file (str): 输出的CSV文件名。
    fill_value: 用于填充较短列的值，默认为None。
    """
    # 获取所有列的最大长度
    max_length = max(len(v) for v in data.values())

    # 写入CSV文件
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        # 创建 CSV 字段名（列名）
        fieldnames = list(data.keys())
        
        # 创建 CSV 写入器
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # 写入列名
        writer.writeheader()
        
        # 写入每一行数据，较短的列使用fill_value填充
        for i in range(max_length):
            row = {key: data[key][i] if i < len(data[key]) else fill_value for key in data}
            writer.writerow(row)

    print(f"数据已成功写入到 {csv_file}")




if __name__ == "__main__":
    pass