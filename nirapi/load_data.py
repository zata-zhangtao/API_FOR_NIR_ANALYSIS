'''loading data
-------
Functions:
---------
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



def get_sqlite_file(host = '47.121.138.184',username='root', password='Zata123@',local_path = "/data" ):
    import pysftp
    from tqdm import tqdm
    import os
    remote_files = ["/sqlite/样机数据库.db", "/sqlite/光谱数据库.db"]
    local_path = local_path
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # 创建一个 tqdm 进度条
    with tqdm(total=len(remote_files), desc="Downloading files") as pbar:
        with pysftp.Connection(host=host, username=username, password=password) as srv:
            for remote_file in remote_files:
                local_file = os.path.join(local_path, os.path.basename(remote_file))
                if os.path.exists(local_file):
                    continue
                srv.get(remote_file, local_file, callback=lambda x, y: pbar.update(y))  # 更新进度条
                pbar.update(1)  # 确保进度条更新正确
    





if __name__ == "__main__":
    # a = get_file_list_include_name(r"D:\Desktop\NIR spectroscopy\main\Features_Selection_Analysis", ".py")
    # print(a)
    get_sqlite_file()