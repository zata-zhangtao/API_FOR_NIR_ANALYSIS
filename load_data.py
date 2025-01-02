'''
loading data
-------
Functions:
---------
    - create_connection_for_Guangyin_database( database:str,host:str=GUANGYIN_DATABASE_IP, port:int=53306, user:str='root', password:str='Guangyin88888888@',charset:str='utf8mb4'): 创建与Guangyin数据库的连接
    - insert_prototype_data_to_mysql(connection:object,   table_name:str,  PD样品:list, PD光源:Union[list,None], PD背景:Union[list,None], 重建样品:Union[list,None], 重建光源:Union[list,None], 重建样品扣背景:Union[list,None], 项目名称:str, 项目类型:str, 采集部位:Union[str,None], 采集日期:str, 志愿者:Union[str,None], 理化值:dict, 创建时间:Union[str,datetime.datetime],备注信息:Union[str,None]=None, 是否删除:Union[int,None]=None, 删除时间:Union[datetime.datetime,None]=None): 没有返回值
    - insert_spectrum_data_to_mysql(table_name:str,   光谱:list,  项目名称:str, 项目类型:str,  采集日期:str,理化值:dict,创建时间:str,光谱类型:str=None,采集部位:str=None, 志愿者:str=None,是否删除:int=None, 删除时间:str=None):没有返回值
    - get_data_from_mysql(sql): return data
    - get_dataset_from_mysql(sql): return dataset
    - add_alcoholXlsxData_to_GuangyinDatabase(file_path): 没有返回值
    - sort_by_date(data_time) :  return(sorted_datetime, *sorted_data_arrays)
    - datetime_to_timestamp(data_time) :  将日期时间字符串的NumPy数组转换为时间戳（以秒为单位）的NumPy数组
    - split_date_time(date_time, X_train, X_val) : 划分得到数据集的时间戳列表
    - save_model(model, file_name=None) :  保存模型到指定文件
    - load_model(file_name) :  从指定文件加载模型
    - split_data_by_date(X , y , date_time,timestamp_split_point ) :  根据时间戳，分割数据集
    - split_date_time(date_time, start_timestamp = '2024-09-21',    end_timestamp = '2024-09-27 23:59:59') :  根据时间戳，分割日期索引
    - Filter_from_prototype_data_by_volunteer(file_path =r"C:\BaiduSyncdisk\code&note\0-data_analysis\0923酒精数据分析\data\MZI酒精数据_ALL.xlsx" , volunteer_name= None)
    - load_prototype_data(file_path):  加载样机的数据
    - get_data : X,y  加载光谱数据,输入X,y所在的列,如果有名字可以输入名字返回特定名字自愿者的数据
    - get_feat_index_accroding_wave : list(int) 根据波长范围，返回对应的索引
    - get_wave_accroding_feat_index : list(int) 根据索引，返回对应的波长
    - get_file_list_include_name : list(str) 根据文件名所包含的字符串，返回文件列表
    - send_email_to_zhangtao() :  给我发邮件
    - Transforming_raw_xlsx_data_into_trainable_csv_data 把原始的采集的数据转成dataframe
    - save_dict_to_csv(data, csv_file, fill_value=None)

---------
Examples:
---------
    - 创建与Guangyin数据库的连接 create_connection_for_Guangyin_database( database:str,host:str=GUANGYIN_DATABASE_IP, port:int=53306, user:str='root', password:str='Guangyin88888888@',charset:str='utf8mb4')
    - 向mysql数据库中插入台式光谱仪光谱数据 insert_spectrum_data_to_mysql(table_name:str,   光谱:list,  项目名称:str, 项目类型:str,  采集日期:str,理化值:dict,创建时间:str,光谱类型:str=None,采集部位:str=None, 志愿者:str=None,是否删除:int=None, 删除时间:str=None)
    - 从mysql数据库中获取字典数据 get_data_from_mysql(sql)
    - 从光引mysql数据库中获取数据集 get_dataset_from_mysql(sql)
    - 把样机采集得到的xlsx数据插入到数据库里 add_alcoholXlsxData_to_GuangyinDatabase(file_path)
    - 根据 datetime_array 排序其他数据数组，并返回排序后的结果 sort_by_date(data_time)
    - 将日期时间字符串的NumPy数组转换为时间戳（以秒为单位）的NumPy数组 datetime_to_timestamp(data_time)
    - 保存模型到指定文件 save_model(model, file_name="model.pkl")
    - 从指定文件加载模型 load_model("model.pkl")
    - 根据时间戳，分割数据集 返回X_train, X_val, X_test, y_train, y_val, y_test = split_data_by_date(X , y , date_time,['2024-09-27 23:59:59', '2024-09-29 23:59:59'])
    - 根据时间戳，返回对应时间范围在所有数据中的索引  split_date_time(date_time, start_timestamp = '2024-09-21',    end_timestamp = '2024-09-27 23:59:59'): return index_list
    - 根据志愿者的名字过滤样机的数据
    - 加载样机的数据 load_prototype_data("data.xlsx")
    - 加载光谱数据  get_data()  不过提前要对数据做一下处理
    - 根据波长区间，返回对应的索引 get_feat_index_accroding_wave( wave_range:list,wavelengths = None)
    - 根据索引list——返回对应的波长 get_wave_accroding_feat_index(index:list,wavelengths = None)
    - 把字典数据保存为csv文件
'''

# wavelengths = pd.read_csv(r"C:\Users\zata\AppData\Local\Programs\Python\Python310\Lib\site-packages\nirapi\Alcohol.csv").columns[:1899].values.astype("float")
from typing import Union
import pandas as pd
import numpy as np
import os
import datetime
import joblib
import requests
import os
from tqdm import tqdm
import time
import pymysql
import json

# 定义数据库IP常量
GUANGYIN_DATABASE_IP: str = '192.168.110.150'

#############################################################################################################################################################################################
######################################################   数据库的相关操作
#############################################################################################################################################################################################

def get_dataset_by_indices(dataset, indices):
    """ first create in 12-19
    get dataset by indices
    参数:
        dataset (dict): original dataset
        indices (tuple): indices of dataset
    返回:
        dict: new dataset
    """
    new_dataset = {}
    for key in dataset.keys():
        new_dataset[key] = dataset[key][indices[0]]
    return new_dataset


def load_alcohol_data_for_volunteer(volunteer, condition):
    train_start_time = condition['train_start_time']
    train_end_time = condition['train_end_time']
    val_start_time = condition['val_start_time']
    val_end_time = condition['val_end_time']
    test_start_time = condition['test_start_time']
    test_end_time = condition['test_end_time']

    data_split_timestrip = [
        (train_start_time, train_end_time),
        (val_start_time, val_end_time),
        (test_start_time, test_end_time)
    ]

    data_list = []
    for start_time, end_time in data_split_timestrip:
        sql = f"SELECT 志愿者, PD样品, 理化值, 创建时间 FROM `样机数据库`.`样机_计算式_MZI_v2` " \
              f"WHERE `志愿者` = '{volunteer}' AND `项目名称` = '2024人体酒精数据_样机芯片2' " \
              f"AND `创建时间` BETWEEN '{start_time}' AND '{end_time} 23:59:59'"
        data = get_data_from_mysql(sql)
        pd_sample = np.array([json.loads(i) for i in data['PD样品'].values])
        label = np.array([json.loads(i)['是否饮酒'] for i in data['理化值'].values])
        created_time = data['创建时间'].values
        data_list.append((pd_sample, label, created_time))
        # data_list.append((pd_sample, label,))

    return data_list[0], data_list[1], data_list[2]


def create_connection_for_Guangyin_database(database:str,host:str=GUANGYIN_DATABASE_IP, port:int=53306, user:str='select_user1', password:str='select_user1',charset:str='utf8mb4',dict = False):
    """
    创建与Guangyin数据库的连接。
    -----
    parameters:
    -----
    :param database: 数据库名称
    :param host: 数据库主机地址
    :param port: 数据库端口
    :param user: 数据库用户名
    :param password: 数据库密码
    :param charset: 数据库字符集
    :param dict: 是否返回字典形式的结果
    :return: 数据库连接对象
    
    -----
    example:
    -----
    create_connection_for_Guangyin_database(database='样机数据库',host='47.121.138.184', port= 1001, user='select_user1', password='select_user1',charset='utf8mb4')

    """

    try:
        if dict:
            connection = pymysql.connect(
                host=host,  # 你的数据库主机地址
                port=port,  # 你的数据库端口
                user=user,  # 你的数据库用户名
                password=password,  # 你的数据库密码
                database=database,  # 你的数据库名
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
        else:
            # 建立数据库连接
            connection = pymysql.connect(
                host=host,  # 你的数据库主机地址
                port=port,  # 你的数据库端口
                user=user,  # 你的数据库用户名
                password=password,  # 你的数据库密码
                database=database,  # 你的数据库名
                charset='utf8mb4',
                # cursorclass=pymysql.cursors.DictCursor
            )
        return connection
    except pymysql.MySQLError as e:
        print(f"数据库连接失败: {e}")
        return None

def update_spectrum_column(connection:object,  machine_name:str, spectrum:list):
    """
    更新指定样机的波段信息。

    :param host: 数据库主机地址
    :param port: 数据库端口
    :param user: 数据库用户名
    :param password: 数据库密码
    :param database: 数据库名称
    :param table_name: 表名
    :param machine_name: 样机名称
    :param spectrum: 需要更新的波段信息（DataFrame类型）
    """
    try:
        # 连接数据库

            
            # 创建游标并执行SQL语句
            with connection.cursor() as cursor:
                # 将波段信息转化为JSON格式字符串
                band_info = json.dumps(spectrum.columns.tolist(), ensure_ascii=False)
                sql = f"""UPDATE 样机信息 SET 波段 = %s WHERE 样机名称 = %s"""
                
                # 打印SQL语句用于调试
                print(sql)
                
                # 执行SQL并提交更改
                cursor.execute(sql, ( machine_name))
                connection.commit()
                
                print("波段信息更新成功！")
    
    except Exception as e:
        # 输出异常信息
        print(f"更新波段信息失败: {e}")

def insert_prototype_data_to_mysql(connection:pymysql.Connection,   table_name:str,  PD样品:list, PD光源:Union[list,None], PD背景:Union[list,None], 重建样品:Union[list,None], 重建光源:Union[list,None], 重建样品扣背景:Union[list,None], 项目名称:str, 项目类型:str, 采集部位:Union[str,None], 采集日期:str, 志愿者:Union[str,None], 理化值:dict, 创建时间:Union[str,datetime.datetime],备注信息:Union[str,None]=None, 是否删除:Union[int,None]=None, 删除时间:Union[datetime.datetime,None]=None):
    '''把样机数据插入到 MySQL 数据库中，传入的数据必须为对应的类型，不然会报错。
    

    -----
    example:
    -----
    # ## test
    # if __name__ == '__main__':
    #     # 连接数据库
    #     connection = create_connection_for_Guangyin_database("样机数据库")
    #     # 准备数据
    #     PD样品 = [1,2,3,4,5]
    #     PD光源 = [1,2,3,4,5]
    #     PD背景 = [1,2,3,4,5]
    #     重建样品 = [1,2,3,4,5]
    #     重建光源 = [1,2,3,4,5]
    #     重建样品扣背景 = [1,2,3,4,5]
    #     项目名称 = '测试项目'
    #     项目类型 = '人体'
    #     采集部位 = '手'
    #     采集日期 = '2021-01-01'
    #     志愿者 = '测试志愿者'
    #     理化值 = {'血糖': 10, '血氧': 20}
    #     创建时间 = '2021-01-01 00:00:00'
    #     备注信息 = '测试备注'
    #     是否删除 = 0
    #     删除时间 = None
    #     # 插入数据
    #     insert_prototype_data_to_mysql(connection, '样机_卷积式_v1', PD样品, PD光源, PD背景, 重建样品, 重建光源, 重建样品扣背景, 项目名称, 项目类型, 采集部位, 采集日期, 志愿者, 理化值, 创建时间, 备注信息, 是否删除, 删除时间)



    '''
    

    if 项目类型 == '人体':
        if 采集部位 is None:
            raise ValueError("项目类型为人体时，采集部位不能为空")
        
    if PD样品 is None :
        raise ValueError("PD样品数据不能为空")
    if isinstance(PD样品,  list):
        PD样品 = json.dumps(PD样品, ensure_ascii=False)
    
    

    if not (isinstance(PD光源,  (list)) or PD光源 is None):
        raise ValueError("PD光源数据类型错误，必须为列表或None")
    if isinstance(PD光源,  list):
        PD光源 = json.dumps(PD光源, ensure_ascii=False)
    
    if not (isinstance(PD背景,  (list)) or PD背景 is None):
        raise ValueError("PD背景数据类型错误，必须为列表或None")
    if isinstance(PD背景,  list):
        PD背景 = json.dumps(PD背景, ensure_ascii=False)
    
    if not (isinstance(重建样品,  list) or 重建样品 is None):
        raise ValueError("重建样品数据类型错误，必须为列表或None")
    if isinstance(重建样品,  list):
        重建样品 = json.dumps(重建样品, ensure_ascii=False)
    
    if not (isinstance(重建光源,  list) or 重建光源 is None):
        raise ValueError("重建光源数据类型错误，必须为列表或None")
    if isinstance(重建光源,  list):
        重建光源 = json.dumps(重建光源, ensure_ascii=False)
    
    if not (isinstance(重建样品扣背景,  list) or 重建样品扣背景 is None):
        raise ValueError("重建样品扣背景数据类型错误，必须为列表或None")
    if isinstance(重建样品扣背景,  list):
        重建样品扣背景 = json.dumps(重建样品扣背景, ensure_ascii=False)
    
    if not isinstance(项目名称,  str):
        raise ValueError("项目名称数据类型错误，必须为字符串")
    
    if not isinstance(项目类型,  str):
        raise ValueError("项目类型数据类型错误，必须为字符串")
    
    if not (isinstance(采集部位,  str) or 采集部位 is None):
        raise ValueError("采集部位数据类型错误，必须为字符串或None")
    
    if not isinstance(采集日期,  str):
        raise ValueError("采集日期数据类型错误，必须为字符串")
    
    if not (isinstance(志愿者,  str) or 志愿者 is None):
        raise ValueError("志愿者数据类型错误，必须为字符串")
    
    if not isinstance(理化值,  dict):
        raise ValueError("理化值数据类型错误，必须为字典")
    if isinstance(理化值,  dict):
        理化值 = json.dumps(理化值, ensure_ascii=False)
    
    if not isinstance(创建时间,  str) and not isinstance(创建时间, datetime.datetime):
        raise ValueError("创建时间数据类型错误，必须为字符串")
    
    if not (isinstance(删除时间, str) or 删除时间 is None):
        raise ValueError("备注信息数据类型错误，必须为字符串或None")
    
    if not (isinstance(是否删除,  int)   or 是否删除 is None):
        raise ValueError("是否删除数据类型错误，必须为整数或None")
    
    if not (isinstance(删除时间,  str) or 删除时间 is None):
        raise ValueError("删除时间数据类型错误，必须为字符串或None")




    try:
        # 建立数据库连接

        with connection.cursor() as cursor:
    

            
            
            # 插入数据的SQL语句
            insert_query = f"""
                INSERT INTO {table_name}  (PD样品,PD光源, PD背景, 重建样品, 重建光源,重建样品扣背景,项目名称,项目类型, 采集部位, 采集日期,  志愿者,理化值, 创建时间,备注信息, 是否删除, 删除时间)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            # 执行插入操作
            cursor.execute(insert_query, (PD样品,PD光源, PD背景, 重建样品, 重建光源,重建样品扣背景,项目名称,项目类型, 采集部位, 采集日期,  志愿者,理化值, 创建时间,备注信息, 是否删除, 删除时间))
            
            # 提交事务
            connection.commit()
            print("数据插入成功")

    except pymysql.MySQLError as e:
        print(f"数据库错误: {e}")
    
    finally:
        # 确保连接关闭
        connection.close()
        print("数据库连接已关闭")

def insert_spectrum_data_to_mysql(table_name:str,光谱:list,项目名称:str,项目类型:str,采集日期:str,理化值:dict,创建时间:str,光谱类型:str=None,采集部位:str=None, 志愿者:str=None,是否删除:int=None, 删除时间:str=None):
    '''
    把台式光谱仪光谱数据插入到 MySQL 数据库中，传入的数据必须为对应的类型，不然会报错。

    example:
        file_path = r'C:\BaiduSyncdisk\0A-ZATA\data\光谱数据\血酒精\酒精-计算式-刘波1111-1122.xlsx'
        spectrum_data = pd.read_excel(file_path,header=0,sheet_name='光谱')
        biomarks = pd.read_excel(file_path,header=0,sheet_name='理化值').to_dict('index')
        biomarks = list(biomarks.values())



        for i in tqdm(range(len(spectrum_data))):
            now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            insert_spectrum_data_to_mysql(table_name='复享光谱仪',光谱=spectrum_data.iloc[i].tolist(),项目名称='多发单收探头血糖数据',项目类型='人体',采集日期=file_path.split('\\')[-1].split('_')[0],理化值=biomarks[i],创建时间=now_time,光谱类型='',采集部位='',志愿者=file_path.split('\\')[-1].split('_')[1],是否删除=0, 删除时间=None)


    '''

    
    if 项目类型 == '人体':
        if 采集部位 is None:
            raise ValueError("项目类型为人体时，采集部位不能为空")
    # try:
    # 建立数据库连接
    connection = pymysql.connect(host=GUANGYIN_DATABASE_IP,port=53306, user='root', password='Guangyin88888888@', database='光谱数据库', charset='utf8mb4')

    
    
    with connection.cursor() as cursor:
        if isinstance(光谱,  list):
            光谱 = json.dumps(光谱, ensure_ascii=False)
        else:
            raise ValueError("光谱数据类型错误，必须为列表")
        if isinstance(理化值,  dict):
            理化值 = json.dumps(理化值, ensure_ascii=False)
        else:
            raise ValueError("理化值数据类型错误，必须为字典")
            
        # # 检查是否存在重复数据
        # check_query = f"""
        # SELECT COUNT(*) FROM {table_name} 
        # WHERE 志愿者 = %s
        # AND 采集日期 = %s
        # AND 理化值 = %s
        # AND 光谱 = %s 
        # """
        # cursor.execute(check_query, (志愿者, 采集日期, 理化值, 光谱))
        # count = cursor.fetchone()[0]
        
        # if count > 0:
        #     raise ValueError("数据已存在:发现相同志愿者、采集日期、理化值和光谱的记录")
        
        
        # 插入数据的SQL语句
        insert_query = f"""
        INSERT INTO {table_name} (光谱, 光谱类型, 项目名称, 项目类型, 采集部位, 采集日期, 志愿者, 理化值, 创建时间, 是否删除, 删除时间)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # 执行插入操作
        cursor.execute(insert_query, (光谱, 光谱类型, 项目名称, 项目类型, 采集部位, 采集日期, 志愿者, 理化值, 创建时间, 是否删除, 删除时间))
        
        # 提交事务
        connection.commit()

    # except pymysql.MySQLError as e:
    #     print(f"数据库错误: {e}")
    #     return False
    
    # finally:
    #     # 确保连接关闭
    #     connection.close()
    #     return True

def get_dataset_from_mysql(table_name:str, project_name:str, X_type:list, y_type:list=None,  start_time:str="1970-01-01 00:00:00", end_time:str="2100-01-01 00:00:00",volunteer:str=None,database='样机数据库'):
    '''
    example:
        dataset_X,dataset_y = get_dataset_from_mysql(database='光谱数据库',table_name="复享光谱仪", project_name="多发光单收光探头血糖数据", X_type=['光谱',"采集日期","志愿者"], )

    '''
    if volunteer is None:
        sql = f"SELECT {','.join(X_type)},理化值 FROM {table_name} WHERE 项目名称 = '{project_name}' AND  `采集日期` BETWEEN '{start_time}' AND '{end_time}' "
    else:
        sql = f"SELECT {','.join(X_type)},理化值 FROM {table_name} WHERE 项目名称 = '{project_name}' AND  `采集日期` BETWEEN '{start_time}' AND '{end_time} 23:59:59' AND 志愿者 = '{volunteer}'"
    data = get_data_from_mysql(sql,database)
    dataset = {}
    label = {}

    def get_json_or_str(data):
        if isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return data
        else:
            return data
    for i in X_type:
        dataset[i] = np.array([ get_json_or_str(j) for j in data[i].values])
    if y_type is not None:
        for i in y_type:
            temp = []
            for j in data['理化值'].values:
                temp.append(json.loads(j)[i])
            dataset[i] = np.array(temp)

    elif y_type is None:
        for i in json.loads(data['理化值'][0]).keys():
            temp = []
            for j in data['理化值'].values:
                temp.append(json.loads(j)[i])
            dataset[i] = np.array(temp)
    


    

    return dataset

def get_data_from_mysql(sql,database='样机数据库'):
    '''
    example:
        sql = "SELECT id,志愿者,采集日期,理化值 FROM `光谱数据库`.`FT光谱仪` WHERE `项目名称`='血糖数据'"
    data = get_data_from_mysql(sql)
    print(data)
    data.to_csv("血糖数据.csv",index=False)
    '''
    conn = pymysql.connect(host=GUANGYIN_DATABASE_IP,port=53306, user='select_user1', password='select_user1', database=database, charset='utf8mb4')
    data = pd.read_sql(sql, conn)
    conn.close()
    return data

# 使用csv文件更新数据库
def update_data_to_mysql(file_path,database='样机数据库',table_name='样机_计算式_MZI_v2'):
    '''
    example:
        update_data_to_mysql(f"{volunteer}FT.csv",database='光谱数据库',table_name='FT光谱仪')
    '''
    data = pd.read_csv(file_path)

    conn = pymysql.connect(host=GUANGYIN_DATABASE_IP,port=53306, user='root', password='Guangyin88888888@', database=database, charset='utf8mb4')
    cursor = conn.cursor()
    for i in tqdm(range(len(data))):
        # 获取data中除id外的所有列名
        columns = [col for col in data.columns if col != 'id']
        
        # 为每个列构建更新语句
        for col in columns:
            # 如果值不是None才更新
            if pd.notna(data.iloc[i][col]):
                sql = f"UPDATE {table_name} SET {col} = %s WHERE id = %s"
                cursor.execute(sql, (data.iloc[i][col], data.iloc[i]['id']))
    conn.commit()
    conn.close()


# V4版本 2024-12-18 修改了数据插入的逻辑，可以使用selected_data_datetime来选择插入的数据，可以插入多条数据
def add_XlsxData_to_GuangyinDatabase_v4(file_path:str,table:str,database:str='样机数据库',project:str= '2024人体酒精数据_样机芯片2', y_type:list=['实测值','序号','是否饮酒','皮肤水分'],selected_data_datetime:str=None):
    # V4版本 2024-12-18 修改了数据插入的逻辑，可以使用selected_data_datetime来选择插入的数据，可以插入多条数据
    # V3版本，2024-12-13 修改为add_XlsxData_to_GuangyinDatabase
    # V2版本 2024-10-31 修改了v1版本存理化值的方式，不兼容，因此V1版本被删除，该版本重命名为add_alcoholXlsxData_to_GuangyinDatabase
    '''
    -----
    example:
    -----
        add_XlsxData_to_GuangyinDatabase(file_path=r'data\光谱数据\20241030_170359alcohol_data.xlsx',table ='样机_计算式_MZI_v2' ,database='样机数据库',project= '2024人体酒精数据_样机芯片2', y_type=['实测值','序号','是否饮酒','表皮水分'])
    
    '''
    data = load_prototype_data_v2(file_path)
    
    def numpy_to_json(arr):
        if arr is None:
            return None
        if isinstance(arr, str):
            return arr
        if isinstance(arr, datetime.datetime):
            return arr.strftime('%Y-%m-%d %H:%M:%S')
        return json.dumps(arr.tolist())
    
    """
    将生成的数据插入到 prototype_data 表中，同时使用当前时间填充 datetime 字段。
    
    参数:
    data (list of tuples): 每个元组代表一行数据
    y_type (list): 需要从Measured_Value中提取的列名列表
    """
    
    # 建立数据库连接
    connection = pymysql.connect(
        host=GUANGYIN_DATABASE_IP,
        port=53306,
        user='root',
        password='Guangyin88888888@',
        database=database,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    
    PD_Sample, PD_Source, PD_BG, Recon_Sample, Recon_Source, Corrected_spectrum, Biomark, Measured_Value, y, date_time, volunteer = data
    print("数据维度：",   PD_Sample.shape,PD_Source.shape,PD_BG.shape,Recon_Sample.shape,Recon_Source.shape,Corrected_spectrum.shape,Biomark.shape,Measured_Value.shape,y.shape,date_time.shape,volunteer.shape)
    
    if selected_data_datetime is not None:
        selected_data_datetime = datetime.datetime.strptime(selected_data_datetime, '%Y-%m-%d')
        selected_indes = np.where([datetime.datetime.strptime(str(d if d is not np.nan else '1970-01-01 00:00:00').split()[0], '%Y-%m-%d') == selected_data_datetime for d in date_time])[0]
        print(selected_indes)
        PD_Sample = PD_Sample[selected_indes]
        PD_Source = PD_Source[selected_indes]
        PD_BG = PD_BG[selected_indes]
        Recon_Sample = Recon_Sample[selected_indes]
        Recon_Source = Recon_Source[selected_indes]
        Corrected_spectrum = Corrected_spectrum[selected_indes]
        Biomark = Biomark[selected_indes]
        Measured_Value = Measured_Value.iloc[selected_indes,:]
        y = y[selected_indes]
        date_time = date_time[selected_indes]
        volunteer = volunteer[selected_indes]
        print(PD_Sample.shape,PD_Source.shape,PD_BG.shape,Recon_Sample.shape,Recon_Source.shape,Corrected_spectrum.shape,Biomark.shape,Measured_Value.shape,y.shape,date_time.shape,volunteer.shape)
    # 从Measured_Value中提取指定列的数据
    selected_values = {}
    for col in y_type:
        if col in Measured_Value.columns:
            selected_values[col] = Measured_Value[col].tolist()
        else:
            print(f"警告: 列 '{col}' 在Measured_Value中未找到")
    
    try:

        sql  = f"""select 理化值列表 from 项目信息 where 项目名称 = '{project}' """
        with connection.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchone()
            if result is None:
                print(f"项目 '{project}' 不存在，请先添加项目信息")
                return
            else:
                print(f"理化值列表: {result['理化值列表']}")
                if not all(item in json.loads(result['理化值列表']) for item in y_type):
                    print(f"警告: {y_type}  中有不在理化值列表中, 请先检查")
                    return

        

        for i in tqdm(range(len(PD_Sample))):
            # 建立游标
            with connection.cursor() as cursor:
                # # 先检查记录是否存在
                # check_sql = f"""
                # SELECT COUNT(*) AS count FROM {table}
                # WHERE 采集日期 = %s
                # """
                
                # cursor.execute(check_sql, (numpy_to_json(date_time[i])))
                # result = cursor.fetchone()
                
                # 如果记录不存在，插入新数据
                if 1:
                    # 创建 SQL 插入语句
                    sql = f"""
                    INSERT INTO {table}(
                        项目类型, 项目名称, 采集日期, 采集部位, 志愿者, 
                        PD背景, PD样品, PD光源, 重建样品扣背景, 重建样品, 
                        重建光源, 理化值, 备注信息, 创建时间, 是否删除, 删除时间
                    )
                    VALUES (
                        '人体', '{project}', %s, '手臂外侧', %s, 
                        %s, %s, %s, %s, %s, %s, %s, NULL, %s, 0, NULL
                    )
                    """
                    
                    # 构建理化值数据
                    measured_data = {k: selected_values[k][i] for k in y_type if k in selected_values}
                    
                    insert_data = [
                        (
                            datetime.datetime.strptime(str(date_time[i]).replace('T', ' ').split('.')[0], '%Y-%m-%d %H:%M:%S'),
                            numpy_to_json(volunteer[i]),
                            numpy_to_json(PD_BG[i]),
                            numpy_to_json(PD_Sample[i]),
                            numpy_to_json(PD_Source[i]),
                            numpy_to_json(Corrected_spectrum[i]),
                            numpy_to_json(Recon_Sample[i]),
                            numpy_to_json(Recon_Source[i]),
                            json.dumps(measured_data,ensure_ascii=False),  # 使用提取的多个列值
                            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        )
                    ]
                    
                    # 执行 SQL 插入操作
                    cursor.executemany(sql, insert_data)
                    
                    # 提交事务
                    connection.commit()
                else:
                    print(f"Record for date {date_time[i]} and volunteer {volunteer[i]} already exists. Skipping insertion.")
    
    finally:
        # 关闭连接
        connection.close()

def add_XlsxData_to_GuangyinDatabase(file_path,table,database='样机数据库',project= '2024人体酒精数据_样机芯片2', y_type=['实测值','序号','是否饮酒','皮肤水分']):
    # V3版本，2024-12-13 修改为add_XlsxData_to_GuangyinDatabase
    # V2版本 2024-10-31 修改了v1版本存理化值的方式，不兼容，因此V1版本被删除，该版本重命名为add_alcoholXlsxData_to_GuangyinDatabase
    '''
    -----
    example:
    -----
        add_XlsxData_to_GuangyinDatabase(file_path=r'data\光谱数据\20241030_170359alcohol_data.xlsx',table ='样机_计算式_MZI_v2' ,database='样机数据库',project= '2024人体酒精数据_样机芯片2', y_type=['实测值','序号','是否饮酒','表皮水分'])
    
    '''
    data = load_prototype_data_v2(file_path)
    
    def numpy_to_json(arr):
        if arr is None:
            return None
        if isinstance(arr, str):
            return arr
        if isinstance(arr, datetime.datetime):
            return arr.strftime('%Y-%m-%d %H:%M:%S')
        return json.dumps(arr.tolist())
    
    """
    将生成的数据插入到 prototype_data 表中，同时使用当前时间填充 datetime 字段。
    
    参数:
    data (list of tuples): 每个元组代表一行数据
    y_type (list): 需要从Measured_Value中提取的列名列表
    """
    
    # 建立数据库连接
    connection = pymysql.connect(
        host=GUANGYIN_DATABASE_IP,
        port=53306,
        user='root',
        password='Guangyin88888888@',
        database=database,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    
    PD_Sample, PD_Source, PD_BG, Recon_Sample, Recon_Source, Corrected_spectrum, Biomark, Measured_Value, y, date_time, volunteer = data
    print("数据维度：",   PD_Sample.shape,PD_Source.shape,PD_BG.shape,Recon_Sample.shape,Recon_Source.shape,Corrected_spectrum.shape,Biomark.shape,Measured_Value.shape,y.shape,date_time.shape,volunteer.shape)

    # 从Measured_Value中提取指定列的数据
    selected_values = {}
    for col in y_type:
        if col in Measured_Value.columns:
            selected_values[col] = Measured_Value[col].tolist()
        else:
            print(f"警告: 列 '{col}' 在Measured_Value中未找到")
    
    try:

        sql  = f"""select 理化值列表 from 项目信息 where 项目名称 = '{project}' """
        with connection.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchone()
            if result is None:
                print(f"项目 '{project}' 不存在，请先添加项目信息")
                return
            else:
                print(f"理化值列表: {result['理化值列表']}")
                if not all(item in json.loads(result['理化值列表']) for item in y_type):
                    print(f"警告: {y_type}  中有不在理化值列表中, 请先检查")
                    return

        

        for i in tqdm(range(len(PD_Sample))):
            # 建立游标
            with connection.cursor() as cursor:
                # # 先检查记录是否存在
                # check_sql = f"""
                # SELECT COUNT(*) AS count FROM {table}
                # WHERE 采集日期 = %s
                # """
                
                # cursor.execute(check_sql, (numpy_to_json(date_time[i])))
                # result = cursor.fetchone()
                
                # 如果记录不存在，插入新数据
                if 1:
                    # 创建 SQL 插入语句
                    sql = f"""
                    INSERT INTO {table}(
                        项目类型, 项目名称, 采集日期, 采集部位, 志愿者, 
                        PD背景, PD样品, PD光源, 重建样品扣背景, 重建样品, 
                        重建光源, 理化值, 备注信息, 创建时间, 是否删除, 删除时间
                    )
                    VALUES (
                        '人体', '{project}', %s, '手臂外侧', %s, 
                        %s, %s, %s, %s, %s, %s, %s, NULL, %s, 0, NULL
                    )
                    """
                    
                    # 构建理化值数据
                    measured_data = {k: selected_values[k][i] for k in y_type if k in selected_values}
                    
                    insert_data = [
                        (
                            datetime.datetime.strptime(str(date_time[i]).replace('T', ' ').split('.')[0], '%Y-%m-%d %H:%M:%S'),
                            numpy_to_json(volunteer[i]),
                            numpy_to_json(PD_BG[i]),
                            numpy_to_json(PD_Sample[i]),
                            numpy_to_json(PD_Source[i]),
                            numpy_to_json(Corrected_spectrum[i]),
                            numpy_to_json(Recon_Sample[i]),
                            numpy_to_json(Recon_Source[i]),
                            json.dumps(measured_data,ensure_ascii=False),  # 使用提取的多个列值
                            numpy_to_json(date_time[i])
                        )
                    ]
                    
                    # 执行 SQL 插入操作
                    cursor.executemany(sql, insert_data)
                    
                    # 提交事务
                    connection.commit()
                else:
                    print(f"Record for date {date_time[i]} and volunteer {volunteer[i]} already exists. Skipping insertion.")
    
    finally:
        # 关闭连接
        connection.close()

def add_alcoholXlsxData_to_GuangyinDatabase(file_path,table,database='样机数据库',project= '2024人体酒精数据_样机芯片2', y_type=['实测值','序号','是否饮酒','皮肤水分']):
    # V3版本，2024-12-13 修改为Xls
    # V2版本 2024-10-31 修改了v1版本存理化值的方式，不兼容，因此V1版本被删除，该版本重命名为add_alcoholXlsxData_to_GuangyinDatabase
    '''
    -----
    example:
    -----
        add_alcoholXlsxData_to_GuangyinDatabase(file_path=r'data\光谱数据\20241030_170359alcohol_data.xlsx',table ='样机_计算式_MZI_v2' ,database='样机数据库', y_type=['实测值','序号','是否饮酒','表皮水分'])
    
    '''
    data = load_prototype_data_v2(file_path)
    
    def numpy_to_json(arr):
        if arr is None:
            return None
        if isinstance(arr, str):
            return arr
        if isinstance(arr, datetime.datetime):
            return arr.strftime('%Y-%m-%d %H:%M:%S')
        return json.dumps(arr.tolist())
    
    """
    将生成的数据插入到 prototype_data 表中，同时使用当前时间填充 datetime 字段。
    
    参数:
    data (list of tuples): 每个元组代表一行数据
    y_type (list): 需要从Measured_Value中提取的列名列表
    """
    
    # 建立数据库连接
    connection = pymysql.connect(
        host=GUANGYIN_DATABASE_IP,
        port=53306,
        user='root',
        password='Guangyin88888888@',
        database=database,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    
    PD_Sample, PD_Source, PD_BG, Recon_Sample, Recon_Source, Corrected_spectrum, Biomark, Measured_Value, y, date_time, volunteer = data
    
    # 从Measured_Value中提取指定列的数据
    selected_values = {}
    for col in y_type:
        if col in Measured_Value.columns:
            selected_values[col] = Measured_Value[col].tolist()
        else:
            print(f"警告: 列 '{col}' 在Measured_Value中未找到")
    
    try:

        sql  = f"""select 理化值列表 from 项目信息 where 项目名称 = '{project}' """
        with connection.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchone()
            if result is None:
                print(f"项目 '{project}' 不存在，请先添加项目信息")
                return
            else:
                print(f"理化值列表: {result['理化值列表']}")
                if not all(item in json.loads(result['理化值列表']) for item in y_type):
                    print(f"警告: {y_type}  中有不在理化值列表中, 请先检查")
                    return

        

        for i in tqdm(range(len(PD_Sample))):
            # 建立游标
            with connection.cursor() as cursor:
                # 先检查记录是否存在
                check_sql = f"""
                SELECT COUNT(*) AS count FROM {table}
                WHERE 采集日期 = %s
                """
                cursor.execute(check_sql, (numpy_to_json(date_time[i])))
                result = cursor.fetchone()
                
                # 如果记录不存在，插入新数据
                if result['count'] == 0:
                    # 创建 SQL 插入语句
                    sql = f"""
                    INSERT INTO {table}(
                        项目类型, 项目名称, 采集日期, 采集部位, 志愿者, 
                        PD背景, PD样品, PD光源, 重建样品扣背景, 重建样品, 
                        重建光源, 理化值, 备注信息, 创建时间, 是否删除, 删除时间
                    )
                    VALUES (
                        '人体', '{project}', %s, '手臂外侧', %s, 
                        %s, %s, %s, %s, %s, %s, %s, NULL, %s, 0, NULL
                    )
                    """
                    
                    # 构建理化值数据
                    measured_data = {k: selected_values[k][i] for k in y_type if k in selected_values}
                    
                    insert_data = [
                        (
                            numpy_to_json(date_time[i]),
                            numpy_to_json(volunteer[i]),
                            numpy_to_json(PD_BG[i]),
                            numpy_to_json(PD_Sample[i]),
                            numpy_to_json(PD_Source[i]),
                            numpy_to_json(Corrected_spectrum[i]),
                            numpy_to_json(Recon_Sample[i]),
                            numpy_to_json(Recon_Source[i]),
                            json.dumps(measured_data,ensure_ascii=False),  # 使用提取的多个列值
                            numpy_to_json(date_time[i])
                        )
                    ]
                    
                    # 执行 SQL 插入操作
                    cursor.executemany(sql, insert_data)
                    
                    # 提交事务
                    connection.commit()
                else:
                    print(f"Record for date {date_time[i]} and volunteer {volunteer[i]} already exists. Skipping insertion.")
    
    finally:
        # 关闭连接
        connection.close()

def sort_by_datetime(datetime_array, *data_arrays):
    """
    根据 datetime_array 排序其他数据数组，并返回排序后的结果

    参数:
    datetime_array: numpy array, 必须是 datetime64 类型
    *data_arrays: 其他任意数量的 numpy array，与 datetime_array 一一对应

    返回:
    sorted_datetime, sorted_data_arrays: 排序后的 datetime 和其他数据数组
    """
    # 将 datetime_array 转换为 datetime64[ns] 格式

    # 获取排序索引
    sorted_indices = np.argsort(np.array(datetime_array, dtype='datetime64[ns]'))

    # 对 datetime_array 进行排序
    sorted_datetime = datetime_array[sorted_indices]

    # 对其他传入的数据数组进行排序
    sorted_data_arrays = [data_array[sorted_indices] for data_array in data_arrays]
    return(sorted_datetime, *sorted_data_arrays)

def datetime_to_timestamp(data_time):
    """
    将日期时间字符串的NumPy数组转换为时间戳（以秒为单位）的NumPy数组。

    参数:
    data_time : np.array
        包含日期时间字符串的NumPy数组，格式为 'YYYY-MM-DD HH:MM:SS'

    返回:
    np.array
        包含对应时间戳的NumPy数组（以秒为单位的整数）
    """
    # 将NumPy数组转换为Pandas Series
    s = pd.Series(data_time)
    
    # 将字符串转换为datetime对象
    dt = pd.to_datetime(s)
    
    # 将datetime对象转换为时间戳（以秒为单位）
    timestamps = dt.astype('int64') // 10**9
    
    # 转换回NumPy数组并返回
    return timestamps.to_numpy()

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

def split_data_by_date(X, y, date_time, timestamp_split_point, 
                       start_timestamp='1970-09-21 00:00:00', 
                       end_timestamp='2099-12-11 23:59:59'):
    '''
    2024-10-18 V2版本,增加起始和结束时间点
    example: timestamp_split_point= ['2024-09-27 23:59:59', '2024-09-29 23:59:59']
    start_timestamp: 默认起始时间为 '1970-09-21 00:00:00'，可以通过参数传递自定义
    end_timestamp: 默认结束时间为 '2099-12-11 23:59:59'，可以通过参数传递自定义
    '''
    
    # 如果用户没有传入 start_timestamp 和 end_timestamp，则使用默认值
    train_indice = split_date_time(date_time, 
                                   start_timestamp=start_timestamp, 
                                   end_timestamp=timestamp_split_point[0])
    
    val_indice = split_date_time(date_time, 
                                 start_timestamp=timestamp_split_point[0], 
                                 end_timestamp=timestamp_split_point[1])
    
    test_indice = split_date_time(date_time, 
                                  start_timestamp=timestamp_split_point[1], 
                                  end_timestamp=end_timestamp)
    
    X_train = X[train_indice, :]
    X_val = X[val_indice, :]   
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

def load_prototype_data(file_path,pos=None):
    import time
    print('该函数将于2024年11月30日后停止使用！！！')
    time.sleep(10)
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

def load_prototype_data_v2(file_path,pos=None):
    # 2024-10-31 V2版本
    # 相比于V1版本，实测值Measured_Value的sheet的返回值修改为df格式，方便后续处理

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
        Measured_Value = Measured_Value
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
        return Measured_Value
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
    """ 与 load_dict_from_csv 配套使用，将字典数据保存为CSV文件。
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


import csv

def load_dict_from_csv(file_path):
    """
    从CSV文件中加载数据到字典，自动检测和处理数组格式的列。
    
    参数:
    file_path (str): 要读取的CSV文件路径。
    
    返回:
    dict: 包含从CSV文件加载的数据的字典，每个键对应一个列。
    数组格式的列会被解析为NumPy数组。
    """

    def is_array_like(s):
        """
        检查字符串是否看起来像一个数组。
        """
        return s.strip().startswith('[') and s.strip().endswith(']')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    
    df = pd.read_csv(file_path)
    data = {}
    
    # 动态检测数组列
    array_columns = [col for col in df.columns if df[col].dtype == 'object' and df[col].apply(is_array_like).all()]
    
    # 处理数组列
    for col in array_columns:
        data[col] = np.array([np.fromstring(i.strip('[]'), sep=' ', dtype=np.float32) for i in df[col].values])
    
    # 处理其他列
    for col in df.columns:
        if col not in array_columns:
            data[col] = df[col].values
    
    print(f"已成功从 {file_path} 加载数据")
    print("检测到的数组列:", array_columns)
    for key, value in data.items():
        print(f"{key} shape:", value.shape)
    
    return data





class FileSync:
    '''
        FileSync 类用于在本地和远程服务器之间同步文件。

    这个类提供了文件上传、下载、检查文件存在性以及文件同步的功能。
    它支持两个服务器 URL，如果主服务器不可用，会自动切换到备用服务器。

    属性:
        SERVER_URL1 (str): 主服务器的 URL。
        SERVER_URL2 (str): 备用服务器的 URL。
        server_url (str): 当前正在使用的服务器 URL。
    # 使用示例
    if __name__ == "__main__":
        sync = FileSync()
        sync.sync_file(r"C:\BaiduSyncdisk\code&note\0A-ZATA\data\光谱数据\MZI酒精数据_21&27&79&30_.xlsx")
    '''
    def __init__(self, server_url1 = f"http://{GUANGYIN_DATABASE_IP}:12185/", server_url2 = ""):
        self.SERVER_URL1 = server_url1
        self.SERVER_URL2 = server_url2
        self.server_url = None

    def check_server(self, url=None):
        if url is None:
            url = self.SERVER_URL1
        print(url)
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_server_url(self):
        if self.check_server(self.SERVER_URL1 + "check/test.txt"):
            self.server_url = self.SERVER_URL1
            return self.SERVER_URL1
        elif self.check_server(self.SERVER_URL2 + "check/test.txt"):
            self.server_url = self.SERVER_URL2
            return self.SERVER_URL2
        else:
            raise Exception("无法连接到任何服务器")

    def upload_file(self, filename):
        self.get_server_url()
        file_size = os.path.getsize(filename)

        with open(filename, 'rb') as f:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"正在上传 {filename}") as pbar:
                response = requests.post(
                    f"{self.server_url}upload",
                    files={"file": (os.path.basename(filename), f)},
                    data={"filename": os.path.basename(filename)},
                    stream=True
                )
                for chunk in iter(lambda: f.read(8192), b''):
                    if chunk:
                        pbar.update(len(chunk))
        print(response.json())

    def download_file(self, filename):
        self.get_server_url()
        with requests.get(f"{self.server_url}download/{os.path.basename(filename)}", stream=True) as response:
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                with open(filename, 'wb') as f, tqdm(
                    total=total_size, unit='B', unit_scale=True, desc=f"正在下载 {filename}"
                ) as pbar:
                    for chunk in response.iter_content(block_size):
                        size = f.write(chunk)
                        pbar.update(size)
                print(f"文件 {filename} 下载成功")
            else:
                print(response.json())

    def check_file(self, filename):
        self.get_server_url()
        basename = os.path.basename(filename)
        response = requests.get(f"{self.server_url}check/{basename}")
        return response.json()['exists']

    def sync_file(self, filename):
        try:
            self.get_server_url()  # 确保使用可用的服务器
            basename = os.path.basename(filename)
            if os.path.exists(filename) and not self.check_file(basename):
                print(f"正在将 {filename} 上传到服务器...")
                self.upload_file(filename)
            elif self.check_file(basename) and not os.path.exists(filename):
                print(f"正在从服务器下载 {filename}...")
                self.download_file(filename)
            elif not self.check_file(basename) and not os.path.exists(filename):
                print(f"文件 {filename} 在本地和服务器上都不存在")
            else:
                print(f"文件 {filename} 在本地和服务器上都已存在")
        except Exception as e:
            print(f"同步过程中发生错误: {str(e)}")












if __name__ == "__main__":
    pass