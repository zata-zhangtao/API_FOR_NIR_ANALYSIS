# 数据加载模块 (load_data)

数据加载模块提供了从各种数据源加载光谱数据的功能，包括数据库操作、文件处理和数据转换。

## 模块导入

```python
from nirapi import load_data
# 或者导入特定函数
from nirapi.load_data import get_dataset_from_mysql, get_wavelength_list
```

## 数据库操作

### get_dataset_from_mysql

从 MySQL 数据库中获取光谱数据集。

```python
def get_dataset_from_mysql(database, table_name, project_name, X_type,
                          y_type=None, start_time="1970-01-01 00:00:00",
                          end_time="2100-01-01 00:00:00", volunteer=None):
    """
    从 MySQL 数据库中获取光谱数据集

    Parameters:
    -----------
    database : str
        数据库名称，如 '光谱数据库'
    table_name : str
        数据库表名，如 "卷积式_v1"
    project_name : str
        项目名称
    X_type : list
        需要获取的特征类型，如 ['光谱', '采集日期', '志愿者']
    y_type : list, optional
        需要获取的标签类型，默认获取所有标签
    start_time : str, optional
        采集开始时间，默认 "1970-01-01 00:00:00"
    end_time : str, optional
        采集结束时间，默认 "2100-01-01 00:00:00"
    volunteer : str, optional
        志愿者名称，如 "张三"

    Returns:
    --------
    dict
        包含光谱数据和标签的字典
    """
```

**示例:**

```python
# 基本用法
dataset = load_data.get_dataset_from_mysql(
    database='光谱数据库',
    table_name="卷积式_v1",
    project_name="血糖检测项目",
    X_type=['光谱']
)

# 获取特定时间范围的数据
dataset = load_data.get_dataset_from_mysql(
    database='光谱数据库',
    table_name="复享光谱仪",
    project_name="多发光单收光探头血糖数据",
    X_type=['光谱', "采集日期", "志愿者"],
    y_type=['血糖值'],
    start_time="2024-01-01 00:00:00",
    end_time="2024-12-31 23:59:59",
    volunteer="张三"
)

# 返回的数据结构
print(dataset.keys())  # ['光谱', '采集日期', '志愿者', '血糖值']
```

### transform_xlsx_to_mysql

将 Excel 文件数据上传到 MySQL 数据库。

```python
def transform_xlsx_to_mysql(file_path, machine_type='卷积式_v1', upload_database=True):
    """
    将 Excel 文件数据上传到 MySQL 数据库

    Parameters:
    -----------
    file_path : str
        Excel 文件路径
    machine_type : str
        光谱仪类型，默认 '卷积式_v1'
    upload_database : bool
        是否上传到数据库，默认 True

    Returns:
    --------
    DataFrame
        处理后的数据
    """
```

**示例:**

```python
# 上传 Excel 数据到数据库
df = load_data.transform_xlsx_to_mysql(
    file_path='data/template.xlsx',
    machine_type='卷积式_v1',
    upload_database=True
)

# 仅转换不上传
df = load_data.transform_xlsx_to_mysql(
    file_path='data/template.xlsx',
    machine_type='卷积式_v1',
    upload_database=False
)
```

## 波长信息管理

### get_wavelength_list

获取商用光谱仪的波长列表。

```python
def get_wavelength_list(machine_type):
    """
    获取商用光谱仪的波长列表

    Parameters:
    -----------
    machine_type : str
        光谱仪类型，支持 "FT" 和 "FX"

    Returns:
    --------
    list
        波长列表
    """
```

**示例:**

```python
# 获取 FT-NIR 光谱仪波长
ft_wavelengths = load_data.get_wavelength_list("FT")
print(f"FT 波长数量: {len(ft_wavelengths)}")

# 获取 FX-NIR 光谱仪波长
fx_wavelengths = load_data.get_wavelength_list("FX")
print(f"FX 波长数量: {len(fx_wavelengths)}")
```

### get_feat_index_accroding_wave

根据波长范围获取对应的特征索引。

::: nirapi.nirapi.load_data.get_feat_index_accroding_wave

**示例:**

```python
# 获取特定波长范围的索引
wave_range = [1000, 1500]  # 1000-1500 nm
indices = load_data.get_feat_index_accroding_wave(
    wave_range=wave_range,
    wavelengths=ft_wavelengths
)
print(f"波长范围 {wave_range} 对应的索引: {indices}")
```

### get_wave_accroding_feat_index

根据特征索引获取对应的波长。

::: nirapi.nirapi.load_data.get_wave_accroding_feat_index

**示例:**

```python
# 根据索引获取波长
indices = [100, 200, 300]
wavelengths = load_data.get_wave_accroding_feat_index(
    index=indices,
    wavelengths=ft_wavelengths
)
print(f"索引 {indices} 对应的波长: {wavelengths}")
```

## 模型持久化

### save_model

保存训练好的模型到文件。

::: nirapi.nirapi.load_data.save_model

**示例:**

```python
from sklearn.ensemble import RandomForestRegressor

# 训练模型
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 保存模型
load_data.save_model(model, "my_model.pkl")
```

### load_model

从文件加载已保存的模型。

::: nirapi.nirapi.load_data.load_model

**示例:**

```python
# 加载模型
loaded_model = load_data.load_model("my_model.pkl")

# 使用加载的模型进行预测
predictions = loaded_model.predict(X_test)
```

## 数据处理工具

### split_data_by_date

根据时间戳分割数据集。

::: nirapi.nirapi.load_data.split_data_by_date

**示例:**

```python
# 按时间分割数据
X_train, X_val, X_test, y_train, y_val, y_test = load_data.split_data_by_date(
    X=spectral_data,
    y=target_values,
    date_time=timestamps,
    split_points=['2024-09-27 23:59:59', '2024-09-29 23:59:59']
)
```

### repeat_values_to_csv

将数组中的每个元素重复 n 遍并保存到 CSV。

::: nirapi.nirapi.load_data.repeat_values_to_csv

**示例:**

```python
import numpy as np

# 示例数据
data = np.array([
    [2.1, 4.151],
    [0.84, 1.762], 
    [0.21, 0.49],
    [1.68, 3.375]
])

# 重复数据 30 遍并保存
load_data.repeat_values_to_csv(
    input_data=data,
    n_repeats=30,
    output_file="repeated_data.csv"
)
```

## 文件操作

### save_dict_to_csv

将字典数据保存为 CSV 文件。

::: nirapi.nirapi.load_data.save_dict_to_csv

**示例:**

```python
# 保存字典到 CSV
data_dict = {
    '光谱': spectral_data,
    '血糖值': glucose_values,
    '志愿者': volunteer_names
}

load_data.save_dict_to_csv(data_dict, "dataset.csv")
```

### load_dict_from_csv

从 CSV 文件加载字典数据。

::: nirapi.nirapi.load_data.load_dict_from_csv

**示例:**

```python
# 从 CSV 加载字典
loaded_dict = load_data.load_dict_from_csv("dataset.csv")
print(loaded_dict.keys())
```

## 数据库配置

### 连接配置

数据库连接需要在配置文件中设置：

```python
# config.py
DATABASE_CONFIG = {
    'host': 'your_host',
    'user': 'your_username', 
    'password': 'your_password',
    'database': '光谱数据库',
    'charset': 'utf8mb4'
}
```

### 支持的数据库表

目前支持的光谱仪类型和对应的数据库表：

- `卷积式_v1` - 卷积式光谱仪版本1
- `复享光谱仪` - 复享品牌光谱仪
- `FT_NIR` - 傅里叶变换近红外光谱仪
- `FX_NIR` - FX 系列近红外光谱仪

## 错误处理

```python
try:
    dataset = load_data.get_dataset_from_mysql(
        database='光谱数据库',
        table_name="不存在的表",
        project_name="测试项目",
        X_type=['光谱']
    )
except Exception as e:
    print(f"数据库操作失败: {e}")
```

## 最佳实践

### 1. 数据验证

```python
# 加载数据后进行验证
dataset = load_data.get_dataset_from_mysql(...)
assert '光谱' in dataset, "光谱数据缺失"
assert len(dataset['光谱']) > 0, "光谱数据为空"
```

### 2. 内存管理

```python
# 对于大数据集，分批处理
def load_data_in_batches(table_name, batch_size=1000):
    # 实现分批加载逻辑
    pass
```

### 3. 缓存机制

```python
# 使用缓存避免重复数据库查询
@load_data.cache_data
def load_cached_dataset():
    return load_data.get_dataset_from_mysql(...)
```

## 相关模块

- [预处理模块](preprocessing.md) - 数据预处理方法
- [分析模块](analysis.md) - 数据分析工具
- [工具模块](utils.md) - 实用工具函数
