# NIR API 文档

欢迎使用 NIR API - 一个专为近红外光谱分析设计的 Python 包。

## 概述

NIR API 是一个功能强大的近红外光谱分析工具包，提供了从数据加载到模型部署的完整工作流程。该包专门为光谱数据分析、机器学习建模和结果可视化而设计。

## 主要功能

### 🔬 光谱数据处理
- **数据加载**: 支持多种格式的光谱数据导入
- **预处理**: 提供标准化、归一化、基线校正等预处理方法
- **质量控制**: 异常值检测和数据清洗功能

### 🤖 机器学习
- **回归模型**: PLSR、SVR、随机森林等多种回归算法
- **分类模型**: SVM、决策树、神经网络等分类算法
- **特征选择**: CARS、SPA、PCA 等特征选择方法
- **自动机器学习**: 基于 Optuna 的超参数优化

### 📊 数据可视化
- **光谱图**: 吸收光谱、平均光谱可视化
- **分析图表**: PCA 分析、相关性分析图表
- **模型评估**: 预测值对比、性能指标可视化

### 🔧 光谱重建
- **字典学习**: 基于字典学习的光谱重建
- **神经网络**: 深度学习方法进行光谱转换
- **传统方法**: 矩阵乘法、傅里叶变换等经典方法

## 快速开始

### 安装

```bash
pip install nirapi
```

### 基本使用

```python
import nirapi

# 加载数据
from nirapi.load_data import get_dataset_from_mysql
data = get_dataset_from_mysql(
    database='光谱数据库',
    table_name="复享光谱仪", 
    project_name="血糖检测项目",
    X_type=['光谱']
)

# 数据预处理
from nirapi.preprocessing import SNV, SG
X_processed = SNV(data['光谱'])
X_filtered = SG(X_processed)

# 机器学习建模
from nirapi.ML_model import PLSR
y_train, y_test, y_train_pred, y_test_pred = PLSR(
    X_train, X_test, y_train, y_test
)

# 结果可视化
from nirapi.draw import pred_plot
pred_plot(y_test, y_test_pred)
```

## 模块结构

- **`load_data`**: 数据加载和数据库操作
- **`preprocessing`**: 数据预处理方法
- **`ML_model`**: 机器学习模型和算法
- **`draw`**: 数据可视化功能
- **`analysis`**: 光谱数据分析工具
- **`utils`**: 工具函数和自动机器学习
- **`AnalysisClass`**: 高级分析类和光谱重建

## 应用场景

- **生物医学**: 血糖检测、皮肤水分分析
- **食品工业**: 成分分析、质量控制
- **化学分析**: 物质识别、浓度测定
- **环境监测**: 污染物检测、成分分析

## 支持的光谱仪

- 卷积式光谱仪
- FT-NIR 光谱仪
- FX-NIR 光谱仪
- 其他商用近红外光谱仪

## 下一步

- [快速开始指南](getting-started.md) - 详细的安装和使用说明
- [API 参考](api/overview.md) - 完整的 API 文档
- [教程](tutorials/basic-usage.md) - 逐步学习指南
- [示例](examples/complete-workflow.md) - 实际应用案例

## 许可证

本项目采用 MIT 许可证。详情请参见 [LICENSE](https://github.com/zata/nirapi/blob/main/LICENSE) 文件。

## 贡献

欢迎贡献代码！请参阅 [贡献指南](development/contributing.md) 了解如何参与项目开发。

## 联系方式

- 作者: zata
- 项目主页: [https://github.com/zata/nirapi](https://github.com/zata/nirapi)
- 问题反馈: [GitHub Issues](https://github.com/zata/nirapi/issues)
