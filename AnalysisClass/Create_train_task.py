# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# import datetime
# import pandas as pd
# import numpy as np
# from typing import Dict, List, Union, Optional
# from nirapi.utils import run_optuna_v5
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import os
# class Create_train_task:
    
#     def __init__(self, output_pdf_path: str):
#         """
#         初始化分析报告类
#         Args:
#             output_pdf_path: PDF报告的输出路径
#         """
#         self.output_pdf_path = output_pdf_path
#         self.pdf = PdfPages(output_pdf_path)
        
#     def analyze_data(self, 
#                     data_dict: Dict, 
#                     train_data: str = '2024-11-14',
#                     test_data: str = 'test',
#                     exclude_date: Optional[Union[str, List[str]]] = None,
#                     n_trials: int = 5,**kw) -> pd.DataFrame:
#         """
#         分析数据并生成报告
#         Args:
#             data_dict: 包含光谱和实测值的字典w
#             train_date: 用于训练的日期
#             exclude_date: 需要排除的日期列表
#             n_trials: optuna优化迭代次数
#         Returns:
#             score_df: 包含预测结果的DataFrame
#         """
#         # 数据验证
#         if not data_dict:
#             raise ValueError("数据字典不能为空")
#         if train_date not in data_dict:
#             raise ValueError(f"训练日期 {train_date} 不在数据集中")
            
#         # 准备训练数据
#         data_dict_train = data_dict.copy()
#         if exclude_date:
#             if isinstance(exclude_date, str):
#                 exclude_date = [exclude_date]
#             for date in exclude_date:
#                 data_dict_train.pop(date, None)

#         # 运行模型训练和预测
#         now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
#         try:
#             results = run_optuna_v5(
#                 data_dict_train, 
#                 train_key=train_date, 
#                 isReg=kw.get('isReg',True), 
#                 chose_n_trails=n_trials, 
#                 selected_metric=kw.get('selected_metric','r'), 
#                 save="./", 
#                 save_name=f"temp_{now}"
#             )
#         except Exception as e:
#             raise RuntimeError(f"模型训练失败: {str(e)}")

#         # 计算评分
#         acc_score = {}
#         for key, value in results['dataset_scores'].items():
#             acc_score[key] = [
#                 value['score'],
#                 value['y_pred'],
#                 data_dict_train[key][1]
#             ]
        
#         score_df = pd.DataFrame(acc_score, index=['score', 'y_pred', 'y_true'])
        
#         self.add_text_with_level(results['best_selection_steps'])
#         # 生成图表
#         self._plot_prediction_results(score_df)
#         self._plot_score_bars(score_df)
        
#         return score_df
    
#     def _plot_prediction_results(self, score_df: pd.DataFrame) -> None:
#         """
#         绘制预测结果散点图和指标
#         Args:
#             score_df: 包含预测结果的DataFrame
#         """
#         plt.figure(figsize=(12, 8))
        
#         # 绘制散点图
#         min_val = np.inf
#         max_val = -np.inf
#         colors = plt.cm.tab10(np.linspace(0, 1, len(score_df.columns)))
        
#         # 计算所有指标的垂直位置
#         text_y_positions = []
#         for col in score_df.columns:
#             y_pred = score_df.loc['y_pred', col]
#             text_y_positions.append(max(y_pred))
        
#         # 按列名顺序绘制散点图和指标
#         for i, (col, color) in enumerate(zip(score_df.columns, colors)):
#             y_true = score_df.loc['y_true', col]
#             y_pred = score_df.loc['y_pred', col]
#             score = score_df.loc['score', col]
#             plt.scatter(y_true, y_pred, c=[color], label=f'{col} (r={score:.3f})')
#             min_val = min(min_val, min(y_true), min(y_pred))
#             max_val = max(max_val, max(y_true), max(y_pred))
            
#             # 计算所有指标
#             rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#             mae = mean_absolute_error(y_true, y_pred)
#             r2 = r2_score(y_true, y_pred)
#             r = np.corrcoef(y_true, y_pred)[0,1]
            
#             # 在散点附近添加带框的指标文本
#             stats_text = (f'{col}\n'
#                          f'RMSE = {rmse:.3f}\n'
#                          f'MAE = {mae:.3f}\n'
#                          f'R2 = {r2:.3f}\n'
#                          f'r = {r:.3f}')
            
#             # 根据数据集顺序设置文本位置
#             text_x = min(y_true)
#             text_y = max_val - (i * (max_val - min_val) * 0.1)  # 根据索引调整垂直位置
            
#             plt.text(text_x, text_y,
#                     stats_text,
#                     bbox=dict(facecolor='white',
#                              edgecolor='black', 
#                              alpha=0.8,
#                              pad=1.0,
#                              boxstyle='round'))
            
#         # 添加对角线
#         plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

#         plt.xlabel('Measured Value')
#         plt.ylabel('Predicted Value')
#         plt.title('Prediction Results Comparison for Different Dates')
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
        
#         # 保存到PDF
#         self.pdf.savefig(bbox_inches='tight')
#         plt.close()

#     def _plot_score_bars(self, score_df: pd.DataFrame) -> None:
#         """
#         绘制评分柱状图
#         Args:
#             score_df: 包含评分结果的DataFrame
#         """
#         plt.figure(figsize=(10, 6))
#         plt.clf()  # 清除当前图形
        
#         # 提取评分数据
#         scores = score_df.loc['score']
        
#         # 设置柱状图颜色
#         colors = plt.cm.tab10(np.linspace(0, 1, len(scores)))
        
#         # 绘制柱状图
#         x = range(len(scores))
#         bars = plt.bar(x, scores.values, color=colors)
        
#         # 在柱子上方添加具体数值
#         for i, bar in enumerate(bars):
#             height = bar.get_height()
#             plt.text(bar.get_x() + bar.get_width()/2., height,
#                     f'{height:.3f}',
#                     ha='center', va='bottom')
        
#         # 设置坐标轴标签和范围
#         plt.ylim(0, max(scores.values) * 1.2)  # 设置y轴范围，留出空间显示数值
#         plt.xticks(x, scores.index, rotation=45)
#         plt.xlabel('Dataset Date')
#         plt.ylabel('Correlation Coefficient')
#         plt.title('Prediction Score Comparison for Different Datasets')
        
#         # 添加网格线
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
        
#         # 保存到PDF
#         self.pdf.savefig(bbox_inches='tight')
#         plt.close()
        
#     def add_text_with_level(self, text: Union[str, Dict], level: int = 1) -> None:
#         """
#         添加带层级的文字内容到PDF
#         Args:
#             text: 要添加的文本内容,可以是字符串或字典
#             level: 文字等级(1-5),影响字体大小和缩进
#         """
#         # 根据level设置字体大小和缩进
#         font_sizes = {
#             1: 16,  # 一级标题
#             2: 14,  # 二级标题 
#             3: 12,  # 三级标题
#             4: 11,  # 四级标题
#             5: 10   # 五级标题
#         }
        
#         indents = {
#             1: 0.1,  # 一级缩进
#             2: 0.15, # 二级缩进
#             3: 0.2,  # 三级缩进
#             4: 0.25, # 四级缩进
#             5: 0.3   # 五级缩进
#         }
        
#         font_size = font_sizes.get(level, 10)  # 默认最小字号
#         indent = indents.get(level, 0.3)  # 默认最大缩进
        
#         plt.figure(figsize=(12, 8))
        
#         # 将字典转换为格式化文本
#         if isinstance(text, dict):
#             formatted_text = []
#             for key, value in text.items():
#                 if isinstance(value, dict):
#                     formatted_text.append(f"{key}:")
#                     for sub_key, sub_value in value.items():
#                         formatted_text.append(f"  {sub_key}: {sub_value}")
#                 else:
#                     formatted_text.append(f"{key}: {value}")
#             text = "\n".join(formatted_text)
            
#         # 分段处理文本
#         paragraphs = text.split('\n')
#         y_pos = 0.95  # 起始位置
        
#         for para in paragraphs:
#             if para.strip():  # 忽略空行
#                 plt.text(indent, y_pos, para,
#                         fontsize=font_size,
#                         wrap=True,
#                         transform=plt.gca().transAxes)
#                 y_pos -= 0.05  # 段落间距
        
#         plt.axis('off')
#         self.pdf.savefig()
#         plt.close()
#     def add_summary_page(self, summary_text: str) -> None:
#         """
#         添加总结页面到PDF
#         Args:
#             summary_text: 要添加的总结文本
#         """
#         fig = plt.figure(figsize=(12, 8))
#         plt.text(0.1, 0.9, summary_text, wrap=True, fontsize=12)
#         plt.axis('off')
#         self.pdf.savefig()
#         plt.close()
    
#     def close(self) -> None:
#         """关闭PDF文件"""
#         self.pdf.close()



# # if __name__ == "__main__":

# #     def test_data_analysis_report():
# #         """测试DataAnalysisReport类的功能"""
# #         # 创建测试数据
# #         test_data = {
# #             '2024-01-01': (np.random.rand(10, 5), np.random.rand(10)),
# #             '2024-01-02': (np.random.rand(10, 5), np.random.rand(10)),
# #             '2024-01-03': (np.random.rand(10, 5), np.random.rand(10))
# #         }
# #         # 测试报告生成
# #         report = Create_train_task('test_report.pdf')
# #         try:
# #             if os.path.exists('test_report.pdf'):
# #                 os.remove('test_report.pdf')
# #             score_df = report.analyze_data(test_data, train_date='2024-01-01')
# #             report.add_summary_page("测试总结"+str(score_df.to_dict()))
# #             report.close()
# #             print("测试通过: 报告生成成功")
# #         except Exception as e:
# #             print(f"测试失败: {str(e)}")

# #     test_data_analysis_report()