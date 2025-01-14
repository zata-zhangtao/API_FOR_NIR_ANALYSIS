import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


"""
example:
def test_data_analysis_report():
    测试DataAnalysisReport类的功能
    # 创建测试数据
    test_data = {
        '2024-01-01': (np.random.rand(10, 5), np.random.rand(10)),
        '2024-01-02': (np.random.rand(10, 5), np.random.rand(10)),
        '2024-01-03': (np.random.rand(10, 5), np.random.rand(10))
    }
    
    # 测试报告生成
    report = Create_train_task('test_report.pdf')
    try:
        if os.path.exists('test_report.pdf'):
            os.remove('test_report.pdf')
        score_df = report.analyze_data(test_data, train_date='2024-01-01')
        report.add_summary_page("测试总结"+str(score_df.to_dict()))
        report.close()
        print("测试通过: 报告生成成功")
    except Exception as e:
        print(f"测试失败: {str(e)}")

test_data_analysis_report()
"""



import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages

class CreateReportbyData:
    def __init__(self, output_pdf_path: str):
        """
        初始化分析报告类
        Args:
            output_pdf_path: PDF报告的输出路径
        """
        # 检查输出路径是否存在,如果不存在则创建
        output_dir = os.path.dirname(output_pdf_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        self.output_pdf_path = output_pdf_path
        self.pdf = PdfPages(output_pdf_path)

    def generate_report(self, 
                       y_train: np.ndarray, 
                       y_train_pred: np.ndarray, 
                       y_test: np.ndarray, 
                       y_test_pred: np.ndarray) -> None:
        """
        生成分析报告并保存为PDF
        Args:
            y_train: 训练集的真实值
            y_train_pred: 训练集的预测值
            y_test: 测试集的真实值
            y_test_pred: 测试集的预测值
        """
        # 绘制训练集和测试集的散点图
        self._plot_scatter(y_train, y_train_pred, y_test, y_test_pred)

        # 绘制训练集和测试集的折线图
        self._plot_line_chart(y_train, y_train_pred, y_test, y_test_pred)

        # 绘制MAE分布图
        self.plot_mae_distribution(y_test, y_test_pred)

        # 绘制误差分布图
        self._plot_error_distribution(y_test, y_test_pred)

        # 关闭PDF文件
        self.close()

    def _plot_scatter(self, 
                     y_train: np.ndarray, 
                     y_train_pred: np.ndarray, 
                     y_test: np.ndarray, 
                     y_test_pred: np.ndarray) -> None:
        """绘制训练集和测试集的散点图"""
        plt.figure(figsize=(12, 8))

        # 绘制训练集散点图
        plt.scatter(y_train, y_train_pred, c='blue', label='Train Data', alpha=0.6)

        # 绘制测试集散点图
        plt.scatter(y_test, y_test_pred, c='red', label='Test Data', alpha=0.6)

        # 绘制对角线
        min_val = min(min(y_train), min(y_test))
        max_val = max(max(y_train), max(y_test))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

        # 计算评估指标
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # 添加文本框显示评估指标
        textstr = f'Train:\nR2 = {r2_train:.5f}\nMAE = {mae_train:.5f}\nRMSE = {rmse_train:.5f}\n\nTest:\nR2 = {r2_test:.5f}\nMAE = {mae_test:.5f}\nRMSE = {rmse_test:.5f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        plt.xlabel('Measured Value')
        plt.ylabel('Predicted Value')
        plt.title('Scatter Plot of Train and Test Data')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self.pdf.savefig(bbox_inches='tight')
        plt.close()

    def _plot_line_chart(self, 
                        y_train: np.ndarray, 
                        y_train_pred: np.ndarray, 
                        y_test: np.ndarray, 
                        y_test_pred: np.ndarray) -> None:
        """绘制训练集和测试集的折线图"""
        plt.figure(figsize=(12, 8))

        # 绘制训练集折线图
        # plt.plot(y_train, 'b-', label='Train True', alpha=0.6)
        # plt.plot(y_train_pred, 'b--', label='Train Pred', alpha=0.6)

        plt.plot(y_test, 'b-', label='Train True', alpha=0.6)
        plt.plot(y_test_pred, 'b--', label='Train Pred', alpha=0.6)
        # 绘制测试集折线图
        # plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, 'r-', label='Test True', alpha=0.6)
        # plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test_pred, 'r--', label='Test Pred', alpha=0.6)

        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title('Line Chart of Test Data')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self.pdf.savefig(bbox_inches='tight')
        plt.close()

    
    def plot_mae_distribution(self, y_true, y_pred, title='MAE Distribution'):
        """
        绘制每个样本点的MAE分布图
        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
        """
        # 计算每个点的MAE
        mae_per_point = np.abs(y_true - y_pred)
        plt.figure(figsize=(12, 8))
        
        # 绘制MAE分布散点图
        plt.scatter(y_true, mae_per_point, alpha=0.6)
        
        # 添加均值和标准差线
        mean_mae = np.mean(mae_per_point)
        std_mae = np.std(mae_per_point)
        plt.axhline(y=mean_mae, color='r', linestyle='--', label=f'Mean MAE: {mean_mae:.5f}')
        plt.axhline(y=mean_mae + std_mae, color='g', linestyle=':', label=f'Mean + Std: {(mean_mae + std_mae):.5f}')
        plt.axhline(y=mean_mae - std_mae, color='g', linestyle=':', label=f'Mean - Std: {(mean_mae - std_mae):.5f}')
        
        # 找出MAE最大的几个点
        worst_indices = np.argsort(mae_per_point)[-5:]
        for idx in worst_indices:
            plt.annotate(f'idx={idx}\n({y_true[idx]:.2f}, {mae_per_point[idx]:.5f})',
                        (y_true[idx], mae_per_point[idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.xlabel('Measured Value')
        plt.ylabel('MAE')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self.pdf.savefig(bbox_inches='tight')
        plt.close()

    

    def _plot_error_distribution(self, y_test: np.ndarray, y_test_pred: np.ndarray) -> None:
        """绘制测试集的误差分布图"""
        errors = y_test - y_test_pred
        plt.figure(figsize=(12, 8))

        # 绘制误差分布直方图
        plt.hist(errors, bins=30, color='blue', alpha=0.6, label='Error Distribution')

        # 添加均值和标准差线
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        plt.axvline(mean_error, color='red', linestyle='--', label=f'Mean Error: {mean_error:.5f}')
        plt.axvline(mean_error + std_error, color='green', linestyle=':', label=f'Mean + Std: {mean_error + std_error:.5f}')
        plt.axvline(mean_error - std_error, color='green', linestyle=':', label=f'Mean - Std: {mean_error - std_error:.5f}')

        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution of Test Data')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self.pdf.savefig(bbox_inches='tight')
        plt.close()

    def close(self) -> None:
        """关闭PDF文件"""
        self.pdf.close()



class CreateTrainReport:
    

    def __init__(self, output_pdf_path: str):
        """
        初始化分析报告类
        Args:
            output_pdf_path: PDF报告的输出路径
        """
        # 检查输出路径是否存在,如果不存在则创建
        output_dir = os.path.dirname(output_pdf_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_pdf_path = output_pdf_path
        self.pdf = PdfPages(output_pdf_path)
        from nirapi.utils import run_optuna_v5,rebuild_model_v2
    def analyze_data(self, 
                    data_dict: Dict, 
                    train_key: str = 'train',
                    test_key: str = 'test',
                    exclude_date: Optional[Union[str, List[str]]] = None,
                    n_trials: int = 100,
                    **kw) -> pd.DataFrame:
        """
        分析数据并生成报告
        Args:
            data_dict: 包含光谱和实测值的字典w
            train_date: 用于训练的日期
            exclude_date: 需要排除的日期列表
            n_trials: optuna优化迭代次数
            kw: isReg:True   selected_metric:  ['mae','mse','rmse','r2','r',"accuracy", "precision", "recall"]
        Returns:
            score_df: 包含预测结果的DataFrame
        """
        # 数据验证
        if not data_dict:
            raise ValueError("数据字典不能为空")
        if train_key not in data_dict:
            raise ValueError(f"关键字 {train_key} 不在数据集中")
            
            
        # 准备训练数据
        data_dict_train = data_dict.copy()
        if test_key in data_dict:
            data_dict_train.pop(test_key, None)
            print(f"测试数据 {test_key} 已从数据集中移除")
        if exclude_date:
            if isinstance(exclude_date, str):
                exclude_date = [exclude_date]
            for date in exclude_date:
                data_dict_train.pop(date, None)

        # 运行模型训练和预测
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        try:
            from nirapi.utils import run_optuna_v5
            print(kw)
            results = run_optuna_v5(
                data_dict_train, 
                train_key=train_key, 
                isReg=kw.get('isReg',True), 
                chose_n_trails=n_trials, 
                # selected_metric=kw.get('selected_metric','r2'), 
                save="./", 
                save_name=f"temp_{now}",
                **kw
            )
        except Exception as e:
            raise RuntimeError(f"模型训练失败: {str(e)}")

        # 计算评分
        acc_score = {}
        # # 先添加训练集的结果
        # train_X, train_y = data_dict[train_data]
        # train_y_pred = results['dataset_scores'][train_data]['y_pred']
        # train_score = results['dataset_scores'][train_data]['score']
        # acc_score[f"{train_data}(训练集)"] = [
        #     train_score,
        #     train_y_pred,
        #     train_y
        # ]
        
        # 添加其他数据集的结果
        for key, value in results['dataset_scores'].items():
            # if key != train_data:  # 跳过训练集
            acc_score[key] = [
                value['score'],
                value['y_pred'],
                data_dict_train[key][1]
            ]
        
        score_df = pd.DataFrame(acc_score, index=['score', 'y_pred', 'y_true'])
        
        self.add_text_with_level(results['best_selection_steps'])
        # 生成图表
        self._plot_prediction_results(score_df)
        self._plot_score_bars(score_df)
        self._plot_test_data_results(data_dict,train_key,test_key,results)
        self.close()
        
        return score_df
    
    def _plot_prediction_results(self, score_df: pd.DataFrame) -> None:
        """
        绘制预测结果散点图和指标
        Args:
            score_df: 包含预测结果的DataFrame
        """
        plt.figure(figsize=(12, 8))
        
        # 绘制散点图
        min_val = np.inf
        max_val = -np.inf
        colors = plt.cm.tab10(np.linspace(0, 1, len(score_df.columns)))
        
        # 计算所有指标的垂直位置
        text_y_positions = []
        for col in score_df.columns:
            y_pred = score_df.loc['y_pred', col]
            text_y_positions.append(max(y_pred))
        
        # 按列名顺序绘制散点图和指标
        textstr = ""
        for i, (col, color) in enumerate(zip(score_df.columns, colors)):
            y_true = score_df.loc['y_true', col]
            y_pred = score_df.loc['y_pred', col]
            score = score_df.loc['score', col]
            plt.scatter(y_true, y_pred, c=[color], label=f'{col} (r={score:.3f})')
            min_val = min(min_val, min(y_true), min(y_pred))
            max_val = max(max_val, max(y_true), max(y_pred))
            
            # 计算所有指标
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            r = np.corrcoef(y_true, y_pred)[0,1]
            
            # 累加指标文本
            textstr += f'{col}:\nR2 = {r2:.5f}\nMAE = {mae:.5f}\nRMSE = {rmse:.5f}\n\n'
            
        # 添加对角线
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

        # 在图表左上角添加所有指标文本
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr.rstrip(), fontsize=10,
                transform=plt.gca().transAxes,
                verticalalignment='top', bbox=props)

        plt.xlabel('Measured Value')
        plt.ylabel('Predicted Value')
        plt.title('Prediction Results Comparison for Different Dates')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存到PDF
        self.pdf.savefig(bbox_inches='tight')
        plt.close()

    def _plot_score_bars(self, score_df: pd.DataFrame) -> None:
        """
        绘制评分柱状图
        Args:
            score_df: 包含评分结果的DataFrame
        """
        plt.figure(figsize=(10, 6))
        plt.clf()  # 清除当前图形
        
        # 提取评分数据
        scores = score_df.loc['score']
        
        # 设置柱状图颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(scores)))
        
        # 绘制柱状图
        x = range(len(scores))
        bars = plt.bar(x, scores.values, color=colors)
        
        # 在柱子上方添加具体数值
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        # 设置坐标轴标签和范围
        plt.ylim(0, max(scores.values) * 1.2)  # 设置y轴范围，留出空间显示数值
        plt.xticks(x, scores.index, rotation=45)
        plt.xlabel('Dataset Date')
        plt.ylabel('Correlation Coefficient')
        plt.title('Prediction Score Comparison for Different Datasets')
        
        # 添加网格线
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存到PDF
        self.pdf.savefig(bbox_inches='tight')
        plt.close()
        
    def _plot_test_data_results(self, data_dict: Dict, train_data: str, test_data: str, results: Dict) -> None:
        """
        绘制测试数据结果
        Args:
            score_df: 包含评分结果的DataFrame
        """
        splited_data = (data_dict[train_data][0],data_dict[test_data][0],data_dict[train_data][1],data_dict[test_data][1])
        # 获取测试集结果
        from nirapi.utils import rebuild_model_v2
        y_test, y_pred = rebuild_model_v2(splited_data=splited_data,params_dict=results['best_selection_steps'])
        
        # 获取训练集结果
        train_data_split = (data_dict[train_data][0],data_dict[train_data][0],data_dict[train_data][1],data_dict[train_data][1])
        y_train, y_train_pred = rebuild_model_v2(splited_data=train_data_split,params_dict=results['best_selection_steps'])

        plt.figure(figsize=(12, 8))
        # 绘制散点图
        plt.scatter(y_train, y_train_pred, c='red', label='Train Data', alpha=0.5)
        plt.scatter(y_test, y_pred, c='blue', label='Test Data')
        
        # 绘制对角线
        min_val = min(min(y_train), min(y_test))
        max_val = max(max(y_train), max(y_test))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # 计算评估指标
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # 添加文本框显示评估指标
        textstr = f'Train:\nR2 = {r2_train:.5f}\nMAE = {mae_train:.5f}\nRMSE = {rmse_train:.5f}\n\nTest:\nR2 = {r2_test:.5f}\nMAE = {mae_test:.5f}\nRMSE = {rmse_test:.5f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.xlabel('Measured Value')
        plt.ylabel('Predicted Value')
        plt.title('Model Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self.pdf.savefig(bbox_inches='tight')
        plt.close()
        
        # 绘制RMSE分布图
        self.plot_test_line(y_test, y_pred)
        self.plot_rmse_distribution(y_test, y_pred, title='RMSE Distribution')
        self.plot_mae_distribution(y_test, y_pred, title='MAE Distribution')

    def plot_test_line(self, y_test, y_pred):
        """
        绘制测试集的折线图
        Args:
            y_test: 真实值
            y_pred: 预测值
        """
        plt.figure(figsize=(12, 8))
        
        # 绘制真实值和预测值的折线
        x = np.arange(len(y_test))
        plt.plot(x, y_test, 'b-', label='真实值', alpha=0.6)
        plt.plot(x, y_pred, 'r--', label='预测值', alpha=0.6)
        
        # 计算评估指标
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # 添加文本框显示评估指标
        textstr = f'R2 = {r2:.5f}\nMAE = {mae:.5f}\nRMSE = {rmse:.5f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.xlabel('样本序号')
        plt.ylabel('值')
        plt.title('测试集预测结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self.pdf.savefig(bbox_inches='tight')
        plt.close()
    def plot_rmse_distribution(self, y_true, y_pred, title='RMSE Distribution'):
        """
        绘制每个样本点的RMSE分布图
        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
        """
        # 计算每个点的RMSE
        rmse_per_point = np.abs(y_true - y_pred)  # 修改为绝对误差,因为RMSE是针对整体样本的概念
        plt.figure(figsize=(12, 8))
        
        # 绘制误差分布散点图
        plt.scatter(y_true, rmse_per_point, alpha=0.6)
        
        # 添加均值和标准差线
        mean_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))  # 计算整体RMSE
        std_rmse = np.std(rmse_per_point)
        plt.axhline(y=mean_rmse, color='r', linestyle='--', label=f'RMSE: {mean_rmse:.5f}')
        plt.axhline(y=mean_rmse + std_rmse, color='g', linestyle=':', label=f'Mean + Std: {(mean_rmse + std_rmse):.5f}')
        plt.axhline(y=mean_rmse - std_rmse, color='g', linestyle=':', label=f'Mean - Std: {(mean_rmse - std_rmse):.5f}')
        
        # 找出误差最大的几个点
        worst_indices = np.argsort(rmse_per_point)[-5:]
        for idx in worst_indices:
            plt.annotate(f'idx={idx}\n({y_true[idx]:.2f}, {rmse_per_point[idx]:.5f})',
                        (y_true[idx], rmse_per_point[idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.xlabel('真实值')
        plt.ylabel('绝对误差')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self.pdf.savefig(bbox_inches='tight')
        plt.close()

    def plot_mae_distribution(self, y_true, y_pred, title='MAE Distribution'):
        """
        绘制每个样本点的MAE分布图
        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
        """
        # 计算每个点的MAE
        mae_per_point = np.abs(y_true - y_pred)
        plt.figure(figsize=(12, 8))
        
        # 绘制MAE分布散点图
        plt.scatter(y_true, mae_per_point, alpha=0.6)
        
        # 添加均值和标准差线
        mean_mae = np.mean(mae_per_point)
        std_mae = np.std(mae_per_point)
        plt.axhline(y=mean_mae, color='r', linestyle='--', label=f'Mean MAE: {mean_mae:.5f}')
        plt.axhline(y=mean_mae + std_mae, color='g', linestyle=':', label=f'Mean + Std: {(mean_mae + std_mae):.5f}')
        plt.axhline(y=mean_mae - std_mae, color='g', linestyle=':', label=f'Mean - Std: {(mean_mae - std_mae):.5f}')
        
        # 找出MAE最大的几个点
        worst_indices = np.argsort(mae_per_point)[-5:]
        for idx in worst_indices:
            plt.annotate(f'idx={idx}\n({y_true[idx]:.2f}, {mae_per_point[idx]:.5f})',
                        (y_true[idx], mae_per_point[idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.xlabel('Measured Value')
        plt.ylabel('MAE')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self.pdf.savefig(bbox_inches='tight')
        plt.close()

    

        
    def add_text_with_level(self, text: Union[str, Dict], level: int = 1) -> None:
        """
        添加带层级的文字内容到PDF
        Args:
            text: 要添加的文本内容,可以是字符串或字典
            level: 文字等级(1-5),影响字体大小和缩进
        """
        plt.figure(figsize=(12, 8))
        plt.clf()  # 清除当前图形
        
        # 将字典转换为格式化文本
        if isinstance(text, dict):
            formatted_text = []
            for key, value in text.items():
                if isinstance(value, list):  # 处理列表类型的值
                    formatted_text.append(f"{key}:")
                    if len(value) >= 2:  # 确保value至少有两个元素
                        formatted_text.append(f"  方法: {value[0]}")
                        if isinstance(value[1], dict):  # 如果第二个元素是字典
                            for param_key, param_value in value[1].items():
                                formatted_text.append(f"    {param_key}: {param_value}")
                        elif isinstance(value[1], list):
                            formatted_text.append(f"    {value[1]}")
                elif isinstance(value, dict):
                    formatted_text.append(f"{key}:")
                    for sub_key, sub_value in value.items():
                        formatted_text.append(f"  {sub_key}: {sub_value}")
                else:
                    formatted_text.append(f"{key}: {value}")
            text = "\n".join(formatted_text)
        
        # 设置字体大小
        font_size = {1: 16, 2: 14, 3: 12, 4: 11, 5: 10}.get(level, 10)
        
        # 计算每行文本的高度
        line_height = 0.05
        lines = text.split('\n')
        total_height = len(lines) * line_height
        
        # 确保所有文本都能显示在图中
        y_start = 0.95
        if total_height > 0.9:  # 如果文本太长
            y_start = 0.95
        
        # 逐行添加文本
        for i, line in enumerate(lines):
            indent = 0.1
            # 根据缩进级别调整
            if line.startswith('    '):  # 三级缩进
                indent = 0.3
            elif line.startswith('  '):   # 二级缩进
                indent = 0.2
                
            plt.text(indent, y_start - i * line_height, 
                    line.lstrip(),  # 移除开头的空格
                    fontsize=font_size,
                    transform=plt.gca().transAxes)
        
        plt.axis('off')
        self.pdf.savefig(bbox_inches='tight')
        plt.close()

    def add_summary_page(self, summary_text: str) -> None:
        """
        添加总结页面到PDF
        Args:
            summary_text: 要添加的总结文本
        """
        fig = plt.figure(figsize=(12, 8))
        plt.text(0.1, 0.9, summary_text, wrap=True, fontsize=12)
        plt.axis('off')
        self.pdf.savefig()
        plt.close()
    
    def close(self) -> None:
        """关闭PDF文件"""
        self.pdf.close()



# if __name__ == "__main__":

#     def test_data_analysis_report():
#         """测试DataAnalysisReport类的功能"""
#         # 创建测试数据
#         test_data = {
#             '2024-01-01': (np.random.rand(10, 5), np.random.rand(10)),
#             '2024-01-02': (np.random.rand(10, 5), np.random.rand(10)),
#             '2024-01-03': (np.random.rand(10, 5), np.random.rand(10))
#         }
#         # 测试报告生成
#         report = Create_train_task('test_report.pdf')
#         try:
#             if os.path.exists('test_report.pdf'):
#                 os.remove('test_report.pdf')
#             score_df = report.analyze_data(test_data, train_date='2024-01-01')
#             report.add_summary_page("测试总结"+str(score_df.to_dict()))
#             report.close()
#             print("测试通过: 报告生成成功")
#         except Exception as e:
#             print(f"测试失败: {str(e)}")

#     test_data_analysis_report()