import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import io
import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import os
matplotlib.use('Agg')
import scipy
import sys

# 设置matplotlib中文字体


class SpectralAnalysisReport:
    def __init__(self, dataset, output_path='spectral_analysis_report.pdf'):
        """
        初始化光谱数据分析报告类
        """
        # matplotlib.rcParams['font.family'] = ['sans-serif']
        # matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 
        #                                         'Microsoft YaHei', 'WenQuanYi Micro Hei']
        # matplotlib.rcParams['axes.unicode_minus'] = False
        # if '光谱' not in dataset:
        #     raise KeyError("数据集中必须包含'光谱'数据")
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 获取系统中可用的中文字体
        font_paths = []
        
        # Windows系统字体路径
   
            
        # Linux系统字体路径
        if os.name == 'posix':
            font_paths.extend([
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                
                os.path.expanduser('~/.fonts')
            ])
            
        # macOS系统字体路径
        elif sys.platform == 'darwin':
            font_paths.extend([
                '/System/Library/Fonts',
                '/Library/Fonts',
                os.path.expanduser('~/Library/Fonts')
            ])
            
        # 加载系统字体
        for font_path in font_paths:
            if os.path.exists(font_path):
                matplotlib.font_manager.fontManager.addfont(font_path)
        self.dataset = dataset
        self.output_path = output_path
        self.spectral_data = dataset['光谱']
        self.n_samples, self.n_features = self.spectral_data.shape
        
        # 配置中文字体
        self._setup_fonts()
        
        # 初始化PDF文档
        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # 初始化样式
        self.styles = getSampleStyleSheet()
        self._setup_styles()
        self.pdf_elements = []

    def _setup_fonts(self):
        """配置中文字体"""
        try:
            # 尝试注册 Microsoft YaHei 字体
            pdfmetrics.registerFont(TTFont('MyFont', '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'))
        except:
            try:
                # 尝试注册 SimSun 字体
                pdfmetrics.registerFont(TTFont('MyFont', 'simsun.ttc'))
            except:
                try:
                    # 尝试注册 SimHei 字体
                    pdfmetrics.registerFont(TTFont('MyFont', 'simhei.ttf'))
                except:
                    print("警告：未能找到合适的中文字体，可能会影响PDF中的中文显示")
                    print("请确保系统中安装了以下字体之一：Microsoft YaHei (msyh.ttc)、SimSun (simsun.ttc)、SimHei (simhei.ttf)")

    def _setup_styles(self):
        """设置文档样式"""
        # 标题样式
        self.styles.add(ParagraphStyle(
            name='ChineseHeading1',
            fontName='MyFont',
            fontSize=18,
            leading=22,
            spaceAfter=12,
            alignment=1  # 居中
        ))
        
        self.styles.add(ParagraphStyle(
            name='ChineseHeading2',
            fontName='MyFont',
            fontSize=16,
            leading=20,
            spaceAfter=10,
            spaceBefore=10
        ))
        
        # 正文样式
        self.styles.add(ParagraphStyle(
            name='ChineseBody',
            fontName='MyFont',
            fontSize=12,
            leading=14,
            alignment=0  # 左对齐
        ))

    def add_heading(self, text, level=1):
        """添加标题"""
        style = 'ChineseHeading1' if level == 1 else 'ChineseHeading2'
        self.pdf_elements.append(Paragraph(text, self.styles[style]))
        self.pdf_elements.append(Spacer(1, 12))

    def add_paragraph(self, text):
        """添加段落"""
        self.pdf_elements.append(Paragraph(text, self.styles['ChineseBody']))
        self.pdf_elements.append(Spacer(1, 12))

    def add_table(self, data, colWidths=None):
        """添加表格"""
        # 设置表格样式
        style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'MyFont'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]
        
        if not colWidths:
            colWidths = [self.doc.width/len(data[0])] * len(data[0])
        table = Table(data, colWidths=colWidths, style=style)
        
        self.pdf_elements.append(table)
        self.pdf_elements.append(Spacer(1, 12))

# ... existing code ...
    def figure_to_image(self, fig):
        """将matplotlib图形转换为reportlab图像"""
        buf = io.BytesIO()
        
        # 获取PDF页面的可用空间
        available_width = self.doc.width * 0.9  # 留出10%边距
        available_height = self.doc.height * 0.6  # 留出40%用于其他内容
        
        # 计算当前图像尺寸
        fig_size = fig.get_size_inches()
        dpi = 100  # 降低DPI以减小文件大小
        
        # 计算图像的实际像素尺寸
        img_width = fig_size[0] * dpi
        img_height = fig_size[1] * dpi
        
        # 计算缩放比例
        width_ratio = available_width / img_width
        height_ratio = available_height / img_height
        scale = min(width_ratio, height_ratio, 1.0)  # 不要放大，只缩小
        
        # 设置中文字体
        
 # 保存图像前确保所有文本元素使用中文字体
        # for text_obj in fig.findobj(match=lambda x: hasattr(x, 'get_text')):
        #     try:
        #         text_obj.set_fontproperties(plt.font_manager.FontProperties(
        #             family=['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'sans-serif']
        #         ))
        #     except:
        #         pass
                
        # 保存图像
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                   pad_inches=0.1)
    
        # plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 
        #                                 'Computer Modern Sans Serif', 'Lucida Grande', 
        #                                 'Verdana', 'Geneva', 'Lucid', 'Arial', 
        #                                 'Helvetica', 'Avant Garde', 'sans-serif']
        # plt.rcParams['axes.unicode_minus'] = False
        
        # # 保存图像时指定额外的字体设置
        # fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
        #             # fonttype=3,  # 使用Type 3字体，可以更好地支持中文
        #             pad_inches=0.1)
        
        buf.seek(0)
        img = Image(buf)
        
        # 设置图像在PDF中的显示尺寸
        img.drawWidth = img_width * scale
        img.drawHeight = img_height * scale
        
        return img
# ... existing code ...

    def analyze_and_generate_report(self):
        try:
            """执行分析并生成PDF报告"""
            self.add_heading("数据分析报告", 1)
                
            # 1. 数据集基本信息
            self.add_heading("1. 数据集基本信息", 2)
            self._analyze_dataset_info()

            self.add_heading("光谱数据", 2)
            self._plot_all_spectra()

            self.add_heading("数据关系分析", 2)
            self._plot_data_relationships()

            self.add_heading("两两分布", 2)
            self._plot_pairwise_relationships()
            
            # 2. 光谱数据分析
            self.add_heading("2. 光谱数据分析", 2)
            self._analyze_spectral_data()
            
            # 3. 其他特征分析
            self.add_heading("3. 其他特征分析", 2)
            self._analyze_other_features()
            
            # 4. 时间模式分析
            if '采集日期' in self.dataset:
                self.add_heading("4. 时间模式分析", 2)
                daily_stats = self._analyze_temporal_patterns()
                self.add_paragraph("时间模式分析结果：")
                
                # 添加时间模式统计表格
                stats_table = [['日期', '平均强度', '标准差', '样本数']]
                for _, row in daily_stats.iterrows():
                    stats_table.append([
                        str(row['date']),  # 直接访问 date 列
                        f"{row[('mean_intensity', 'mean')]:.4f}",  # 正确访问多级索引
                        f"{row[('mean_intensity', 'std')]:.4f}",   # 正确访问多级索引
                        str(int(row[('mean_intensity', 'count')]))  # 正确访问多级索引
                    ])
                self.add_table(stats_table)
            
            # 5. 志愿者模式分析
            if '志愿者' in self.dataset:
                self.add_heading("5. 志愿者模式分析", 2)
                volunteer_stats = self._analyze_volunteer_patterns()
                self.add_paragraph("志愿者模式分析结果：")
                
                # 添加志愿者统计表格
                stats_table = [['志愿者ID', '平均强度', '标准差', '样本数']]
                for _, row in volunteer_stats.iterrows():
                    stats_table.append([
                        str(row['volunteer']),
                        f"{row['mean_intensity']['mean']:.4f}",
                        f"{row['mean_intensity']['std']:.4f}",
                        str(int(row['mean_intensity']['count']))
                    ])
                self.add_table(stats_table)
            
            # 6. 相关性分析
            if len(self.dataset.keys()) > 1:
                self.add_heading("6. 特征相关性分析", 2)
                self._analyze_correlations()

            # 
            self._analyze_spectral_details()

            # 7. 模型分析
            self.add_heading("7. 模型分析", 2)
            self._analyze_models()

        
        
        # 生成PDF文件
        # try:
            self.doc.build(self.pdf_elements)
            print(f"报告已生成: {self.output_path}")
        except Exception as e:
            print(f"错误发生在: {e.__traceback__.tb_frame.f_code.co_filename} 第 {e.__traceback__.tb_lineno} 行")
            raise
    
    def _plot_data_relationships(self):
        """绘制任意两个数据类型之间的关系图"""
        # 获取所有可用于绘图的数据列
        plottable_data = {}
        for key, value in self.dataset.items():
            if key != '光谱':  # 排除光谱数据
                try:
                    # 尝试转换为数值类型
                    numeric_data = pd.to_numeric(value, errors='coerce')
                    if not numeric_data.isna().all():  # 如果不是全部为NA，则认为是数值型
                        plottable_data[key] = numeric_data
                    else:  # 如果全部转换失败，则作为分类数据处理
                        plottable_data[key] = pd.Series(value).astype(str)
                except:
                    # 如果转换失败，作为分类数据处理
                    plottable_data[key] = pd.Series(value).astype(str)
        if len(plottable_data) < 2:
            self.add_paragraph("数据集中可用于关系分析的变量少于2个，无法进行关系可视化。")
            return

        # 对所有可能的数据对进行可视化
        for i, (key1, data1) in enumerate(plottable_data.items()):
            for key2, data2 in list(plottable_data.items())[i+1:]:
                fig = plt.figure(figsize=(12, 6))
                
                # 根据数据类型选择适当的可视化方法
                if pd.api.types.is_numeric_dtype(data1) and pd.api.types.is_numeric_dtype(data2) and not data1.isna().all() and not data2.isna().all():
                    # 数值 vs 数值：散点图
                    plt.scatter(data1, data2, alpha=0.5)
                    
                    # 添加趋势线
                    try:
                        z = np.polyfit(data1, data2, 1)
                        p = np.poly1d(z)
                        plt.plot(data1, p(data1), "r--", alpha=0.8)
                        
                        # 计算相关系数
                        corr, p_val = scipy.stats.pearsonr(data1, data2)
                        plt.text(0.05, 0.95, 
                                f'相关系数: {corr:.3f}\np值: {p_val:.3e}',
                                transform=plt.gca().transAxes,
                                bbox=dict(facecolor='white', alpha=0.8))
                    except:
                        pass

                elif pd.api.types.is_numeric_dtype(data1) and not pd.api.types.is_numeric_dtype(data2):
                    # 数值 vs 分类：箱线图
                    df_temp = pd.DataFrame({'value': data1, 'category': data2})
                    unique_categories = df_temp['category'].unique()
                    data_by_category = [df_temp[df_temp['category'] == cat]['value'].dropna().values 
                                      for cat in unique_categories]
                    # 过滤掉空的类别
                    valid_categories = []
                    valid_data = []
                    for cat, data in zip(unique_categories, data_by_category):
                        if len(data) > 0:
                            valid_categories.append(str(cat))
                            valid_data.append(data)
                    
                    if valid_data:
                        plt.boxplot(valid_data, labels=valid_categories)
                        plt.xticks(rotation=45)
                        
                        # 进行方差分析
                        if len(valid_data) >= 2:  # 至少需要两组数据才能进行ANOVA
                            try:
                                # 确保每组至少有两个有效值
                                valid_groups = [group for group in valid_data if len(group) >= 2]
                                if len(valid_groups) >= 2:
                                    f_stat, p_val = scipy.stats.f_oneway(*valid_groups)
                                    plt.text(0.05, 0.95, 
                                            f'ANOVA检验:\nF统计量: {f_stat:.3f}\np值: {p_val:.3e}',
                                            transform=plt.gca().transAxes,
                                            bbox=dict(facecolor='white', alpha=0.8))
                                else:
                                    plt.text(0.05, 0.95, 
                                            '无法进行ANOVA检验:\n每组需至少2个样本',
                                            transform=plt.gca().transAxes,
                                            bbox=dict(facecolor='white', alpha=0.8))
                            except Exception as e:
                                plt.text(0.05, 0.95, 
                                        f'ANOVA检验失败:\n{str(e)}',
                                        transform=plt.gca().transAxes,
                                        bbox=dict(facecolor='white', alpha=0.8))
                    else:
                        plt.text(0.5, 0.5, '没有足够的有效数据可供分析',
                                horizontalalignment='center',
                                verticalalignment='center')

                elif pd.api.types.is_numeric_dtype(data2) and not pd.api.types.is_numeric_dtype(data1):
                    # 分类 vs 数值：箱线图
                    df_temp = pd.DataFrame({'value': data2, 'category': data1})
                    unique_categories = df_temp['category'].unique()
                    data_by_category = [df_temp[df_temp['category'] == cat]['value'].dropna().values 
                                      for cat in unique_categories]
                    
                    # 过滤掉空的类别
                    valid_categories = []
                    valid_data = []
                    for cat, data in zip(unique_categories, data_by_category):
                        if len(data) > 0:
                            valid_categories.append(str(cat))
                            valid_data.append(data)
                    
                    if valid_data:
                        plt.boxplot(valid_data, labels=valid_categories)
                        plt.xticks(rotation=45)
                        
                        # 进行方差分析
                        if len(valid_data) >= 2:  # 至少需要两组数据才能进行ANOVA
                            try:
                                f_stat, p_val = scipy.stats.f_oneway(*valid_data)
                                plt.text(0.05, 0.95, 
                                        f'ANOVA检验:\nF统计量: {f_stat:.3f}\np值: {p_val:.3e}',
                                        transform=plt.gca().transAxes,
                                        bbox=dict(facecolor='white', alpha=0.8))
                            except Exception as e:
                                print(f"ANOVA分析失败: {str(e)}")
                    else:
                        plt.text(0.5, 0.5, '没有有效的数据可供分析',
                                horizontalalignment='center',
                                verticalalignment='center')

                else:
                    # 分类 vs 分类：热力图
                    try:
                        # 确保数据都是字符串类型
                        df_temp = pd.DataFrame({
                            'var1': pd.Series(data1).astype(str),
                            'var2': pd.Series(data2).astype(str)
                        })
                        
                        # 创建交叉表
                        contingency_table = pd.crosstab(df_temp['var1'], df_temp['var2'])
                        # 检查表格大小，调整可视化参数
                        if contingency_table.shape[0] * contingency_table.shape[1] > 100:
                            plt.figure(figsize=(20, 12))
                            sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd',
                                      annot_kws={'size': 8})
                        else:
                            sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd')
                        
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        
                        # 进行卡方检验
                        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                            chi2, p_val, dof, expected = scipy.stats.chi2_contingency(contingency_table)
                            plt.text(1.05, 0.95,
                                   f'卡方检验:\n统计量: {chi2:.3f}\np值: {p_val:.3e}',
                                   transform=plt.gca().transAxes,
                                   bbox=dict(facecolor='white', alpha=0.8))
                    except Exception as e:
                        plt.text(0.5, 0.5, f'无法创建热力图: {str(e)}',
                                horizontalalignment='center',
                                verticalalignment='center')
                    
                    # 进行卡方检验
                    try:
                        chi2, p_val, dof, expected = scipy.stats.chi2_contingency(contingency_table)
                        plt.text(1.05, 0.95, 
                                f'卡方检验:\n统计量: {chi2:.3f}\np值: {p_val:.3e}',
                                transform=plt.gca().transAxes,
                                bbox=dict(facecolor='white', alpha=0.8))
                    except:
                        pass

                plt.title(f'{key1} vs {key2}的关系图')
                plt.xlabel(key1)
                plt.ylabel(key2)
                plt.grid(True, alpha=0.3)
                
                # 调整布局以避免标签重叠
                plt.tight_layout()
                # 添加到PDF
                self.pdf_elements.append(self.figure_to_image(fig))
                plt.close(fig)
                
                # 添加统计描述
                self.add_paragraph(f"\n{key1}与{key2}的关系分析：")
                
                # 根据数据类型添加不同的统计描述
                if data1.dtype.kind in 'iufc' and data2.dtype.kind in 'iufc':
                    # 添加数值型变量之间的统计描述
                    stats_table = [['统计量', '值']]
                    stats_table.append(['样本数', str(len(data1))])
                    
                    if corr is not None:
                        stats_table.append(['Pearson相关系数', f"{corr:.4f}"])
                        stats_table.append(['相关性p值', f"{p_val:.4e}"])
                    
                    self.add_table(stats_table)
                    
                elif data1.dtype.kind not in 'iufc' or data2.dtype.kind not in 'iufc':
                    # 添加分类变量相关的统计描述
                    if 'f_stat' in locals():
                        stats_table = [['统计量', '值']]
                        stats_table.append(['ANOVA F统计量', f"{f_stat:.4f}"])
                        stats_table.append(['ANOVA p值', f"{p_val:.4e}"])
                        self.add_table(stats_table)
                    
                    # 添加基本的描述性统计
                    if data1.dtype.kind in 'iufc':
                        numeric_data = data1
                        category_data = data2
                    else:
                        numeric_data = data2
                        category_data = data1
                    
                    # 计算每个类别的描述性统计
                    desc_table = [['类别', '样本数', '平均值', '标准差', '最小值', '最大值']]
                    
                    # 确保数据为数值型
                    numeric_data = pd.to_numeric(numeric_data, errors='coerce')
                    
                    # 使用pandas进行分组统计,避免空值和非数值的问题
                    df = pd.DataFrame({'numeric': numeric_data, 'category': category_data})
                    for cat in df['category'].unique():
                        cat_data = df[df['category'] == cat]['numeric'].dropna()
                        if len(cat_data) > 0:
                            desc_table.append([
                                str(cat),
                                str(len(cat_data)),
                                f"{cat_data.mean():.4f}",
                                f"{cat_data.std():.4f}",
                                f"{cat_data.min():.4f}", 
                                f"{cat_data.max():.4f}"
                            ])
                        else:
                            desc_table.append([
                                str(cat),
                                '0',
                                'N/A',
                                'N/A', 
                                'N/A',
                                'N/A'
                            ])
                    self.add_table(desc_table)
    
    def _plot_all_spectra(self):
        """绘制按不同标签分组的光谱数据叠加图"""
        # 获取所有非光谱数据的列作为标签
        label_columns = [key for key in self.dataset.keys() if key != '光谱']
        
        for label_column in label_columns:
            try:
                # 创建图形
                
                # 获取唯一的标签值
                unique_labels = np.unique(self.dataset[label_column])
                
                # 为不同标签设置不同的颜色
                colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
                fig = plt.figure(figsize=(15, 10))
                gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
                
                # 上方子图：按标签分组的光谱
                ax1 = plt.subplot(gs[0])
                
                # 为每个标签绘制光谱
                for label, color in zip(unique_labels, colors):
                    # 获取该标签对应的光谱数据索引
                    mask = self.dataset[label_column] == label
                    label_spectra = self.spectral_data[mask]
                    
                    # 计算该标签的平均光谱和标准差
                    label_mean = np.mean(label_spectra, axis=0)
                    label_std = np.std(label_spectra, axis=0)
                    
                    # 绘制该标签的所有光谱（透明度较高）
                    alpha_value = max(0.05, 1.0 / np.sqrt(len(label_spectra)))
                    for spectrum in label_spectra:
                        ax1.plot(spectrum, '-', color=color, alpha=alpha_value, linewidth=0.5)
                    
                    # 绘制该标签的平均光谱（不透明）
                    ax1.plot(label_mean, '-', color=color, linewidth=2, 
                            label=f'{label_column}={label} (n={len(label_spectra)})')
                    
                    # 绘制标准差范围
                    ax1.fill_between(range(len(label_mean)),
                                label_mean - label_std,
                                label_mean + label_std,
                                color=color, alpha=0.2)
                
                # 设置上方子图的标签和标题
                ax1.set_title(f'按{label_column}分组的光谱数据')
                ax1.set_xlabel('波长索引')
                ax1.set_ylabel('光谱强度')
                ax1.grid(True, alpha=0.3)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # 下方子图：各组的变异系数
                ax2 = plt.subplot(gs[1])
                
                # 为每个标签计算和绘制变异系数
                for label, color in zip(unique_labels, colors):
                    mask = self.dataset[label_column] == label
                    label_spectra = self.spectral_data[mask]
                    
                    # 计算变异系数
                    label_mean = np.mean(label_spectra, axis=0)
                    label_std = np.std(label_spectra, axis=0)
                    cv = label_std / np.abs(label_mean) * 100
                    
                    # 绘制变异系数
                    ax2.plot(cv, '-', color=color, label=f'{label_column}={label}')
                
                # 设置下方子图的标签
                ax2.set_xlabel('波长索引')
                ax2.set_ylabel('变异系数 (%)')
                ax2.grid(True, alpha=0.3)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # 为每个标签添加统计信息
                stats_table = [['标签值', '样本数', '平均强度', '标准差', '平均变异系数(%)']]
                for label in unique_labels:
                    mask = self.dataset[label_column] == label
                    label_spectra = self.spectral_data[mask]
                    
                    mean_intensity = np.mean(label_spectra)
                    std_intensity = np.std(label_spectra)
                    mean_cv = np.mean(np.std(label_spectra, axis=0) / 
                                    np.abs(np.mean(label_spectra, axis=0))) * 100
                    
                    stats_table.append([
                        str(label),
                        str(len(label_spectra)),
                        f"{mean_intensity:.4f}",
                        f"{std_intensity:.4f}",
                        f"{mean_cv:.2f}"
                    ])
                
                plt.tight_layout()
                self.pdf_elements.append(self.figure_to_image(fig))
                plt.close(fig)
                
                # 添加该标签的统计表格
                self.add_paragraph(f"\n{label_column}分组统计：")
                self.add_table(stats_table)
            except Exception as e:
                print(f"{sys._getframe().f_lineno}: draw spectra failed: {str(e)}")

    def _analyze_models(self):
        """分析不同数据类型的预测建模，支持基于分类变量的分组分析"""
        # 准备数据
        chemical_features = {}
        categorical_features = {}
        
        for key, value in self.dataset.items():
            if key != '光谱':
                if pd.api.types.is_numeric_dtype(value):
                    chemical_features[key] = value
                else:
                    try:
                        # 尝试转换为数值类型
                        numeric_value = pd.to_numeric(value)
                        chemical_features[key] = numeric_value
                    except:
                        categorical_features[key] = value
        
        if not chemical_features:
            return
        
        spectral_data = self.spectral_data
        # 对每个数值特征进行预测分析
        for target_name, target_values in chemical_features.items():
            self.add_heading(f"{target_name}的模型预测分析", 3)
            
            # 1. 使用光谱数据进行常规预测
            self.add_heading("基于光谱数据的预测", 4)
            spectral_prediction = self._analyze_with_spectral(
                spectral_data, 
                target_values,
                target_name
            )
            
            # 2. 对每个分类变量进行分组预测分析
            if categorical_features:
                for cat_name, cat_values in categorical_features.items():
                    try:
                        self.add_heading(f"按{cat_name}分组的{target_name}预测分析", 4)
                        self._analyze_by_group(
                            spectral_data,
                            target_values,
                            cat_values,
                            target_name,
                            cat_name
                        )
                    except Exception as e:
                        print(f"{sys._getframe().f_lineno}: analyze by group failed: {str(e)}")
                    

            # 3. 如果有日期数据，进行时间序列分析
            if '采集日期' in self.dataset:
                self.add_heading(f"{target_name}的时间序列预测分析", 4)
                time_values = pd.to_datetime(self.dataset['采集日期'])
                try:
                    self._analyze_by_time(
                        spectral_data,
                        target_values,
                        time_values,
                        target_name,
                        '采集日期'
                    )
                except Exception as e:
                    print(f"{sys._getframe().f_lineno}: analyze by time failed: {str(e)}")

    def _categorize_features(self):
        """将数据集特征分类为不同类型"""
        feature_types = {
            'numeric': {},    # 连续数值型特征
            'categorical': {}, # 离散分类型特征
            'temporal': {},   # 时间型特征
            'spectral': {}    # 光谱数据
        }
        
        for key, value in self.dataset.items():
            if key == '光谱':
                feature_types['spectral'][key] = {
                    'data': value,
                    'shape': value.shape
                }
                continue
            
            # 尝试转换为日期类型
            try:
                pd.to_datetime(value)
                feature_types['temporal'][key] = {
                    'data': pd.to_datetime(value),
                    'unique_count': len(pd.unique(value))
                }
                continue
            except:
                pass
            
            # 检查是否为数值型
            if pd.api.types.is_numeric_dtype(value):
                feature_types['numeric'][key] = {
                    'data': value,
                    'mean': np.mean(value),
                    'std': np.std(value),
                    'unique_count': len(pd.unique(value))
                }
            else:
                # 非数值型视为分类变量
                feature_types['categorical'][key] = {
                    'data': value,
                    'unique_count': len(pd.unique(value)),
                    'categories': pd.unique(value)
                }
        
        return feature_types

    def _analyze_with_spectral(self, spectral_data, target_values, target_name):
        """使用光谱数据进行预测分析"""
        # 定义模型
        models = {
            'PLS回归': PLSRegression(n_components=10),
            '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
            '线性回归': LinearRegression()
        }
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            spectral_data, target_values, test_size=0.2, random_state=42
        )
        
        results = {}
        fig = plt.figure(figsize=(16, 12))
        
        for i, (name, model) in enumerate(models.items(), 1):
            # 训练和预测
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # 计算评估指标
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
            
            # 绘制预测散点图
            plt.subplot(2, 2, i)
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', lw=2)
            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.title(f'{name}预测结果')
            
            plt.text(0.05, 0.95, 
                    f'R2 = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}',
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        plt.tight_layout()
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        # 添加结果表格
        results_table = [['模型', 'R2得分', 'RMSE', 'MAE']]
        for name, metrics in results.items():
            results_table.append([
                name,
                f"{metrics['R2']:.3f}",
                f"{metrics['RMSE']:.3f}",
                f"{metrics['MAE']:.3f}"
            ])
        
        self.add_table(results_table)
        return results

    def _analyze_by_group(self, spectral_data, target_values, group_values, 
                        target_name, group_name):
        """按分组进行预测分析"""
        self.add_heading(f"按{group_name}分组的{target_name}预测分析", 4)
        
        unique_groups = np.unique(group_values)
        group_results = {}
        
        # 创建分组结果图
        n_groups = len(unique_groups)
        n_cols = min(2, n_groups)
        n_rows = (n_groups + 1) // 2
        fig = plt.figure(figsize=(15 * n_cols, 10 * n_rows))
        
        for idx, group in enumerate(unique_groups, 1):
            # 获取该组的数据
            mask = group_values == group
            group_spectral = spectral_data[mask]
            group_target = target_values[mask]
            
            if len(group_target) < 10:  # 样本太少的组跳过
                continue
            
            # 为该组训练模型
            X_train, X_test, y_train, y_test = train_test_split(
                group_spectral, group_target, test_size=0.2, random_state=42
            )
            
            # 使用PLS回归作为示例模型
            model = PLSRegression(n_components=10)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # 计算性能指标
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            group_results[group] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
            
            # 绘制该组的预测结果
            plt.subplot(n_rows, n_cols, idx)
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', lw=2)
            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.title(f'{group_name}={group}的预测结果')
            
            plt.text(0.05, 0.95, 
                    f'样本数: {len(group_target)}\n'
                    f'R2 = {r2:.3f}\n'
                    f'RMSE = {rmse:.3f}\n'
                    f'MAE = {mae:.3f}',
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        plt.tight_layout()
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        # 添加分组比较表格
        comparison_table = [[f'{group_name}', '样本数', 'R2得分', 'RMSE', 'MAE']]
        for group, metrics in group_results.items():
            comparison_table.append([
                str(group),
                str(len(spectral_data[group_values == group])),
                f"{metrics['R2']:.3f}",
                f"{metrics['RMSE']:.3f}",
                f"{metrics['MAE']:.3f}"
            ])
        
        self.add_table(comparison_table)

    def _analyze_by_time(self, spectral_data, target_values, time_values, 
                        target_name, time_name):
        """按时间进行预测分析"""
        self.add_heading(f"基于{time_name}的{target_name}时间序列预测分析", 4)
        
        # 将数据按时间排序
        sorted_indices = np.argsort(time_values)
        sorted_spectral = spectral_data[sorted_indices]
        sorted_target = target_values[sorted_indices]
        sorted_time = time_values[sorted_indices]
        
        # 按时间划分训练集和测试集（使用最后20%的数据作为测试集）
        split_idx = int(len(sorted_target) * 0.8)
        X_train = sorted_spectral[:split_idx]
        X_test = sorted_spectral[split_idx:]
        y_train = sorted_target[:split_idx]
        y_test = sorted_target[split_idx:]
        time_test = sorted_time[split_idx:]
        
        # 训练模型
        model = PLSRegression(n_components=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 计算性能指标
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # 绘制时间序列预测结果
        fig = plt.figure(figsize=(15, 8))
        plt.scatter(time_test, y_test, label='实际值', alpha=0.5)
        plt.scatter(time_test, y_pred, label='预测值', alpha=0.5)
        plt.xlabel(time_name)
        plt.ylabel(target_name)
        plt.title(f'{target_name}的时间序列预测')
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.text(0.05, 0.95, 
                f'R2 = {r2:.3f}\n'
                f'RMSE = {rmse:.3f}\n'
                f'MAE = {mae:.3f}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)

    def _plot_pairwise_relationships(self):
        """绘制数据集中变量的分组分布图"""
        # 获取所有数值型变量和分类变量
        numeric_data = {}
        categorical_data = {}
        for key, value in self.dataset.items():
            if key != '光谱':
                if pd.api.types.is_numeric_dtype(value):
                    numeric_data[key] = value
                else:
                    try:
                        pd.to_datetime(value)  # 尝试转换为日期
                        categorical_data[key] = value
                    except:
                        categorical_data[key] = value
        
        if len(numeric_data) < 1 or len(categorical_data) < 1:
            self.add_paragraph("数据集中缺少足够的数值型或分类变量,无法绘制分组分布图。")
            return
            
        # 创建数据框
        df = pd.DataFrame({**numeric_data, **categorical_data})
        
        # 为每个数值变量和分类变量的组合创建分布图
        for num_col in numeric_data.keys():
            for cat_col in categorical_data.keys():
                fig = plt.figure(figsize=(15, 6))
                
                # 创建子图
                plt.subplot(1, 2, 1)
                # 按组绘制核密度估计图
                for group in df[cat_col].unique():
                    group_data = df[df[cat_col] == group][num_col]
                    sns.kdeplot(data=group_data, label=str(group))
                plt.title(f'{num_col}在不同{cat_col}下的密度分布')
                plt.xlabel(num_col)
                plt.ylabel('密度')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                # 按组绘制小提琴图
                sns.violinplot(x=cat_col, y=num_col, data=df)
                plt.title(f'{num_col}在不同{cat_col}下的分布')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                
                # 添加到PDF
                self.pdf_elements.append(self.figure_to_image(fig))
                plt.close(fig)
                
                # 添加分组描述性统计
                self.add_paragraph(f"\n{num_col}按{cat_col}分组的统计描述:")
                stats_table = [['组别', '样本数', '均值', '标准差', '最小值', '25%分位数', '中位数', '75%分位数', '最大值']]
                
                for group in df[cat_col].unique():
                    group_data = df[df[cat_col] == group][num_col]
                    if len(group_data) > 0:
                        stats = group_data.describe()
                        stats_table.append([
                            str(group),
                            f"{stats['count']:.0f}",
                            f"{stats['mean']:.3f}",
                            f"{stats['std']:.3f}",
                            f"{stats['min']:.3f}",
                            f"{stats['25%']:.3f}",
                            f"{stats['50%']:.3f}",
                            f"{stats['75%']:.3f}",
                            f"{stats['max']:.3f}"
                        ])
                    
                self.add_table(stats_table)

    def _analyze_dataset_info(self):
        """分析数据集基本信息"""
        # 创建数据集信息表格
        dataset_info = [
            ['特征名称', '数据类型', '形状', '非空值数量'],
        ]
        
        for key, value in self.dataset.items():
            dataset_info.append([
                key,
                str(value.dtype),
                str(value.shape),
                str(np.sum(~pd.isna(value)))
            ])
        
        self.add_paragraph("数据集包含以下特征：")
        self.add_table(dataset_info)

    def _analyze_spectral_data(self):
        """分析光谱数据"""
        # 基本统计信息
        stats = {
            '样本数量': self.n_samples,
            '特征数量': self.n_features,
            '光谱数据统计': {
                '平均值': f"{np.mean(self.spectral_data):.4f}",
                '标准差': f"{np.std(self.spectral_data):.4f}",
                '最小值': f"{np.min(self.spectral_data):.4f}",
                '最大值': f"{np.max(self.spectral_data):.4f}"
            }
        }
        
        for key, value in stats.items():
            if isinstance(value, dict):
                self.add_paragraph(f"{key}:")
                for sub_key, sub_value in value.items():
                    self.add_paragraph(f"    {sub_key}: {sub_value}")
            else:
                self.add_paragraph(f"{key}: {value}")
        
        # 平均光谱图
        fig = plt.figure(figsize=(12, 6))
        mean_spectrum = np.mean(self.spectral_data, axis=0)
        std_spectrum = np.std(self.spectral_data, axis=0)
        
        plt.plot(mean_spectrum, 'b-', label='平均光谱')
        plt.fill_between(range(len(mean_spectrum)),
                        mean_spectrum - std_spectrum,
                        mean_spectrum + std_spectrum,
                        alpha=0.2,
                        color='b',
                        label='±1 标准差')
        
        plt.xlabel('波长索引')
        plt.ylabel('光谱强度')
        plt.title('平均光谱图及其变异范围')
        plt.legend()
        plt.grid(True)
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        # PCA分析
        self.add_heading("主成分分析 (PCA)", 3)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.spectral_data)
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(scaled_data)
        
        # 绘制解释方差比
        fig = plt.figure(figsize=(10, 5))
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        plt.plot(range(1, len(explained_variance_ratio) + 1), 
                cumulative_variance_ratio, 
                'bo-')
        plt.xlabel('主成分数量')
        plt.ylabel('累积解释方差比')
        plt.title('PCA累积解释方差比')
        plt.grid(True)
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        self.add_paragraph(f"前三个主成分解释方差比：")
        for i, ratio in enumerate(explained_variance_ratio[:3], 1):
            self.add_paragraph(f"PC{i}: {ratio:.4f}")

    def _analyze_other_features(self):
        """分析其他特征"""
        for key, value in self.dataset.items():
            if key != '光谱':
                self.add_heading(f"{key}特征分析", 3)
                
                if value.dtype.kind in 'iufc':  # 数值型数据
                    # 统计信息
                    stats = {
                        '平均值': np.mean(value),
                        '标准差': np.std(value),
                        '最小值': np.min(value),
                        '最大值': np.max(value),
                        '中位数': np.median(value)
                    }
                    
                    stats_table = [['统计量', '值']]
                    for stat_name, stat_value in stats.items():
                        stats_table.append([stat_name, f"{stat_value:.4f}"])
                    
                    self.add_table(stats_table)
                    
                    # 绘制分布图
                    fig = plt.figure(figsize=(10, 5))
                    plt.hist(value, bins=30, edgecolor='black')
                    plt.title(f"{key}分布直方图")
                    plt.xlabel(key)
                    plt.ylabel('频次')
                    plt.grid(True)
                    self.pdf_elements.append(self.figure_to_image(fig))
                    plt.close(fig)
                
                else:  # 类别型数据
                    # 统计每个类别的数量
                    value_counts = pd.Series(value).value_counts()
                    
                    counts_table = [['类别', '数量']]
                    for cat, count in value_counts.items():
                        counts_table.append([str(cat), str(count)])
                    
                    self.add_table(counts_table)
                    
                    # 绘制条形图
                    fig = plt.figure(figsize=(10, 5))
                    plt.bar(range(len(value_counts)), value_counts.values)
                    plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
                    plt.title(f"{key}类别分布")
                    plt.xlabel(key)
                    plt.ylabel('数量')
                    plt.grid(True)
                    plt.tight_layout()
                    self.pdf_elements.append(self.figure_to_image(fig))
                    plt.close(fig)
    def _analyze_spectral_details(self):
        """详细分析光谱数据特征"""
        # 1. 计算并绘制一阶导数和二阶导数
        spectra = self.spectral_data
        # 计算导数
        first_derivative = np.gradient(spectra, axis=1)
        second_derivative = np.gradient(first_derivative, axis=1)
        
        # 绘制导数图
        fig = plt.figure(figsize=(12, 8))
        
        # 原始光谱
        plt.subplot(3, 1, 1)
        mean_spectrum = np.mean(spectra, axis=0)
        std_spectrum = np.std(spectra, axis=0)
        plt.plot(mean_spectrum, 'b-', label='平均光谱')
        plt.fill_between(range(len(mean_spectrum)),
                        mean_spectrum - std_spectrum,
                        mean_spectrum + std_spectrum,
                        alpha=0.2,
                        color='b',
                        label='±1 标准差')
        plt.title('原始光谱')
        plt.xlabel('波长索引')
        plt.ylabel('光谱强度')
        plt.grid(True)
        plt.legend()
        
        # 一阶导数
        plt.subplot(3, 1, 2)
        mean_first_deriv = np.mean(first_derivative, axis=0)
        std_first_deriv = np.std(first_derivative, axis=0)
        plt.plot(mean_first_deriv, 'r-', label='平均一阶导数')
        plt.fill_between(range(len(mean_first_deriv)),
                        mean_first_deriv - std_first_deriv,
                        mean_first_deriv + std_first_deriv,
                        alpha=0.2,
                        color='r',
                        label='±1 标准差')
        plt.title('一阶导数')
        plt.xlabel('波长索引')
        plt.ylabel('一阶导数值')
        plt.grid(True)
        plt.legend()
        
        # 二阶导数
        plt.subplot(3, 1, 3)
        mean_second_deriv = np.mean(second_derivative, axis=0)
        std_second_deriv = np.std(second_derivative, axis=0)
        plt.plot(mean_second_deriv, 'g-', label='平均二阶导数')
        plt.fill_between(range(len(mean_second_deriv)),
                        mean_second_deriv - std_second_deriv,
                        mean_second_deriv + std_second_deriv,
                        alpha=0.2,
                        color='g',
                        label='±1 标准差')
        plt.title('二阶导数')
        plt.xlabel('波长索引')
        plt.ylabel('二阶导数值')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        # 2. 绘制光谱特征分布图
        fig = plt.figure(figsize=(12, 6))
        
        # 计算每个样本的统计特征
        mean_intensities = np.mean(spectra, axis=1)
        max_intensities = np.max(spectra, axis=1)
        min_intensities = np.min(spectra, axis=1)
        range_intensities = max_intensities - min_intensities
        
        # 创建箱线图
        data = [mean_intensities, max_intensities, min_intensities, range_intensities]
        labels = ['平均强度', '最大强度', '最小强度', '强度范围']
        
        plt.boxplot(data, labels=labels)
        plt.title('光谱特征分布')
        plt.ylabel('强度值')
        plt.grid(True)
        
        plt.tight_layout()
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        # 3. 添加统计信息到报告
        stats_table = [
            ['统计量', '平均值', '标准差', '最小值', '最大值', '中位数']
        ]
        
        features = {
            '原始光谱': spectra.mean(axis=1),
            '一阶导数': first_derivative.mean(axis=1),
            '二阶导数': second_derivative.mean(axis=1)
        }
        
        for name, values in features.items():
            stats = [
                name,
                f"{np.mean(values):.4f}",
                f"{np.std(values):.4f}",
                f"{np.min(values):.4f}",
                f"{np.max(values):.4f}",
                f"{np.median(values):.4f}"
            ]
            stats_table.append(stats)
        
        self.add_table(stats_table)
        
        # 4. 特征峰识别和标注
        peak_indices = scipy.signal.find_peaks(mean_spectrum)[0]
        valley_indices = scipy.signal.find_peaks(-mean_spectrum)[0]
        
        fig = plt.figure(figsize=(12, 6))
        plt.plot(mean_spectrum, 'b-', label='平均光谱')
        plt.plot(peak_indices, mean_spectrum[peak_indices], 'ro', label='峰值')
        plt.plot(valley_indices, mean_spectrum[valley_indices], 'go', label='谷值')
        
        # 标注主要峰值
        for idx in peak_indices:
            plt.annotate(f'Peak: {mean_spectrum[idx]:.2f}',
                        (idx, mean_spectrum[idx]),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=8)
        
        plt.title('光谱特征峰识别')
        plt.xlabel('波长索引')
        plt.ylabel('光谱强度')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        # 记录峰值信息
        self.add_paragraph("主要特征峰位置：")
        peak_info = [['峰值类型', '波长索引', '强度']]
        
        for idx in peak_indices:
            peak_info.append(['峰值', str(idx), f"{mean_spectrum[idx]:.4f}"])
        for idx in valley_indices:
            peak_info.append(['谷值', str(idx), f"{mean_spectrum[idx]:.4f}"])
        
        self.add_table(peak_info)
    def _analyze_correlations(self):
        """分析光谱特征与理化值之间的相关性"""
        # 准备数据
        chemical_features = {}
        for key, value in self.dataset.items():
            if key != '光谱' and value.dtype.kind in 'iufc':
                chemical_features[key] = value
        
        if chemical_features:
            # 创建相关性分析结果
            correlations = {}
            p_values = {}
            
            # 对每个理化指标进行分析
            for chem_name, chem_value in chemical_features.items():
                # 计算每个波长点与该理化值的相关系数
                wave_correlations = []
                wave_p_values = []
                
                for i in range(self.spectral_data.shape[1]):
                    # 使用scipy.stats计算相关系数和p值
                    corr, p_val = scipy.stats.pearsonr(
                        self.spectral_data[:, i],
                        chem_value
                    )
                    wave_correlations.append(corr)
                    wave_p_values.append(p_val)
                
                correlations[chem_name] = wave_correlations
                p_values[chem_name] = wave_p_values
            
            # 绘制相关性图
            n_chemicals = len(chemical_features)
            fig = plt.figure(figsize=(12, 4 * n_chemicals))
            
            for idx, (chem_name, correlation) in enumerate(correlations.items(), 1):
                plt.subplot(n_chemicals, 1, idx)
                
                # 绘制相关系数曲线
                plt.plot(correlation, 'b-', label='相关系数')
                
                # 标记显著性区域
                significant = np.array(p_values[chem_name]) < 0.05
                if np.any(significant):
                    plt.fill_between(
                        range(len(correlation)),
                        np.where(significant, correlation, np.nan),
                        alpha=0.3,
                        color='r',
                        label='p < 0.05'
                    )
                
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                plt.axhline(y=0.5, color='g', linestyle=':', alpha=0.5)
                plt.axhline(y=-0.5, color='g', linestyle=':', alpha=0.5)
                
                plt.title(f'光谱与{chem_name}的相关性分析')
                plt.xlabel('波长索引')
                plt.ylabel('相关系数')
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            plt.tight_layout()
            self.pdf_elements.append(self.figure_to_image(fig))
            plt.close(fig)
            
            # 添加文字说明
            for chem_name, correlation in correlations.items():
                # 找出最强相关的波长点
                max_corr_idx = np.argmax(np.abs(correlation))
                max_corr = correlation[max_corr_idx]
                max_corr_p = p_values[chem_name][max_corr_idx]
                
                self.add_paragraph(f"{chem_name}相关性分析结果：")
                self.add_paragraph(
                    f"最强相关波长索引：{max_corr_idx}，"
                    f"相关系数：{max_corr:.4f}，"
                    f"p值：{max_corr_p:.4e}"
                )
                
                # 统计显著相关的波长数量
                sig_count = np.sum(np.array(p_values[chem_name]) < 0.05)
                self.add_paragraph(
                    f"显著相关(p<0.05)的波长点数量：{sig_count}，"
                    f"占总波长点的{sig_count/len(correlation)*100:.2f}%"
                )

    def _analyze_temporal_patterns(self):
        """分析时间模式"""
        if '采集日期' not in self.dataset:
            raise ValueError("数据集中缺少'采集日期'信息")
            
        # 将日期转换为datetime对象
        dates = pd.to_datetime(self.dataset['采集日期'])
        self.dates = dates
        # 计算每日平均光谱  
        daily_means = pd.DataFrame({
            'date': dates,
            'mean_intensity': np.mean(self.dataset['光谱'], axis=1)
        })
        # 按日期分组并计算统计量
        daily_stats = daily_means.groupby('date').agg({
            'mean_intensity': ['mean', 'std', 'count']
        }).reset_index()
        # 绘制时间序列图
        fig = plt.figure(figsize=(15, 6))
        plt.errorbar(daily_stats['date'],
                    daily_stats['mean_intensity']['mean'],
                    yerr=daily_stats['mean_intensity']['std'],
                    fmt='o-',
                    capsize=5)
        plt.xlabel('日期')
        plt.ylabel('平均光谱强度')
        plt.title('光谱强度随时间的变化')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        
        # 将图形添加到PDF文档
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        return daily_stats
    
    def _analyze_volunteer_patterns(self):
        """分析志愿者模式"""
        if '志愿者' not in self.dataset:
            raise ValueError("数据集中缺少'志愿者'信息")
            
        # 计算每个志愿者的平均光谱
        volunteer_means = pd.DataFrame({
            'volunteer': self.dataset['志愿者'],
            'mean_intensity': np.mean(self.dataset['光谱'], axis=1)
        })
        
        # 创建志愿者统计信息
        volunteer_stats = volunteer_means.groupby('volunteer').agg({
            'mean_intensity': ['mean', 'std', 'count']
        }).reset_index()
        
        # 绘制志愿者箱线图
        fig = plt.figure(figsize=(15, 6))
        sns.boxplot(data=volunteer_means, x='volunteer', y='mean_intensity')
        plt.xlabel('志愿者ID')
        plt.ylabel('平均光谱强度')
        plt.title('各志愿者光谱强度分布')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        
        # 将图形添加到PDF文档
        self.pdf_elements.append(self.figure_to_image(fig))
        plt.close(fig)
        
        return volunteer_stats

def generate_analysis_report(dataset, output_path='data_analysis_report.pdf'):
    """
    生成数据分析报告的便捷函数
    
    Parameters:
    -----------
    dataset : dict
        数据集字典
    output_path : str, optional
        输出PDF文件路径
        
    Returns:
    --------
    str or None
        成功则返回报告路径，失败则返回None
    """
    try:
        analyzer = SpectralAnalysisReport(dataset, output_path)
        analyzer.analyze_and_generate_report()
        return output_path
    except Exception as e:
        print(f"生成报告时发生错误: {str(e)}")
        print(f"错误发生在: {e.__traceback__.tb_frame.f_code.co_filename} 第 {e.__traceback__.tb_lineno} 行")
        return None

def search_system_fonts():
    """
    搜索系统中的中文字体
    
    Returns:
    --------
    list
        可用的中文字体文件路径列表
    """
    font_paths = []
    
    # Windows 字体路径
    if os.name == 'nt':
        windows_font_path = os.path.join(os.environ['SystemRoot'], 'Fonts')
        font_files = ['msyh.ttc', 'simsun.ttc', 'simhei.ttf']
        for font_file in font_files:
            full_path = os.path.join(windows_font_path, font_file)
            if os.path.exists(full_path):
                font_paths.append(full_path)
    
    # Linux 字体路径
    elif os.name == 'posix':
        linux_font_paths = [
            '/usr/share/fonts',
            '/usr/local/share/fonts',
            os.path.expanduser('~/.fonts')
        ]
        for path in linux_font_paths:
            if os.path.exists(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.endswith(('.ttc', '.ttf')):
                            font_paths.append(os.path.join(root, file))
    
    # macOS 字体路径
    elif sys.platform == 'darwin':
        mac_font_paths = [
            '/System/Library/Fonts',
            '/Library/Fonts',
            os.path.expanduser('~/Library/Fonts')
        ]
        for path in mac_font_paths:
            if os.path.exists(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.endswith(('.ttc', '.ttf')):
                            font_paths.append(os.path.join(root, file))
    
    return font_paths

# 使用示例
if __name__ == "__main__":
    now_time = datetime.datetime.now()
    try:
        # 加载数据
        from nirapi.load_data import *
        
        # 从数据库获取数据集
        dataset_X = get_dataset_from_mysql(database='光谱数据库',table_name="复享光谱仪", project_name="多发光单收光探头血糖数据", X_type=['光谱',"采集日期","志愿者"])

        
        # 生成报告
        output_path = '数据分析报告.pdf'
        report_path = generate_analysis_report(dataset_X, output_path)
        
        if report_path:
            print(f"报告已成功生成: {report_path}")
            
            # 尝试自动打开生成的PDF文件
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(report_path)
                elif sys.platform == 'darwin':  # macOS
                    os.system(f'open {report_path}')
                else:  # Linux
                    os.system(f'xdg-open {report_path}')
            except Exception as e:
                print(f"无法自动打开PDF文件: {str(e)}")
                print(f"请手动打开文件: {report_path}")
        else:
            print("报告生成失败")
            
    except Exception as e:
        print(f"程序执行过程中发生错误: {str(e)}")
        
    finally:
        # 清理matplotlib图形
        plt.close('all')
        print(f"程序执行时间: {(datetime.datetime.now() - now_time).total_seconds() / 60:.2f}分钟")