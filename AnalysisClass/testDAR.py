import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from nirapi.AnalysisClass.DataAnalysisReport import SpectralAnalysisReport  # 假设原始代码在 spectral_analysis.py 中


class TestSpectralAnalysis:

    @staticmethod
    def generate_sample_data(n_samples=100, n_wavelengths=50):
        """生成测试用的样本数据"""
        # 生成光谱数据
        spectral_data = np.random.normal(0, 1, (n_samples, n_wavelengths))
        
        # 生成日期数据
        base_date = datetime.now()
        dates = [base_date + timedelta(days=i) for i in range(n_samples)]
        
        # 生成志愿者ID
        volunteers = [f"V{i}" for i in np.random.randint(1, 6, n_samples)]
        
        # 生成一些数值型特征
        feature1 = np.random.normal(100, 15, n_samples)
        feature2 = np.random.uniform(0, 10, n_samples)
        
        # 创建数据集字典
        dataset = {
            '光谱': spectral_data,
            '采集日期': dates,
            '志愿者': volunteers,
            '特征1': feature1,
            '特征2': feature2
        }
        
        return dataset

    def test_plot_data_relationships(self):
        """测试数据关系可视化函数"""
        print("\n测试 _plot_data_relationships 函数...")
        try:
            # 生成测试数据
            dataset = self.generate_sample_data()
            
            # 创建报告对象
            report = SpectralAnalysisReport(dataset, 'test_relationships.pdf')
            
            # 测试函数
            report._plot_data_relationships()
            print("✓ 数据关系可视化测试成功")
        except Exception as e:
            print(f"✗ 数据关系可视化测试失败: {str(e)}")

    def test_analyze_spectral_data(self):
        """测试光谱数据分析函数"""
        print("\n测试 _analyze_spectral_data 函数...")
        try:
            dataset = self.generate_sample_data()
            report = SpectralAnalysisReport(dataset, 'test_spectral.pdf')
            report._analyze_spectral_data()
            print("✓ 光谱数据分析测试成功")
        except Exception as e:
            print(f"✗ 光谱数据分析测试失败: {str(e)}")

    def test_analyze_spectral_details(self):
        """测试光谱详细分析函数"""
        print("\n测试 _analyze_spectral_details 函数...")
        try:
            dataset = self.generate_sample_data()
            report = SpectralAnalysisReport(dataset, 'test_details.pdf')
            report._analyze_spectral_details()
            print("✓ 光谱详细分析测试成功")
        except Exception as e:
            print(f"✗ 光谱详细分析测试失败: {str(e)}")

    def test_analyze_correlations(self):
        """测试相关性分析函数"""
        print("\n测试 _analyze_correlations 函数...")
        try:
            dataset = self.generate_sample_data()
            report = SpectralAnalysisReport(dataset, 'test_correlations.pdf')
            report._analyze_correlations()
            print("✓ 相关性分析测试成功")
        except Exception as e:
            print(f"✗ 相关性分析测试失败: {str(e)}")

    def test_analyze_temporal_patterns(self):
        """测试时间模式分析函数"""
        print("\n测试 _analyze_temporal_patterns 函数...")
        try:
            dataset = self.generate_sample_data()
            report = SpectralAnalysisReport(dataset, 'test_temporal.pdf')
            report._analyze_temporal_patterns()
            print("✓ 时间模式分析测试成功")
        except Exception as e:
            print(f"✗ 时间模式分析测试失败: {str(e)}")

    def test_analyze_volunteer_patterns(self):
        """测试志愿者模式分析函数"""
        print("\n测试 _analyze_volunteer_patterns 函数...")
        try:
            dataset = self.generate_sample_data()
            report = SpectralAnalysisReport(dataset, 'test_volunteer.pdf')
            report._analyze_volunteer_patterns()
            print("✓ 志愿者模式分析测试成功")
        except Exception as e:
            print(f"✗ 志愿者模式分析测试失败: {str(e)}")

    def test_edge_cases(self):
        """测试边界情况"""
        print("\n测试边界情况...")
        
        # 测试空数据集
        try:
            empty_dataset = {'光谱': np.array([])}
            report = SpectralAnalysisReport(empty_dataset, 'test_empty.pdf')
            print("✓ 空数据集处理测试成功")
        except Exception as e:
            print(f"✗ 空数据集处理测试失败: {str(e)}")
        
        # 测试缺失值
        try:
            dataset = self.generate_sample_data()
            dataset['特征1'][0] = np.nan
            report = SpectralAnalysisReport(dataset, 'test_missing.pdf')
            print("✓ 缺失值处理测试成功")
        except Exception as e:
            print(f"✗ 缺失值处理测试失败: {str(e)}")
        
        # 测试异常值
        try:
            dataset = self.generate_sample_data()
            dataset['特征1'][0] = 1e9  # 添加一个极端值
            report = SpectralAnalysisReport(dataset, 'test_outlier.pdf')
            print("✓ 异常值处理测试成功")
        except Exception as e:
            print(f"✗ 异常值处理测试失败: {str(e)}")

def run_all_tests():
    """运行所有测试"""
    tester = TestSpectralAnalysis()
    
    # 运行所有测试函数
    test_functions = [func for func in dir(tester) if func.startswith('test_')]
    for func_name in test_functions:
        getattr(tester, func_name)()

    # tester = TestSpectralAnalysis()

    # # 测试单个函数
    # tester.test_plot_data_relationships()  # 测试数据关系可视化
    # tester.test_analyze_spectral_data()    # 测试光谱数据分析
    # tester.test_analyze_correlations()     # 测试相关性分析

    #     # 测试边界情况
    # tester.test_edge_cases()

if __name__ == "__main__":
    run_all_tests()