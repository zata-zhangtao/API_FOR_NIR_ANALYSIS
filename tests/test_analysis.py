"""
Unit tests for analysis functions.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from nirapi.analysis import print_basic_data_info, plot_correlation_graph


class TestAnalysis(unittest.TestCase):
    """Test cases for analysis functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data
        self.data_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6],
            'feature3': [3, 4, 5, 6, 7],
            'target': [10, 20, 30, 40, 50]
        })
        
        self.X = self.data_df.iloc[:, :-1]
        self.y = self.data_df.iloc[:, -1]
    
    def test_print_basic_data_info_basic(self):
        """Test basic functionality of print_basic_data_info."""
        # This should not raise any exceptions
        try:
            print_basic_data_info(self.data_df, self.X, self.y)
        except Exception as e:
            self.fail(f"print_basic_data_info raised an exception: {e}")
    
    def test_print_basic_data_info_validation(self):
        """Test input validation for print_basic_data_info."""
        # Test with wrong types
        with self.assertRaises(TypeError):
            print_basic_data_info("not_a_dataframe", self.X, self.y)
        
        with self.assertRaises(TypeError):
            print_basic_data_info(self.data_df, "not_a_dataframe", self.y)
        
        with self.assertRaises(TypeError):
            print_basic_data_info(self.data_df, self.X, "not_a_series")
        
        # Test with empty data
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            print_basic_data_info(empty_df, self.X, self.y)
        
        empty_X = pd.DataFrame()
        with self.assertRaises(ValueError):
            print_basic_data_info(self.data_df, empty_X, self.y)
        
        empty_y = pd.Series([], dtype=float)
        with self.assertRaises(ValueError):
            print_basic_data_info(self.data_df, self.X, empty_y)
    
    def test_plot_correlation_graph_basic(self):
        """Test basic functionality of plot_correlation_graph."""
        X_array = self.X.values
        y_array = self.y.values
        
        # This should not raise any exceptions
        try:
            plot_correlation_graph(X_array, y_array)
        except Exception as e:
            self.fail(f"plot_correlation_graph raised an exception: {e}")
    
    def test_plot_correlation_graph_with_save(self):
        """Test plot_correlation_graph with save directory."""
        X_array = self.X.values
        y_array = self.y.values
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # This should not raise any exceptions
            try:
                plot_correlation_graph(X_array, y_array, save_dir=temp_dir)
            except Exception as e:
                self.fail(f"plot_correlation_graph with save_dir raised an exception: {e}")


if __name__ == '__main__':
    unittest.main() 