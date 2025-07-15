"""
Unit tests for preprocessing functions.
"""

import unittest
import numpy as np
import pandas as pd
from nirapi.preprocessing import SNV, normalization, MC, remove_outliers


class TestPreprocessing(unittest.TestCase):
    """Test cases for preprocessing functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample spectral data
        self.X = np.array([
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8]
        ])
        
        # Create data with outliers
        self.X_outliers = np.array([
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [100, 200, 300, 400, 500],  # Outlier
            [4, 5, 6, 7, 8]
        ])
    
    def test_SNV_basic(self):
        """Test basic SNV functionality."""
        result = SNV(self.X)
        
        # Check shape is preserved
        self.assertEqual(result.shape, self.X.shape)
        
        # Check result is numeric
        self.assertTrue(np.isfinite(result).all())
        
        # Check that each row has approximately zero mean and unit variance
        for i in range(result.shape[0]):
            self.assertAlmostEqual(np.mean(result[i]), 0, places=10)
            self.assertAlmostEqual(np.std(result[i]), 1, places=10)
    
    def test_SNV_single_spectrum(self):
        """Test SNV with single spectrum."""
        single_spectrum = self.X[0]
        result = SNV(single_spectrum)
        
        # Should return 2D array
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], len(single_spectrum))
    
    def test_SNV_with_replace_wave(self):
        """Test SNV with replace_wave parameter."""
        replace_wave = [1, 3]
        result = SNV(self.X, replace_wave=replace_wave)
        
        # Check shape is preserved
        self.assertEqual(result.shape, self.X.shape)
        
        # Check result is numeric
        self.assertTrue(np.isfinite(result).all())
    
    def test_normalization_basic(self):
        """Test basic normalization functionality."""
        result = normalization(self.X)
        
        # Check shape is preserved
        self.assertEqual(result.shape, self.X.shape)
        
        # Check values are between 0 and 1
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
    
    def test_normalization_axis_0(self):
        """Test normalization along axis 0."""
        result = normalization(self.X, axis=0)
        
        # Check shape is preserved
        self.assertEqual(result.shape, self.X.shape)
        
        # Check each column is normalized
        for j in range(result.shape[1]):
            col = result[:, j]
            self.assertAlmostEqual(np.min(col), 0, places=10)
            self.assertAlmostEqual(np.max(col), 1, places=10)
    
    def test_MC_basic(self):
        """Test mean centering functionality."""
        result = MC(self.X)
        
        # Check shape is preserved
        self.assertEqual(result.shape, self.X.shape)
        
        # Check overall mean is approximately zero
        self.assertAlmostEqual(np.mean(result), 0, places=10)
    
    def test_MC_axis_0(self):
        """Test mean centering along axis 0."""
        result = MC(self.X, axis=0)
        
        # Check shape is preserved
        self.assertEqual(result.shape, self.X.shape)
        
        # Check each column mean is approximately zero
        for j in range(result.shape[1]):
            self.assertAlmostEqual(np.mean(result[:, j]), 0, places=10)
    
    def test_remove_outliers_basic(self):
        """Test outlier removal functionality."""
        result = remove_outliers(self.X_outliers, threshold=2)
        
        # Check shape is preserved
        self.assertEqual(result.shape, self.X_outliers.shape)
        
        # Check that outliers are replaced with NaN
        self.assertTrue(np.isnan(result).any())
    
    def test_remove_outliers_axis_0(self):
        """Test outlier removal along axis 0."""
        result = remove_outliers(self.X_outliers, threshold=2, axis=0)
        
        # Check shape is preserved
        self.assertEqual(result.shape, self.X_outliers.shape)
        
        # Check that outliers are replaced with NaN
        self.assertTrue(np.isnan(result).any())
    
    def test_remove_outliers_axis_1(self):
        """Test outlier removal along axis 1."""
        result = remove_outliers(self.X_outliers, threshold=2, axis=1)
        
        # Check shape is preserved
        self.assertEqual(result.shape, self.X_outliers.shape)
        
        # Check that outliers are replaced with NaN
        self.assertTrue(np.isnan(result).any())


if __name__ == '__main__':
    unittest.main() 