"""
Unit tests for model_class functions.
"""

import unittest
import numpy as np
from nirapi.model_class import SNV, SNVTransformer, NormalizeTransformer, EmscScaler


class TestModelClass(unittest.TestCase):
    """Test cases for model_class functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample spectral data
        self.X = np.array([
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8]
        ])
    
    def test_SNV_function_basic(self):
        """Test basic SNV function."""
        result = SNV(self.X)
        
        # Check shape is preserved
        self.assertEqual(result.shape, self.X.shape)
        
        # Check result is numeric
        self.assertTrue(np.isfinite(result).all())
    
    def test_SNV_function_validation(self):
        """Test SNV function input validation."""
        # Test with None input
        with self.assertRaises(ValueError):
            SNV(None)
        
        # Test with empty array
        with self.assertRaises(ValueError):
            SNV(np.array([]))
        
        # Test with 3D array
        with self.assertRaises(ValueError):
            SNV(np.array([[[1, 2], [3, 4]]]))
    
    def test_SNV_function_replace_wave_validation(self):
        """Test SNV function replace_wave validation."""
        # Test with invalid replace_wave type
        with self.assertRaises(TypeError):
            SNV(self.X, replace_wave="invalid")
        
        # Test with wrong number of elements
        with self.assertRaises(ValueError):
            SNV(self.X, replace_wave=[1])
        
        # Test with negative indices
        with self.assertRaises(ValueError):
            SNV(self.X, replace_wave=[-1, 2])
        
        # Test with start >= end
        with self.assertRaises(ValueError):
            SNV(self.X, replace_wave=[3, 2])
        
        # Test with end > number of features
        with self.assertRaises(ValueError):
            SNV(self.X, replace_wave=[1, 10])
    
    def test_SNVTransformer_basic(self):
        """Test SNVTransformer basic functionality."""
        transformer = SNVTransformer()
        
        # Test fit
        transformer.fit(self.X)
        
        # Test transform
        result = transformer.transform(self.X)
        
        # Check shape is preserved
        self.assertEqual(result.shape, self.X.shape)
        
        # Check result is numeric
        self.assertTrue(np.isfinite(result).all())
    
    def test_SNVTransformer_fit_transform(self):
        """Test SNVTransformer fit_transform method."""
        transformer = SNVTransformer()
        result = transformer.fit_transform(self.X)
        
        # Check shape is preserved
        self.assertEqual(result.shape, self.X.shape)
        
        # Check result is numeric
        self.assertTrue(np.isfinite(result).all())
    
    def test_NormalizeTransformer_basic(self):
        """Test NormalizeTransformer basic functionality."""
        transformer = NormalizeTransformer()
        
        # Test fit
        transformer.fit(self.X)
        
        # Test transform
        result = transformer.transform(self.X)
        
        # Check shape is preserved
        self.assertEqual(result.shape, self.X.shape)
        
        # Check values are between 0 and 1
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
    
    def test_NormalizeTransformer_fit_transform(self):
        """Test NormalizeTransformer fit_transform method."""
        transformer = NormalizeTransformer()
        result = transformer.fit_transform(self.X)
        
        # Check shape is preserved
        self.assertEqual(result.shape, self.X.shape)
        
        # Check values are between 0 and 1
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
    
    def test_EmscScaler_basic(self):
        """Test EmscScaler basic functionality."""
        scaler = EmscScaler()
        
        # Test fit
        scaler.fit(self.X)
        
        # Test transform
        result = scaler.transform(self.X)
        
        # Check shape is preserved
        self.assertEqual(result.shape, self.X.shape)
        
        # Check result is numeric
        self.assertTrue(np.isfinite(result).all())
    
    def test_EmscScaler_fit_transform(self):
        """Test EmscScaler fit_transform method."""
        scaler = EmscScaler()
        result = scaler.fit_transform(self.X)
        
        # Check shape is preserved
        self.assertEqual(result.shape, self.X.shape)
        
        # Check result is numeric
        self.assertTrue(np.isfinite(result).all())
    
    def test_EmscScaler_transform_before_fit(self):
        """Test EmscScaler transform before fit raises error."""
        scaler = EmscScaler()
        
        with self.assertRaises(ValueError):
            scaler.transform(self.X)


if __name__ == '__main__':
    unittest.main() 