"""
Wavelength and spectral band utilities for NIR spectroscopy.

This module provides functions for managing wavelength ranges,
spectral bands, and related utilities for different spectrometer types.
"""

import numpy as np
from typing import List, Tuple


__all__ = [
    'get_MZI_bands',
    'get_wavelength_ranges',
    'validate_wavelength_range'
]


def get_MZI_bands() -> List[Tuple[float, float]]:
    """
    Get the wavelength bands for MZI (Mach-Zehnder Interferometer) prototype.
    
    Returns:
        List of tuples containing (start_wavelength, end_wavelength) for each band
        
    Example:
        >>> bands = get_MZI_bands()
        >>> print(f"Number of bands: {len(bands)}")
        >>> print(f"First band: {bands[0]}")
    """
    # This would contain the actual MZI band definitions
    # Extracted from the original utils.py
    bands = [
        (1000.0, 1100.0),
        (1100.0, 1200.0), 
        (1200.0, 1300.0),
        (1300.0, 1400.0)
    ]
    return bands


def get_wavelength_ranges(spectrometer_type: str) -> List[Tuple[float, float]]:
    """
    Get wavelength ranges for different spectrometer types.
    
    Args:
        spectrometer_type: Type of spectrometer ('MZI', 'FT', 'FX', etc.)
        
    Returns:
        List of wavelength ranges for the specified spectrometer
        
    Raises:
        ValueError: If spectrometer_type is not supported
    """
    ranges_map = {
        'MZI': get_MZI_bands(),
        'FT': [(900.0, 1700.0)],  # Example range
        'FX': [(400.0, 1000.0)]   # Example range
    }
    
    if spectrometer_type not in ranges_map:
        raise ValueError(f"Unsupported spectrometer type: {spectrometer_type}")
    
    return ranges_map[spectrometer_type]


def validate_wavelength_range(
    wavelengths: np.ndarray, 
    min_wavelength: float, 
    max_wavelength: float
) -> np.ndarray:
    """
    Validate and filter wavelengths within specified range.
    
    Args:
        wavelengths: Array of wavelength values
        min_wavelength: Minimum allowed wavelength
        max_wavelength: Maximum allowed wavelength
        
    Returns:
        Boolean mask indicating valid wavelengths
        
    Example:
        >>> wavelengths = np.array([900, 1000, 1100, 1200, 1800])
        >>> mask = validate_wavelength_range(wavelengths, 950, 1150)
        >>> valid_wavelengths = wavelengths[mask]
    """
    return (wavelengths >= min_wavelength) & (wavelengths <= max_wavelength)