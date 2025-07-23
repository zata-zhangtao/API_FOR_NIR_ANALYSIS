# Generic Data Loading Functions

This document describes the new generic data loading functions that provide flexible and efficient ways to load data from Excel files with caching support.

## Overview

The generic data loading functions were refactored from the original `load_skin_color_ft_data` function to provide a more flexible and reusable approach to loading spectral data from Excel files. These functions support:

- **Flexible sheet configuration**: Define how each sheet should be loaded
- **Automatic caching**: Improve performance by caching loaded data
- **Data processing**: Apply custom transformations to loaded data
- **Multiple data formats**: Support different Excel file structures

## Functions

### load_excel_data_with_cache

The core generic function for loading Excel data with flexible configuration.

```python
def load_excel_data_with_cache(
    file_path: str, 
    sheet_configs: dict, 
    use_cache: bool = True,
    cache_dir: str = "cache",
    data_processors: dict = None,
    verbose: bool = True
) -> dict:
```

#### Parameters

- **file_path** (str): Path to the Excel file
- **sheet_configs** (dict): Configuration for each sheet to load
- **use_cache** (bool): Whether to use cached data if available (default: True)
- **cache_dir** (str): Directory to store cache files (default: "cache")
- **data_processors** (dict): Optional post-processing functions for each sheet
- **verbose** (bool): Whether to print loading information (default: True)

#### Sheet Configuration Format

```python
sheet_configs = {
    'sheet_name': {
        'header': int or None,      # Header row (default: 0)
        'usecols': str or list,     # Columns to use (optional)
        'skiprows': int or list,    # Rows to skip (optional)
        'nrows': int,               # Number of rows to read (optional)
        'index_col': int or str,    # Column to use as index (optional)
        'transpose': bool,          # Whether to transpose the data (default: False)
        'slice_rows': tuple,        # (start, end) for row slicing after loading (optional)
        'slice_cols': tuple,        # (start, end) for column slicing after loading (optional)
    }
}
```

#### Example Usage

```python
from nirapi.load_data import load_excel_data_with_cache

# Define sheet configurations
sheet_configs = {
    '光谱': {
        'header': None,
        'transpose': True,
        'slice_rows': (2, None)  # Skip first 2 rows after transpose
    },
    '理化值': {
        'header': 0,
        'slice_cols': (1, 6)  # Columns 1-5
    }
}

# Define data processors
def process_spectral_data(df):
    return df.apply(pd.to_numeric, errors='coerce').fillna(0)

data_processors = {
    '光谱': process_spectral_data
}

# Load data
data = load_excel_data_with_cache(
    'data.xlsx', 
    sheet_configs, 
    data_processors=data_processors
)

# Access loaded data
spectral_data = data['光谱']
biomarks_data = data['理化值']
```

### load_skin_color_ft_data

Specialized function for loading skin color FT spectroscopy data.

```python
def load_skin_color_ft_data(
    file_path: str, 
    use_cache: bool = True, 
    cache_dir: str = "cache"
) -> tuple:
```

#### Parameters

- **file_path** (str): Path to the Excel file
- **use_cache** (bool): Whether to use cached data if available (default: True)
- **cache_dir** (str): Directory to store cache files (default: "cache")

#### Returns

- **tuple**: (PD_sample, PD_source, biomarks) where:
  - PD_sample: pd.DataFrame with spectral data (numeric, NaN filled with 0)
  - PD_source: pd.DataFrame with source data (same as PD_sample)
  - biomarks: pd.DataFrame with biomarker data (columns 1-5)

#### Expected Excel Format

- **'光谱' sheet**: Spectral data with 2 header rows, will be transposed
- **'理化值' sheet**: Biomarker data with headers, columns 1-5 will be extracted

#### Example Usage

```python
from nirapi.load_data import load_skin_color_ft_data

# Load skin color FT data
PD_sample, PD_source, biomarks = load_skin_color_ft_data('skin_data.xlsx')

print(f"Spectral data shape: {PD_sample.shape}")
print(f"Biomarks shape: {biomarks.shape}")
```

### load_prototype_spectral_data

Specialized function for loading prototype spectral data.

```python
def load_prototype_spectral_data(
    file_path: str, 
    use_cache: bool = True, 
    cache_dir: str = "cache"
) -> dict:
```

#### Parameters

- **file_path** (str): Path to the Excel file
- **use_cache** (bool): Whether to use cached data if available (default: True)
- **cache_dir** (str): Directory to store cache files (default: "cache")

#### Returns

- **dict**: Dictionary containing all loaded sheets:
  - 'PD Sample': pd.DataFrame with PD sample data
  - 'PD Source': pd.DataFrame with PD source data  
  - 'Measured_Value': pd.DataFrame with measured values (excluding first column)

#### Expected Excel Format

- **'PD Sample' sheet**: Sample photodiode data with headers
- **'PD Source' sheet**: Source photodiode data with headers
- **'Measured_Value' sheet**: Measured values with headers, first column excluded

#### Example Usage

```python
from nirapi.load_data import load_prototype_spectral_data

# Load prototype spectral data
data = load_prototype_spectral_data('prototype_data.xlsx')

pd_sample = data['PD Sample']
pd_source = data['PD Source']
measured_values = data['Measured_Value']
```

## Caching Mechanism

All functions support automatic caching to improve performance:

1. **Cache file naming**: Based on file path and configuration hash
2. **Cache validation**: Checks if cache is newer than source file
3. **Automatic cache creation**: Saves data after first load
4. **Cache directory**: Configurable cache directory (default: "cache")

### Cache File Format

Cache files are named using the pattern:
```
{base_filename}_{config_hash}_cache.pkl
```

Where:
- `base_filename`: Excel filename without extension
- `config_hash`: 8-character hash of the sheet configuration
- Cache files are stored in pickle format

## Migration from Original Function

If you were using the original `load_skin_color_ft_data` function, you can easily migrate:

### Before (Original)
```python
def load_skin_color_ft_data(file_path: str, use_cache: bool = True):
    # Original implementation
    pass
```

### After (New)
```python
from nirapi.load_data import load_skin_color_ft_data

# Same interface, enhanced functionality
PD_sample, PD_source, biomarks = load_skin_color_ft_data(file_path, use_cache=True)
```

## Best Practices

1. **Use caching**: Enable caching for better performance with large files
2. **Configure cache directory**: Set a dedicated cache directory for your project
3. **Custom processors**: Use data processors for consistent data transformations
4. **Error handling**: Always wrap data loading in try-catch blocks
5. **Validate data**: Check loaded data shapes and types after loading

## Error Handling

```python
try:
    data = load_excel_data_with_cache(file_path, sheet_configs)
    # Validate data
    assert 'spectral_data' in data, "Missing spectral data"
    assert data['spectral_data'] is not None, "Failed to load spectral data"
except Exception as e:
    print(f"Error loading data: {e}")
```

## Performance Considerations

- **First load**: May take longer due to Excel parsing and caching
- **Subsequent loads**: Much faster when using cache
- **Memory usage**: Large datasets are cached to disk, not memory
- **Cache cleanup**: Manually remove cache files when no longer needed
