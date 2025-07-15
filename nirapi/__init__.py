"""
NIR API - A Near-Infrared Spectroscopy Analysis Package

This package provides tools for Near-Infrared spectroscopy analysis including:
- Data loading and preprocessing
- Machine learning models
- Visualization tools
- Analysis utilities
"""

__version__ = "1.0.0"
__author__ = "zata"
__description__ = "A Near-Infrared Spectroscopy Analysis API"

# Import main modules for easier access - only if dependencies are available
try:
    from . import utils
    from . import load_data
    from . import preprocessing
    from . import draw
    from . import analysis
    from . import ML_model
    from . import featsec
    from . import model_class
    
    # Make AnalysisClass available
    from . import AnalysisClass
    
    __all__ = [
        'utils',
        'load_data', 
        'preprocessing',
        'draw',
        'analysis',
        'ML_model',
        'featsec',
        'model_class',
        'AnalysisClass'
    ]
except ImportError:
    # If dependencies are not available, provide a minimal interface
    __all__ = []
    
    def __getattr__(name):
        """Lazy import for modules when dependencies are available."""
        if name in ['utils', 'load_data', 'preprocessing', 'draw', 'analysis', 
                   'ML_model', 'featsec', 'model_class', 'AnalysisClass']:
            try:
                import importlib
                module = importlib.import_module(f'.{name}', __name__)
                return module
            except ImportError as e:
                raise ImportError(f"Module {name} requires additional dependencies: {e}")
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 