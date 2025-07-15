"""
File and utility functions for NIR spectroscopy analysis.

This module provides various utility functions for file operations,
function introspection, and general helper utilities.
"""

import ast
import inspect
from typing import Dict, List, Any, Callable


__all__ = [
    'get_pythonFile_functions',
    'extract_function_info',
    'validate_function_parameters'
]


def get_pythonFile_functions(module_or_path: Any) -> Dict[str, Dict[str, Any]]:
    """
    Extract all functions from a Python file or module.
    
    This function analyzes a Python file or imported module and returns
    information about all defined functions including their signatures,
    docstrings, and other metadata.
    
    Args:
        module_or_path: Either a module object or path to Python file
        
    Returns:
        Dictionary mapping function names to their metadata
        
    Example:
        >>> import nirapi.analysis as analysis_module
        >>> functions = get_pythonFile_functions(analysis_module)
        >>> print(f"Found {len(functions)} functions")
        >>> for name, info in functions.items():
        ...     print(f"{name}: {info['signature']}")
    """
    functions_info = {}
    
    if isinstance(module_or_path, str):
        # Parse file path
        try:
            with open(module_or_path, 'r', encoding='utf-8') as file:
                tree = ast.parse(file.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'line_number': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node),
                        'is_async': isinstance(node, ast.AsyncFunctionDef)
                    }
                    functions_info[node.name] = func_info
                    
        except Exception as e:
            print(f"Error parsing file {module_or_path}: {e}")
            
    else:
        # Analyze module object
        try:
            for name, obj in inspect.getmembers(module_or_path):
                if inspect.isfunction(obj):
                    func_info = extract_function_info(obj)
                    functions_info[name] = func_info
                    
        except Exception as e:
            print(f"Error analyzing module: {e}")
    
    return functions_info


def extract_function_info(func: Callable) -> Dict[str, Any]:
    """
    Extract detailed information from a function object.
    
    Args:
        func: Function object to analyze
        
    Returns:
        Dictionary containing function metadata
    """
    try:
        signature = inspect.signature(func)
        source_lines = inspect.getsourcelines(func)
        
        info = {
            'name': func.__name__,
            'signature': str(signature),
            'docstring': func.__doc__,
            'module': func.__module__,
            'line_number': source_lines[1] if source_lines else None,
            'parameters': {},
            'return_annotation': signature.return_annotation if signature.return_annotation != inspect.Signature.empty else None
        }
        
        # Extract parameter information
        for param_name, param in signature.parameters.items():
            param_info = {
                'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None,
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'kind': param.kind.name
            }
            info['parameters'][param_name] = param_info
            
        return info
        
    except Exception as e:
        return {
            'name': getattr(func, '__name__', 'unknown'),
            'error': str(e)
        }


def validate_function_parameters(func: Callable, **kwargs) -> Dict[str, Any]:
    """
    Validate parameters against function signature.
    
    Args:
        func: Function to validate against
        **kwargs: Parameters to validate
        
    Returns:
        Dictionary with validation results
        
    Example:
        >>> def example_func(a: int, b: str = "default"):
        ...     pass
        >>> result = validate_function_parameters(example_func, a=5, b="test")
        >>> print(result['valid'])  # True
    """
    try:
        signature = inspect.signature(func)
        bound_args = signature.bind(**kwargs)
        bound_args.apply_defaults()
        
        return {
            'valid': True,
            'bound_arguments': dict(bound_args.arguments),
            'missing_required': [],
            'extra_arguments': []
        }
        
    except TypeError as e:
        # Parse error to extract missing/extra arguments
        error_msg = str(e)
        
        result = {
            'valid': False,
            'error': error_msg,
            'bound_arguments': {},
            'missing_required': [],
            'extra_arguments': []
        }
        
        # Try to extract specific parameter issues
        try:
            signature = inspect.signature(func)
            
            # Find missing required parameters
            for param_name, param in signature.parameters.items():
                if (param.default == inspect.Parameter.empty and 
                    param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, 
                                 inspect.Parameter.KEYWORD_ONLY) and
                    param_name not in kwargs):
                    result['missing_required'].append(param_name)
            
            # Find extra arguments
            valid_params = set(signature.parameters.keys())
            provided_params = set(kwargs.keys())
            result['extra_arguments'] = list(provided_params - valid_params)
            
        except Exception:
            pass  # If we can't parse details, just return the basic error
            
        return result