"""
NIR API - A Near-Infrared Spectroscopy Analysis Package

This package provides tools for Near-Infrared spectroscopy analysis including:
- Data loading and preprocessing
- Machine learning models
- Visualization tools
- Analysis utilities
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any, List

__version__ = "1.0.0"
__author__ = "zata"
__description__ = "A Near-Infrared Spectroscopy Analysis API"

__all__ = [
    "utils",
    "load_data",
    "preprocessing",
    "draw",
    "analysis",
    "ML_model",
    "featsec",
    "model_class",
    "AnalysisClass",
]

_SUBMODULES = {name: f".{name}" for name in __all__}


def __getattr__(name: str) -> Any:
    if name not in _SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_SUBMODULES[name], __name__)
    globals()[name] = module
    return module


def __dir__() -> List[str]:
    std_attrs = [key for key in globals().keys() if not key.startswith("_")]
    return sorted(set(std_attrs + __all__))


if TYPE_CHECKING:
    from . import (
        AnalysisClass,
        ML_model,
        analysis,
        draw,
        featsec,
        load_data,
        model_class,
        preprocessing,
        utils,
    )
