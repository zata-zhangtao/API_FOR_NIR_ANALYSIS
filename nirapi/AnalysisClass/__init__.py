"""
AnalysisClass subpackage for NIR API

Contains specialized analysis classes for different NIR analysis tasks.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any, List

__all__ = [
    "Create_rec_task",
    "Create_train_task",
    "CreateTrainReport",
    "DataAnalysisReport",
]

_SUBMODULES = {
    name: f".{name}"
    for name in __all__
}


def __getattr__(name: str) -> Any:
    if name not in _SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_SUBMODULES[name], __name__)
    globals()[name] = module
    return module


def __dir__() -> List[str]:
    return sorted(list(__all__) + [key for key in globals().keys() if not key.startswith("_")])


if TYPE_CHECKING:
    from . import CreateTrainReport, Create_rec_task, Create_train_task, DataAnalysisReport
