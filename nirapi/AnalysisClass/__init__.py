"""
AnalysisClass subpackage for NIR API

Contains specialized analysis classes for different NIR analysis tasks.
"""

from . import Create_rec_task
from . import Create_train_task
from . import CreateTrainReport
from . import DataAnalysisReport

__all__ = [
    'Create_rec_task',
    'Create_train_task', 
    'CreateTrainReport',
    'DataAnalysisReport'
] 