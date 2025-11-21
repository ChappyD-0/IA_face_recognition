"""
IA Face Recognition API
API REST para reconocimiento facial con TensorFlow/Keras
"""

__version__ = "1.0.0"
__author__ = "ChappyD-0"

from .model_handler import ModelHandler
from .utils import allowed_file, get_file_extension

__all__ = [
    "ModelHandler",
    "allowed_file", 
    "get_file_extension",
]
