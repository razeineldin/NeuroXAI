"""
NeuroXAI framework
"""

__version__ = "0.1"

try:
    import cv2
except:
    raise ImportError(
        "NeuroXAI requires Opencv. "
        "Install Opencv via `pip install opencv-python`"
    ) from None
try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "NeuroXAI requires TensorFlow 2.0 or higher. "
        "Install TensorFlow via `pip install tensorflow`"
    ) from None


