# Guided Grad-CAM
from .grad_cam import get_grad_cam
from .guided_backprop import get_guided_backprop
from ..utils.process import get_last_conv_layer

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv3D
from skimage.transform import resize # for N-dimensional images

def get_guided_grad_cam(model, io_imgs, class_id, LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="classification", 
                        DIMENSION="2d", CLASS_IDs=[0,1], TUMOR_LABEL="all"):
    if LAYER_NAME==None:
        LAYER_NAME = get_last_conv_layer(model).name
    # Sanity check
    layer = model.get_layer(LAYER_NAME)
    assert (isinstance(layer, Conv2D) or isinstance(layer, Conv3D)), "Input layer must be convolutional layer for Grad-CAM"


    guided_backprop = get_guided_backprop(model, io_imgs, class_id, LAYER_NAME=LAYER_NAME, MODALITY=MODALITY, XAI_MODE=XAI_MODE)
    if XAI_MODE=="segmentation" and TUMOR_LABEL=="all":
        grad_cam = np.zeros_like(guided_backprop)
        for c_id in CLASS_IDs:
            grad_cam += get_grad_cam(model, io_imgs, c_id, LAYER_NAME=LAYER_NAME, MODALITY=MODALITY, XAI_MODE=XAI_MODE, DIMENSION=DIMENSION)
    else:
        grad_cam = get_grad_cam(model, io_imgs, class_id, LAYER_NAME=LAYER_NAME, MODALITY=MODALITY, XAI_MODE=XAI_MODE, DIMENSION=DIMENSION)
    
    return grad_cam*guided_backprop
