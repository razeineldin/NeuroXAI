# Vanilla Gradient
from ..utils.process import get_last_layer

import numpy as np
import tensorflow as tf
from skimage.transform import resize # for N-dimensional images

def get_vanilla_grad(model, io_imgs, class_id, LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="classification", DIMENSION="2d"):
    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}
    
    io_imgs = tf.convert_to_tensor(io_imgs)
    with tf.GradientTape() as tape:
        tape.watch(io_imgs)
            
        last_layer = get_last_layer(model)
        _, predictions = model(io_imgs)
        if LAYER_NAME == None or LAYER_NAME == last_layer.name:
            if XAI_MODE == "classification":
                predictions = predictions[:,class_id]
            elif XAI_MODE == "segmentation":
                predictions = predictions[:,:,:,:,class_id]

        # Extract filters and gradients
        vanilla_grad = tape.gradient(predictions, io_imgs)[0]
        # Resize to the output layer's shape
        new_shape = io_imgs.shape[1:len(io_imgs.shape)]
        vanilla_grad = resize(np.asarray(vanilla_grad), new_shape) # convert to 3D instead of 4D

    return vanilla_grad
