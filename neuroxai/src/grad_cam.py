# Grad-CAM
from ..utils.process import get_last_conv_layer

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv3D
from skimage.transform import resize # for N-dimensional images

def get_grad_cam(model, io_imgs, class_id, LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="classification", DIMENSION="2d", eps=1e-5):
    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}
    if LAYER_NAME==None:
        LAYER_NAME = get_last_conv_layer(model).name
    # Sanity check
    layer = model.get_layer(LAYER_NAME)
    assert (isinstance(layer, Conv2D) or isinstance(layer, Conv3D)), "Input layer must be convolutional layer for Grad-CAM"

    io_imgs = tf.convert_to_tensor(io_imgs)
    with tf.GradientTape() as tape:
        tape.watch(io_imgs)
        conv_outputs, predictions = model(io_imgs)
        if XAI_MODE == "classification":
            loss = predictions[:,class_id]
        elif XAI_MODE == "segmentation":
            loss = predictions[:,:,:,:,class_id]

        # Compute gradients with automatic differentiation
        output = conv_outputs[0]        
        grads = tape.gradient(loss, conv_outputs)[0]
        
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))
        if DIMENSION=="2d":
            weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        elif DIMENSION=="3d":
            weights = tf.reduce_mean(norm_grads, axis=(0, 1, 2))
    
    # Average gradients spatially
    # Build a ponderated map of filters according to gradients importance
    cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)

    # Apply ReLU
    grad_cam = np.maximum(cam, 0)
    # Resize heatmap to be the same size as the input
    if np.max(grad_cam) > 0:
        grad_cam = grad_cam / np.max(grad_cam)

    # Resize to the output layer's shape
    new_shape = io_imgs.shape[1:len(io_imgs.shape)]
    grad_cam = resize(grad_cam, new_shape)
    return grad_cam
