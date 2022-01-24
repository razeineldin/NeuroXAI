# Gudied BackPropagartion
from ..utils.process import get_last_layer

import numpy as np
import tensorflow as tf
from skimage.transform import resize # for N-dimensional images

@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad

def get_guided_backprop(model, io_imgs, class_id, LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="classification", DIMENSION="2d"):
    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}

    clone_model = tf.keras.models.clone_model(model)
    clone_model.set_weights(model.get_weights())

    if LAYER_NAME==None:
        LAYER_NAME = get_last_layer(clone_model).name
    # Build the GBP model
    gbp_layer = clone_model.get_layer(LAYER_NAME)
    guided_model = tf.keras.models.Model([clone_model.inputs], [gbp_layer.output])    
    layer_dict = [layer for layer in guided_model.layers[1:] if hasattr(layer,"activation")]
    for layer in layer_dict:
        if layer.activation == tf.keras.activations.relu:
            layer.activation = guided_relu
    
    io_imgs = tf.convert_to_tensor(io_imgs)
    with tf.GradientTape() as tape:
        tape.watch(io_imgs)
        
        if XAI_MODE == "classification":
            output = guided_model(io_imgs)
            output = output[:,class_id]
        elif XAI_MODE == "segmentation":
            output = guided_model(io_imgs)
            output = output[:,:,:,:,class_id]

        # Extract filters and gradients
        guided_backprop = tape.gradient(output, io_imgs)[0] # whole brain

    # Resize to the output layer's shape
    new_shape = io_imgs.shape[1:len(io_imgs.shape)]
    guided_backprop = resize(np.asarray(guided_backprop), new_shape) # convert to 3D instead of 4D

    return guided_backprop
