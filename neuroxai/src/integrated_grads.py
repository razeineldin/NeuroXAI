# Integrated Gradient
from ..utils.process import get_last_layer

import numpy as np
import tensorflow as tf

def compute_grads(model, io_imgs, class_id, LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="classification"):
    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}
    
    io_imgs = tf.convert_to_tensor(io_imgs)
    with tf.GradientTape() as tape:
        tape.watch(io_imgs)

        last_layer = get_last_layer(model)
        if LAYER_NAME == None or LAYER_NAME == last_layer.name:
            _, predictions = model(io_imgs)
            if XAI_MODE == "classification":
                predictions = predictions[:,class_id]
            elif XAI_MODE == "segmentation":
                predictions = predictions[:,:,:,:,class_id]
                    
            gradients = np.array(tape.gradient(predictions, io_imgs))
        else:
            conv_outputs, predictions = model(io_imgs)
            gradients = np.array(tape.gradient(conv_outputs, io_imgs))
            
    return gradients

def get_integrated_grads(model, io_imgs, class_id, LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="classification", M_STEPS=25, BS=1):
    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}
    
    baseline = np.zeros_like(io_imgs)
    diff = io_imgs - baseline
    total_gradients = np.zeros_like(io_imgs, dtype=np.float32)

    m_step_batched = []
    for alpha in np.linspace(0, 1, M_STEPS):
        m_step = baseline + alpha * diff
        m_step_batched.append(m_step)
        if len(m_step_batched) == BS or alpha == 1:
            m_step_batched = np.asarray(m_step_batched)
            gradients = compute_grads(model, io_imgs, class_id, LAYER_NAME, MODALITY, XAI_MODE)

            total_gradients += gradients.sum(axis=0)
            m_step_batched = []

    integrated_grads = total_gradients * diff / M_STEPS

    return integrated_grads[0]
