# SmoothGrad
from .guided_backprop import get_guided_backprop
from .grad_cam import get_grad_cam
from .guided_grad_cam import get_guided_grad_cam
from .guided_ig import get_guided_integrated_grads, compute_grads
from .integrated_grads import get_integrated_grads
from .vanilla_grad import get_vanilla_grad
from ..utils.process import get_last_layer

import numpy as np

def get_smoothgrad(model, io_imgs, class_id, LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="classification",
                   XAI="GBP", DIMENSION="2d", STDEV_SPREAD=.15, N_SAMPLES=5, MAGNITUDE=True):
                   #XAI="GBP", DIMENSION="2d", STDEV_SPREAD=.15, N_SAMPLES=25, MAGNITUDE=True):
    
    new_shape = io_imgs.shape[1:len(io_imgs.shape)]

    if XAI_MODE == "segmentation" and XAI=="GBP":
        new_shape = io_imgs.shape[1:len(io_imgs.shape)] +(3,)
    
    total_gradients = np.zeros(new_shape, dtype=np.float32)
    #print("Shape of total_gradients:", total_gradients.shape)
    stdev = STDEV_SPREAD * (np.max(io_imgs) - np.min(io_imgs))
    for _ in range(N_SAMPLES):
        noise = np.random.normal(0, stdev, io_imgs.shape)
        x_plus_noise = io_imgs + noise
        if XAI=="VANILLA":
            grads = get_vanilla_grad(model, x_plus_noise, class_id, LAYER_NAME, MODALITY, XAI_MODE)
        elif XAI=="GBP":
            grads = get_guided_backprop(model, x_plus_noise, class_id, LAYER_NAME, MODALITY, XAI_MODE)
        elif XAI=="IG":
            grads = get_integrated_grads(model, x_plus_noise, class_id, LAYER_NAME, MODALITY, XAI_MODE)
        elif XAI=="GIG":
            grads = get_guided_integrated_grads(model, x_plus_noise, class_id, LAYER_NAME, MODALITY, XAI_MODE)
        elif XAI=="GCAM":
            grads = get_grad_cam(model, x_plus_noise, class_id, LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)  
        elif XAI=="GGCAM":
            grads = get_guided_grad_cam(model, x_plus_noise, class_id, LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)  
        
        if MAGNITUDE:
            total_gradients += (grads * grads)
        else:
            total_gradients += grads

    return total_gradients / N_SAMPLES
