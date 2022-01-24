# Utils methods
from .config import DIMENSION

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Conv3D

def deprocess_image(img):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    img = np.array(img).copy()
    img -= img.mean()
    img /= (img.std() + K.epsilon())
    img *= 0.25
    # clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)
    # convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def norm_image(img, NORM_TYP="norm"):
    if NORM_TYP == "standard_norm": # standarization, same dataset
        img_mean = img.mean()
        img_std = img.std()
        img_std = 1 if img.std()==0 else img.std()
        img = (img - img_mean) / img_std
    elif NORM_TYP == "norm": # different datasets
        img = (img - np.min(img))/(np.ptp(img)) # (np.max(img) - np.min(img))
    elif NORM_TYP == "norm_slow": # different datasets
        img_ptp = 1 if np.ptp(img)== 0 else np.ptp(img) 
        img = (img - np.min(img))/img_ptp
    return img

def get_last_layer(model):
    last_layer = model.layers[-1]
    return last_layer

def get_last_conv_layer(model, DIM=DIMENSION):
    if DIM == "2d":
        final_conv = list(filter(lambda x: isinstance(x, Conv2D), 
                                   model.layers))[-1]
    elif DIM == "3d":
        final_conv = list(filter(lambda x: isinstance(x, Conv3D), 
                                   model.layers))[-1]
    return final_conv
