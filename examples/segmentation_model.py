import numpy as np
import os, sys
import nibabel as nib

from neuroxai.utils.process import norm_image

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import concatenate, Conv3D, UpSampling3D, Activation, BatchNormalization
from tensorflow.keras.layers import SpatialDropout3D, MaxPooling3D, Conv3DTranspose #old: DECONV3D
from tensorflow.keras.optimizers import Adam

from tensorflow_addons.layers import InstanceNormalization

def crop_image_brats(img, OUT_SHAPE=(192, 224, 160)):
    # manual cropping
    input_shape = np.array(img.shape)
    # center the cropped image
    offset = np.array((input_shape - OUT_SHAPE) / 2).astype(np.int)
    offset[offset < 0] = 0
    x, y, z = offset
    crop_img = img[x:x + OUT_SHAPE[0], y:y + OUT_SHAPE[1], z:z + OUT_SHAPE[2]]
    # pad the preprocessed image
    padded_img = np.zeros(OUT_SHAPE)
    x, y, z = np.array((OUT_SHAPE - np.array(crop_img.shape)) / 2).astype(np.int)
    padded_img[x:x + crop_img.shape[0], y:y + crop_img.shape[1], z:z + crop_img.shape[2]] = crop_img
    return padded_img

def postprocess_tumor(seg_data, OUT_SHAPE = (240, 240, 155), POST_ENHANC=False, THRES=200):
    # post-process the enhancing tumor region
    if POST_ENHANC:
        seg_enhancing = (seg_data == 4)
        if np.sum(seg_enhancing) < THRES:
            if np.sum(seg_enhancing) > 0:
                seg_data[seg_enhancing] = 1
                print("\tConverted {} voxels from label 4 to label 1!".format(np.sum(seg_enhancing)))

    input_shape = np.array(seg_data.shape)
    OUT_SHAPE = np.array(OUT_SHAPE)
    offset = np.array((OUT_SHAPE - input_shape)/2).astype(np.int)
    offset[offset<0] = 0
    x, y, z = offset

    # pad the preprocessed image
    padded_seg = np.zeros(OUT_SHAPE).astype(np.uint8)
    padded_seg[x:x+seg_data.shape[0],y:y+seg_data.shape[1],z:z+seg_data.shape[2]] = seg_data[:,:,2:padded_seg.shape[2]+2]

    return padded_seg #.astype(np.uint8)

def load_images(model, ID, PATH_DATA='./', DIM=(192, 224, 160), VALID_SET=True, POST_ENHANC=False):
    img1 = os.path.join(PATH_DATA, ID, ID+'_flair.nii.gz')
    img2 = os.path.join(PATH_DATA, ID, ID+'_t1.nii.gz')
    img3 = os.path.join(PATH_DATA, ID, ID+'_t1ce.nii.gz')
    img4 = os.path.join(PATH_DATA, ID, ID+'_t2.nii.gz')

    # combine the four imaging modalities (flair, t1, t1ce, t2)
    imgs_input = nib.concat_images([img1, img2, img3, img4]).get_fdata()

    imgs_preprocess = np.zeros((DIM[0],DIM[1],DIM[2],4)) # (5, 192, 224, 160)
    if VALID_SET:
        for i in range(imgs_preprocess.shape[-1]):
            imgs_preprocess[:, :, :, i] = crop_image_brats(imgs_input[:, :, :, i])
            imgs_preprocess[:, :, :, i] = norm_image(imgs_preprocess[:, :, :, i])

    return imgs_preprocess[np.newaxis, ...]

def create_convolution_block(input_layer, n_filters, BN=True, KERNEL=(3, 3, 3), ACTIV=None,
                             PAD='same', STR=(1, 1, 1), IN=False):
    layer = Conv3D(n_filters, KERNEL, padding=PAD, strides=STR)(input_layer)
    if BN:
        layer = BatchNormalization(axis=-1)(layer)
    elif IN:
        layer = InstanceNormalization(axis=-1)(layer)
    if ACTIV is None:
        return Activation('relu')(layer)
    else:
        return ACTIV()(layer)

def get_up_convolution(n_filters, pool_size, KERNEL=(3, 3, 3), STR=(2, 2, 2),
                       DECONV=False):
    if DECONV:
        return Conv3DTranspose(filters=n_filters, kernal=KERNEL,
                               strides=STR)
    else:
        return UpSampling3D(size=pool_size)

def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters)
    return convolution2

def create_context_module(input_layer, n_level_filters, DROPOUT=0.3):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=DROPOUT)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2

def create_up_sampling_module(input_layer, n_filters, SIZE=(2, 2, 2)):
    convolution1 = create_convolution_block(input_layer, n_filters, KERNEL=(2, 2, 2))
    up_sample = UpSampling3D(size=SIZE)(convolution1)
    return up_sample


def get_deepseg(INP_SHAPE=(192, 224, 160, 4), N_FILTERS=8, DEPTH=5, DROPOUT=0.5,
                      N_SEG=3, N_LABELS=4, OPT=Adam, INIT_LR=1e-4,
                      LOSS="mse", ACTIV="softmax"):
    inputs = Input(INP_SHAPE)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(DEPTH):
        n_level_filters = (2**level_number) * N_FILTERS
        level_filters.append(n_level_filters)

        in_conv = create_convolution_block(current_layer, n_level_filters)
        context_output_layer = create_convolution_block(in_conv, n_level_filters)

        level_output_layers.append(current_layer)
        current_layer = MaxPooling3D(pool_size=(2, 2, 2))(context_output_layer)

    current_layer = SpatialDropout3D(rate=DROPOUT)(current_layer)

    for level_number in reversed(range(DEPTH)):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=-1)
        
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output

    output_layer = Conv3D(N_LABELS, (1, 1, 1), name="output_layer")(current_layer)
    activ_block = Activation(ACTIV, name="output_layer_soft")(output_layer)
    model = Model(inputs=inputs, outputs=activ_block)
    
    model.compile(optimizer=Adam(lr=INIT_LR), loss=LOSS, metrics=["accuracy"])
    return model


