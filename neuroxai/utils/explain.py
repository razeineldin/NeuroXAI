from .config import DIMENSIONS, XAI_MODES, MODALITIES, XAIs
from .process import get_last_layer, get_last_conv_layer
from .visualize import show_image, show_gray_image, show_heatmap, visualize_tensor
from .visualize import overlay_gradcam, overlay_grad, overlay_pred
from ..src.guided_backprop import get_guided_backprop
from ..src.grad_cam import get_grad_cam
from ..src.guided_grad_cam import get_guided_grad_cam
from ..src.guided_ig import get_guided_integrated_grads, compute_grads
from ..src.integrated_grads import get_integrated_grads
from ..src.smoothgrad import get_smoothgrad
from ..src.vanilla_grad import get_vanilla_grad

import matplotlib.pyplot as plt
import numpy as np
import os

# Visualization and save of all NeuroXAI methods
def get_neuroxai(ID, model, io_imgs, CLASS_ID=0, SLICE_ID=77, LAYER_NAME=None, 
                 MODALITY="FLAIR", XAI_MODE="classification", 
                 DIMENSION="2d", CLASS_IDs=[0,1], TUMOR_LABEL="all", SAVE_RESULTS=False, SAVE_PATH=None):

    # sanity checks
    assert DIMENSION in DIMENSIONS, "Input dimension must be in {}".format([d for d in DIMENSIONS])
    assert XAI_MODE in XAI_MODES, "XAI mode must be in {}".format([m for m in XAI_MODES])
    assert MODALITY in MODALITIES, "MRI modality must be in {}".format([m for m in MODALITIES])

    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}

    if LAYER_NAME==None:
        LAYER_NAME = get_last_layer(model).name
        CONV_LAYER_NAME = get_last_conv_layer(model, DIMENSION).name

    else:
        # A solution for classification visualization of all XAI
        layer = model.get_layer(LAYER_NAME)
        last_layer = get_last_layer(model)
        if layer.name == last_layer.name:
            if (not isinstance(layer, Conv2D)) or (not isinstance(layer, Conv3D)):
                CONV_LAYER_NAME = get_last_conv_layer(model, DIMENSION).name
            else:
                CONV_LAYER_NAME = LAYER_NAME
        else:
            CONV_LAYER_NAME = LAYER_NAME
    
    print("Visual exaplanations for...\n\t ID: {}, layer: {}".format(ID, LAYER_NAME))
    # Build the model
    if XAI_MODE=="classification": # 2d and 3d
        im_orig = io_imgs[0]
        SmoothXAI="IG"
    elif XAI_MODE=="segmentation": # 3d only
        im_orig = io_imgs[0,:,:,SLICE_ID,modality_dict[MODALITY]] # (1, 192, 224, 160, 4)
        SmoothXAI="GIG"

    # Get XAI heat maps (1 samples for SmoothGrad)
    #vanilla_grads = get_vanilla_grad(model, io_imgs, CLASS_ID, LAYER_NAME=LAYER_NAME, MODALITY=MODALITY, XAI_MODE=XAI_MODE)
    vanilla_grads = compute_grads(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE)[0]
    
    gbp_grads = get_guided_backprop(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE)
    ig_grads = get_integrated_grads(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE)
    gig_grads = get_guided_integrated_grads(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE)
    
    if XAI_MODE=="segmentation" and TUMOR_LABEL=="all":
        gcam_grads = get_grad_cam(model, io_imgs, CLASS_IDs[0], CONV_LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)
        for c_id in CLASS_IDs[1:]:
            gcam_grads += get_grad_cam(model, io_imgs, c_id, CONV_LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)
    else:
        gcam_grads = get_grad_cam(model, io_imgs, CLASS_ID, CONV_LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)
    
    ggcam_grads = get_guided_grad_cam(model, io_imgs, CLASS_ID, CONV_LAYER_NAME, MODALITY, XAI_MODE, DIMENSION, CLASS_IDs, TUMOR_LABEL)
    smooth_grads = get_smoothgrad(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE, XAI=SmoothXAI)
    
    # Get 2D images for visualizations
    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    guided_backprop = visualize_tensor(gbp_grads)
    vanilla_gradient = visualize_tensor(vanilla_grads)
    integrated_gradient = visualize_tensor(ig_grads)
    guided_integrated_gradient = visualize_tensor(gig_grads)
    gradcam = visualize_tensor(gcam_grads)
    guided_gradcam = visualize_tensor(ggcam_grads)
    smooth_grad = visualize_tensor(smooth_grads)
    
    if DIMENSION=="3d":
        guided_backprop = guided_backprop[:,:,SLICE_ID]
        vanilla_gradient = vanilla_gradient[:,:,SLICE_ID]
        integrated_gradient = integrated_gradient[:,:,SLICE_ID]
        guided_integrated_gradient = guided_integrated_gradient[:,:,SLICE_ID]
        gradcam = gradcam[:,:,SLICE_ID]
        guided_gradcam = guided_gradcam[:,:,SLICE_ID]
        smooth_grad = smooth_grad[:,:,SLICE_ID]

    # Get overlay images (Over FLAIR MRI)
    gradcam_overlay = overlay_gradcam(im_orig, gradcam, DIMENSION)
    guided_gradcam_overlay = overlay_grad(im_orig, guided_gradcam, DIMENSION)

    # Set up matplot lib figures.
    ROWS = 1
    COLS = 11 if XAI_MODE=="segmentation" else 9
    UPSCALE_FACTOR = 25
    plt.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

    # Render the saliency masks.
    show_gray_image(im_orig, TITLE=MODALITY, AX=plt.subplot(ROWS, COLS, 1))
    show_gray_image(vanilla_gradient, TITLE='Vanilla', AX=plt.subplot(ROWS, COLS, 2))
    show_gray_image(guided_backprop, TITLE='Backprop', AX=plt.subplot(ROWS, COLS, 3))
    show_gray_image(integrated_gradient, TITLE='IG', AX=plt.subplot(ROWS, COLS, 4))
    show_gray_image(guided_integrated_gradient, TITLE='Guided IG', AX=plt.subplot(ROWS, COLS, 5))
    show_gray_image(smooth_grad, TITLE='SmoothGrad', AX=plt.subplot(ROWS, COLS, 6))
    show_heatmap(gradcam, TITLE='Grad-CAM', AX=plt.subplot(ROWS, COLS, 7), CMAP="jet")
    show_heatmap(gradcam_overlay, TITLE='Overlay Grad-CAM', AX=plt.subplot(ROWS, COLS, 8))
    show_heatmap(guided_gradcam, TITLE='Guided_Grad-CAM', AX=plt.subplot(ROWS, COLS, 9), CMAP="jet")
    
    if XAI_MODE=="segmentation":
        # Predict the tumor segmentation
        _, prediction_3ch = np.squeeze(model(io_imgs)) # model prediction 
        prediction = np.argmax(prediction_3ch, axis=-1)
        prediction = (prediction[:,:,SLICE_ID] == CLASS_ID).astype(np.uint8)
        if CLASS_ID!= 0:
            prediction[prediction>0] = CLASS_ID

        pred_overlay = overlay_pred(io_imgs[0,:,:,SLICE_ID,modality_dict[MODALITY]], prediction)
        show_image(prediction, TITLE='Prediction', AX=plt.subplot(ROWS, COLS, 10))
        show_image(pred_overlay, TITLE='Prediction Overlay', AX=plt.subplot(ROWS, COLS, 11))
        
    # Save the results
    if SAVE_RESULTS:
        if not os.path.exists(os.path.join(SAVE_PATH, ID)):
            os.makedirs(os.path.join(SAVE_PATH, ID))
        # Save 2D heatmaps
        if XAI_MODE=="classification":
            LAYER_NAME=""
        if TUMOR_LABEL=="all":
            CLASS_ID="all"
        plt.imsave("{}/{}/{}_{}_{}_vanilla.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), vanilla_gradient, CMAP=plt.cm.gray)
        plt.imsave("{}/{}/{}_{}_{}_ig.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), integrated_gradient, CMAP=plt.cm.gray)
        plt.imsave("{}/{}/{}_{}_{}_gig.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), guided_integrated_gradient, CMAP=plt.cm.gray)
        plt.imsave("{}/{}/{}_{}_{}_gbp.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), guided_backprop, CMAP=plt.cm.gray)
        plt.imsave("{}/{}/{}_{}_{}_gcam.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), gradcam, CMAP="jet")
        plt.imsave("{}/{}/{}_{}_{}_ggcam.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), guided_gradcam, CMAP="jet")
        plt.imsave("{}/{}/{}_{}_{}_gcam_overlay.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), gradcam_overlay)
        plt.imsave("{}/{}/{}_{}_{}_MRI.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), im_orig, CMAP=plt.cm.gray)
        plt.imsave("{}/{}/{}_{}_{}_smooth.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), smooth_grad, CMAP=plt.cm.gray)
        
        if XAI_MODE=="segmentation":
            plt.imsave("{}/{}/{}_{}_{}_pred.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), prediction)
            #plt.imsave("{}/{}/{}_{}_{}_pred_overlay.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), pred_overlay)
            plt.imsave("{}/{}/{}_{}_{}_truth.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), pred_data_m[:,:,SLICE_ID])


# Visualization and save of a single NeuroXAI method
def visualize_neuroxai_cnn(ID, model, io_imgs, grads, CLASS_ID=0, SLICE_ID=77,
                           LAYER_NAME=None, MODALITY="FLAIR", 
                           XAI_MODE="classification", XAI="GCAM",
                           DIMENSION="2d", TUMOR_LABEL="all",
                           SAVE_RESULTS=False, SAVE_PATH=None):

    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}
        
    if XAI_MODE=="classification": # 2d and 3d
        im_orig = io_imgs[0]
        SmoothXAI="IG"
        #smooth_grads = get_smoothgrad(model, io_imgs, CLASS_ID, XAI_MODE=XAI_MODE, XAI="IG")
    elif XAI_MODE=="segmentation": # 3d only
        im_orig = io_imgs[0,:,:,SLICE_ID,modality_dict[MODALITY]] # (1, 192, 224, 160, 4)
        SmoothXAI="GIG"
        
    # Get 2D images for visualizations
    heatmap = visualize_tensor(grads)    
    if DIMENSION=="3d":
        heatmap = heatmap[:,:,SLICE_ID]

    # Set up matplot lib figures.
    ROWS = 1
    COLS = 5 if XAI_MODE=="segmentation" else 3
    UPSCALE_FACTOR = 25 if XAI_MODE=="segmentation" else 10
    plt.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))
    
    # Render the saliency masks.
    show_gray_image(im_orig, TITLE=MODALITY, AX=plt.subplot(ROWS, COLS, 1))
    # Get overlay images (Over MRI)
    if XAI=="GCAM" or XAI=="GGCAM":
        overlay = overlay_gradcam(im_orig, heatmap, DIMENSION)
        if LAYER_NAME==None:
            LAYER_NAME = get_last_conv_layer(model, DIMENSION).name
        show_heatmap(heatmap, TITLE=XAI+'-'+LAYER_NAME+'-'+str(CLASS_ID), AX=plt.subplot(ROWS, COLS, 2), CMAP="jet")
    else:
        overlay = overlay_grad(im_orig, heatmap, DIMENSION)
        if LAYER_NAME==None:
            LAYER_NAME = get_last_layer(model).name
        show_gray_image(heatmap, TITLE=XAI+'-'+LAYER_NAME+'-'+str(CLASS_ID), AX=plt.subplot(ROWS, COLS, 2)) # CMAP=plt.cm.gray
    show_image(overlay, TITLE=XAI+' Overlay', AX=plt.subplot(ROWS, COLS, 3)) 

    if XAI_MODE=="segmentation":
        # Predict the tumor segmentation
        _, prediction_3ch = np.squeeze(model(io_imgs)) # model prediction 
        prediction = np.argmax(prediction_3ch, axis=-1)
        prediction = (prediction[:,:,SLICE_ID] == CLASS_ID).astype(np.uint8)
        if CLASS_ID!= 0:
            prediction[prediction>0] = CLASS_ID

        pred_overlay = overlay_pred(io_imgs[0,:,:,SLICE_ID,modality_dict[MODALITY]], prediction)
        show_image(prediction, TITLE='Prediction', AX=plt.subplot(ROWS, COLS, 4))
        show_image(pred_overlay, TITLE='Prediction Overlay', AX=plt.subplot(ROWS, COLS, 5))
    
    # Save the results
    if SAVE_RESULTS:
        if not os.path.exists(os.path.join(SAVE_PATH, ID)):
            os.makedirs(os.path.join(SAVE_PATH, ID))
    
        if XAI_MODE=="classification":
            LAYER_NAME=""
        if TUMOR_LABEL=="all":
            CLASS_ID="all"
        
        # Save 2D heatmaps
        if XAI=="GCAM" or XAI=="GGCAM":
            plt.imsave("{}/{}/{}_{}_{}_{}_{}.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID, MODALITY, XAI), heatmap, CMAP="jet")
        else:
            plt.imsave("{}/{}/{}_{}_{}_{}_{}.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID, MODALITY, XAI), heatmap, CMAP=plt.cm.gray)
            
        plt.imsave("{}/{}/{}_{}_{}_{}_{}_overlay.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID, MODALITY, XAI), overlay)
        plt.imsave("{}/{}/{}_{}_{}_{}_MRI.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID, MODALITY), im_orig, CMAP=plt.cm.gray)
        
        if XAI_MODE=="segmentation":
            plt.imsave("{}/{}/{}_{}_{}_{}_{}_pred.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID, MODALITY, XAI), prediction)
            plt.imsave("{}/{}/{}_{}_{}_{}_{}_pred_overlay.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID, MODALITY, XAI), pred_overlay)
        
def get_neuroxai_cnn(ID, model, io_imgs, CLASS_ID=0, SLICE_ID=77, LAYER_NAME=None,
                     MODALITY="FLAIR", XAI_MODE="classification", XAI="GCAM", 
                     DIMENSION="2d", CLASS_IDs=[0,1], TUMOR_LABEL="all", 
                     SAVE_RESULTS=False, SAVE_PATH=None):

    # sanity checks
    assert DIMENSION in DIMENSIONS, "Input dimension must be in {}".format([d for d in DIMENSIONS])
    assert XAI_MODE in XAI_MODES, "XAI mode must be in {}".format([m for m in XAI_MODES])
    assert MODALITY in MODALITIES, "MRI modality must be in {}".format([m for m in MODALITIES])
    assert XAI in XAIs, "XAI method must be in {}".format([m for m in XAIs])

    if LAYER_NAME==None:
        LAYER_NAME = get_last_layer(model).name
        CONV_LAYER_NAME = get_last_conv_layer(model, DIMENSION).name

    else:
        layer = model.get_layer(LAYER_NAME)
        last_layer = get_last_layer(model)
        if layer.name == last_layer.name:
            if (not isinstance(layer, Conv2D)) or (not isinstance(layer, Conv3D)):
                CONV_LAYER_NAME = get_last_conv_layer(model, DIMENSION).name
            else:
                CONV_LAYER_NAME = LAYER_NAME
        else:
            CONV_LAYER_NAME = LAYER_NAME


    print("Visual exaplanations for...\n\t ID: {}, layer: {}".format(ID, LAYER_NAME))
    # Get the gradients    
    if XAI=="VANILLA":
        grads = get_vanilla_grad(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE)
    elif XAI=="GBP":
        grads = get_guided_backprop(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE)
    elif XAI=="IG":
        grads = get_integrated_grads(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE)
    elif XAI=="GIG":
        grads = get_guided_integrated_grads(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE)
    elif XAI=="GCAM":
        if XAI_MODE=="segmentation" and TUMOR_LABEL=="all":
            LAYER_NAME = CONV_LAYER_NAME
            grads = get_grad_cam(model, io_imgs, CLASS_IDs[0], LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)
            for c_id in CLASS_IDs[1:]:
                grads += get_grad_cam(model, io_imgs, c_id, LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)
        else:
            grads = get_grad_cam(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)


    elif XAI=="GGCAM":
        LAYER_NAME = CONV_LAYER_NAME
        grads = get_guided_grad_cam(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE, DIMENSION, CLASS_IDs, TUMOR_LABEL)
    elif XAI=="SMOOTH":
        if XAI_MODE=="classification": # 2d and 3d
            SmoothXAI="IG"
        elif XAI_MODE=="segmentation": # 3d only
            SmoothXAI="GIG"
        grads = get_smoothgrad(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE, SmoothXAI, DIMENSION)

    # Visualize the saliency map
    visualize_neuroxai_cnn(ID, model, io_imgs, grads, CLASS_ID, SLICE_ID, LAYER_NAME, MODALITY, 
                           XAI_MODE, XAI, DIMENSION, SAVE_RESULTS, SAVE_PATH)

        
