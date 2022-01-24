# NeuroXAI parameters
XAI_MODES = ["segmentation","classification"]
XAIs = ["VANILLA","GBP","IG","GIG","GCAM","GGCAM","SMOOTH"]
MODALITIES = ["FLAIR","T1","T1CE","T2"]
DIMENSIONS = ["2d","3d"]
TUMOR_LABEL = "all" # for GCAM visualization

# segmentation mode
#DIMENSION = "3d"
#MODALITY = "FLAIR"
#XAI_MODE = "segmentation"
#CLASS_IDs = [1,2,3]

# classification mode
DIMENSION = "2d"
MODALITY = "FLAIR"
XAI_MODE = "classification"
CLASS_IDs = [0,1]

# GPU handling (TF 2.X)
import os
import tensorflow as tf

assert float(tf.__version__[:3]) >= 2.0, "NeuroXAI requires Tensorflow version 2.0 or higher"

gpu_ids = '0' # '0,1' # for multi-gpu environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs\n")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
