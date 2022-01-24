# Classification model imports
from imutils import paths
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import cv2

def load_images_2d(im, IM_SIZE=224):
    if type(im)!=np.ndarray:
        im = cv2.imread(im) # read the image from the path

    new_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    new_im = cv2.resize(new_im, (IM_SIZE, IM_SIZE))
    new_im = np.array(new_im) / 255.0
    new_im = new_im[np.newaxis, ...]
    return new_im

def get_classification_model(IM_SIZE=224, N_CLASSES=2, INIT_LR=1e-3, MODEL_WEIGHTS="weights/ResNet50_model.hdf5", NETWORK=ResNet50V2):
    # load the base model (ResNet50V2 network) without the head FC layer
    baseModel = NETWORK(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(IM_SIZE, IM_SIZE, 3)))

    # customize the top layers (transfer learning)
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(3, 3))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(N_CLASSES, activation="softmax")(headModel)

    # build the classification model
    model = Model(inputs=baseModel.input, outputs=headModel)

    for layer in baseModel.layers:
        layer.trainable = False

    # compile the classification model
    opt = Adam(learning_rate= INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    if MODEL_WEIGHTS != None:
        model.load_weights(MODEL_WEIGHTS)
    
    return model
