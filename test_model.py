import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

IMG_SIZE = (240, 240)
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'cnn-parameters-improvement-23-0.91.h5')

def build_model(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((4, 4), name='max_pool0')(X)
    X = MaxPooling2D((4, 4), name='max_pool1')(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    model = Model(inputs = X_input, outputs = X, name='BrainDetectionModel')
    return model

print(f"Loading weights from {MODEL_PATH}...")
try:
    model = build_model((IMG_SIZE[0], IMG_SIZE[1], 3))
    # In newer Keras, if it is weights only, use load_weights
    model.load_weights(MODEL_PATH)
    print("Weights loaded successfully!")
    import numpy as np
    dummy_input = np.random.rand(1, 240, 240, 3)
    pred = model.predict(dummy_input)
    print(f"Prediction test success: {pred}")
except Exception as e:
    import traceback
    print(f"Error loading model: {e}")
    traceback.print_exc()
