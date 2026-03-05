#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Brain Tumor Detection - Standalone Execution Script
Extracted from the Brain Tumor Detection.ipynb notebook,
with the deprecated val_acc metric name fixed to val_accuracy
for compatibility with newer TensorFlow/Keras versions.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import cv2
import imutils
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time
from os import listdir
import os

# ── Helpers ─────────────────────────────────────────────────────────────────

def crop_brain_contour(image, plot=False):
    """Crop the brain region from the MRI image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeft  = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop   = tuple(c[c[:, :, 1].argmin()][0])
    extBot   = tuple(c[c[:, :, 1].argmax()][0])
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    if plot:
        plt.figure()
        plt.subplot(1, 2, 1); plt.imshow(image);     plt.title('Original Image')
        plt.subplot(1, 2, 2); plt.imshow(new_image); plt.title('Cropped Image')
        plt.savefig('crop_demo.png'); plt.close()
    return new_image


def load_data(dir_list, image_size):
    """Load, crop, resize and normalise images from the given directories."""
    X, y = [], []
    image_width, image_height = image_size
    for directory in dir_list:
        for filename in listdir(directory):
            image = cv2.imread(os.path.join(directory, filename))
            if image is None:
                continue
            try:
                image = crop_brain_contour(image, plot=False)
                image = cv2.resize(image, dsize=(image_width, image_height),
                                   interpolation=cv2.INTER_CUBIC)
                image = image / 255.
                X.append(image)
                y.append([1] if directory[-3:] == 'yes' else [0])
            except Exception as e:
                print(f"  Skipping {filename}: {e}")
    X = np.array(X)
    y = np.array(y)
    X, y = shuffle(X, y)
    print(f'Number of examples : {len(X)}')
    print(f'X shape            : {X.shape}')
    print(f'y shape            : {y.shape}')
    return X, y


def hms_string(sec_elapsed):
    h = int(sec_elapsed / 3600)
    m = int((sec_elapsed % 3600) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


def build_model(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((4, 4), name='max_pool0')(X)
    X = MaxPooling2D((4, 4), name='max_pool1')(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    model = Model(inputs=X_input, outputs=X, name='BrainDetectionModel')
    return model


def compute_f1_score(y_true, prob):
    """Compute F1-score for binary classification."""
    y_pred = np.where(prob > 0.5, 1, 0)
    score = f1_score(y_true, y_pred)
    return score

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load data ──────────────────────────────────────────────────────
    augmented_path = 'augmented data/'
    augmented_yes  = augmented_path + 'yes'
    augmented_no   = augmented_path + 'no'

    IMG_WIDTH, IMG_HEIGHT = 240, 240
    IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

    print("Loading data …")
    X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))

    # ── 2. Split ───────────────────────────────────────────────────────────
    X_train, X_test_val, y_train, y_test_val = train_test_split(
        X, y, test_size=0.3)
    X_test,  X_val,      y_test,  y_val      = train_test_split(
        X_test_val, y_test_val, test_size=0.5)

    print(f"Train set size: {X_train.shape[0]}")
    print(f"Val   set size: {X_val.shape[0]}")
    print(f"Test  set size: {X_test.shape[0]}")

    # ── 3. Build & compile model ───────────────────────────────────────────
    model = build_model(IMG_SHAPE)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    # ── 4. Callbacks ───────────────────────────────────────────────────────
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    log_file_name = f'brain_tumor_detection_cnn_{int(time.time())}'
    tensorboard   = TensorBoard(log_dir=f'logs/{log_file_name}')

    # FIX: use val_accuracy (not val_acc) — compatible with TF 2.x
    filepath   = "cnn-parameters-improvement-{epoch:02d}-{val_accuracy:.2f}"
    checkpoint = ModelCheckpoint(
        f"models/{filepath}.model",
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    # ── 5. Train ───────────────────────────────────────────────────────────
    print("\nTraining model …")
    start_time = time.time()
    history = model.fit(
        x=X_train, y=y_train,
        batch_size=32,
        epochs=10,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard, checkpoint]
    )
    end_time = time.time()
    print(f"Elapsed time: {hms_string(end_time - start_time)}")

    # ── 6. Evaluate ────────────────────────────────────────────────────────
    # Training set
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    print(f"\nTraining Accuracy : {train_acc:.4f}")

    # Validation set
    _, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # F1 scores
    y_test_prob  = model.predict(X_test)
    train_f1 = compute_f1_score(y_train, model.predict(X_train))
    val_f1   = compute_f1_score(y_val,   model.predict(X_val))
    test_f1  = compute_f1_score(y_test,  y_test_prob)
    print(f"\nTrain F1  : {train_f1:.4f}")
    print(f"Val   F1  : {val_f1:.4f}")
    print(f"Test  F1  : {test_f1:.4f}")

    # ── 7. Plot training history ────────────────────────────────────────────
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'],     label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val   Acc')
    plt.title('Accuracy'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'],     label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val   Loss')
    plt.title('Loss'); plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("\nTraining history plot saved to training_history.png")
    print("Done!")


if __name__ == '__main__':
    main()
