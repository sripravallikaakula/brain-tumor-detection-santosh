#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Brain Tumor Detection - Web Application
Flask backend with CNN model inference for MRI scan analysis.
"""

import os
import io
import base64
import json
import time
import uuid
import sqlite3
from datetime import datetime

import numpy as np
import cv2
import imutils
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── TensorFlow / Keras ──────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model

# ── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'cnn-parameters-improvement-23-0.91.h5')
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
DB_PATH = os.path.join(BASE_DIR, 'brain_tumor.db')
IMG_SIZE = (240, 240)

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Flask App ───────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static')
CORS(app)
app.secret_key = 'brain-tumor-detection-secret-key'

# ── Database ────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_url TEXT NOT NULL,
            result TEXT,
            confidence TEXT,
            confidence_score REAL,
            status TEXT DEFAULT 'pending',
            findings TEXT,
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ── Build Model ─────────────────────────────────────────────────────────────
def build_model(input_shape):
    """
    Build the CNN model architecture as defined in the training notebook.
    """
    from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
    from tensorflow.keras.models import Model

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

# ── Load Model ──────────────────────────────────────────────────────────────
print(f"Loading model weights from {MODEL_PATH}...")
try:
    # Build model architecture
    model = build_model((IMG_SIZE[0], IMG_SIZE[1], 3))
    # Load weights
    model.load_weights(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None

# ── Image Processing ────────────────────────────────────────────────────────
def crop_brain_contour(image):
    """Crop the brain region from the MRI image."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if not cnts:
            return image
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
        return new_image
    except:
        return image

def predict_tumor(image_data):
    """Run prediction on image data (base64 or file bytes)."""
    if model is None:
        return {
            'result': 'Model not loaded',
            'confidence': 'N/A',
            'confidence_score': 0.0,
            'findings': ['The ML model could not be loaded. Please check the model file.']
        }

    # Decode image
    if isinstance(image_data, str):
        # base64
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
    else:
        img_bytes = image_data

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {
            'result': 'Invalid Image',
            'confidence': 'N/A',
            'confidence_score': 0.0,
            'findings': ['Could not decode the uploaded image.']
        }

    # Preprocess
    img_cropped = crop_brain_contour(img)
    img_resized = cv2.resize(img_cropped, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Predict
    prediction = model.predict(img_input, verbose=0)
    prob = float(prediction[0][0])

    if prob > 0.5:
        result = 'Brain Tumor Detected'
        confidence_score = prob
    else:
        result = 'No Brain Tumor Detected'
        confidence_score = 1 - prob

    # Determine confidence level
    if confidence_score >= 0.85:
        confidence = 'High'
    elif confidence_score >= 0.65:
        confidence = 'Medium'
    else:
        confidence = 'Low'

    # Generate findings
    findings = []
    if prob > 0.5:
        findings.append(f'The AI model detected a potential brain tumor with {confidence_score*100:.1f}% confidence.')
        findings.append('A mass or abnormal growth pattern was identified in the scan region.')
        if confidence_score >= 0.85:
            findings.append('High confidence detection — the structural anomaly is clearly visible in the scan.')
        elif confidence_score >= 0.65:
            findings.append('Medium confidence detection — further imaging may help confirm the finding.')
        else:
            findings.append('Low confidence detection — additional scans and clinical evaluation are recommended.')
        findings.append('This is an AI-assisted preliminary screening. Always consult a qualified medical professional for diagnosis.')
    else:
        findings.append(f'No significant brain tumor indicators were detected (confidence: {confidence_score*100:.1f}%).')
        findings.append('The scan appears to show normal brain structure within the analyzed region.')
        findings.append('Regular follow-up scans are recommended as part of routine health monitoring.')
        findings.append('This is an AI-assisted screening tool and does not replace professional medical evaluation.')

    return {
        'result': result,
        'confidence': confidence,
        'confidence_score': round(confidence_score, 4),
        'findings': findings
    }

# ── API Routes ──────────────────────────────────────────────────────────────
@app.route('/api/analyses', methods=['GET'])
def list_analyses():
    conn = get_db()
    rows = conn.execute('SELECT * FROM analyses ORDER BY id DESC').fetchall()
    conn.close()
    result = []
    for row in rows:
        item = dict(row)
        if item.get('findings'):
            try:
                item['findings'] = json.loads(item['findings'])
            except:
                item['findings'] = []
        else:
            item['findings'] = []
        result.append(item)
    return jsonify(result), 200

@app.route('/api/analyses/<int:analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    conn = get_db()
    row = conn.execute('SELECT * FROM analyses WHERE id = ?', (analysis_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({'message': 'Analysis not found'}), 404
    item = dict(row)
    if item.get('findings'):
        try:
            item['findings'] = json.loads(item['findings'])
        except:
            item['findings'] = []
    else:
        item['findings'] = []
    return jsonify(item), 200

@app.route('/api/analyses', methods=['POST'])
def create_analysis():
    data = request.get_json()
    if not data or 'imageBase64' not in data:
        return jsonify({'message': 'No image data provided', 'field': 'imageBase64'}), 400

    image_base64 = data['imageBase64']

    # Save image to disk
    try:
        if ',' in image_base64:
            img_data = image_base64.split(',')[1]
        else:
            img_data = image_base64

        img_bytes = base64.b64decode(img_data)
        filename = f"scan_{uuid.uuid4().hex[:12]}.png"
        filepath = os.path.join(UPLOAD_DIR, filename)

        # Decode and save as PNG
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imwrite(filepath, img)
        else:
            with open(filepath, 'wb') as f:
                f.write(img_bytes)
    except Exception as e:
        return jsonify({'message': f'Failed to process image: {str(e)}'}), 400

    image_url = f'/static/uploads/{filename}'
    created_at = datetime.utcnow().isoformat() + 'Z'

    # Run prediction
    prediction = predict_tumor(image_base64)

    # Save to database
    conn = get_db()
    cursor = conn.execute('''
        INSERT INTO analyses (image_url, result, confidence, confidence_score, status, findings, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        image_url,
        prediction['result'],
        prediction['confidence'],
        prediction['confidence_score'],
        'completed',
        json.dumps(prediction['findings']),
        created_at
    ))
    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return jsonify({
        'id': analysis_id,
        'imageUrl': image_url,
        'result': prediction['result'],
        'confidence': prediction['confidence'],
        'confidenceScore': prediction['confidence_score'],
        'status': 'completed',
        'findings': prediction['findings'],
        'createdAt': created_at
    }), 201

# ── Static File Routes ──────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/history')
def history():
    return send_from_directory('static', 'index.html')

@app.route('/analyses/<path:path>')
def analysis_detail(path):
    return send_from_directory('static', 'index.html')

@app.route('/favicon.png')
def favicon():
    return send_from_directory('static', 'favicon.png')

# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Starting Brain Tumor Detection Web Application...")
    print(f"Model: {MODEL_PATH}")
    print(f"Database: {DB_PATH}")
    print(f"Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=True)
