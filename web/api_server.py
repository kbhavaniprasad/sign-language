"""
Simplified Flask API for Sign Language Recognition Web Interface
"""
import os
import cv2
import numpy as np
import base64
import json
import webbrowser
import threading
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
model = None
class_names = []
prediction_buffer = []  # Buffer for temporal smoothing
BUFFER_SIZE = 5  # Number of predictions to average

def preprocess_frame(frame):
    """Enhanced preprocessing to match training conditions"""
    # Resize to model input size
    img_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Normalize to [0, 1]
    img_normalized = enhanced_rgb.astype('float32') / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def smooth_predictions(current_predictions):
    """Temporal smoothing of predictions using a buffer"""
    global prediction_buffer
    
    # Add current predictions to buffer
    prediction_buffer.append(current_predictions)
    
    # Keep buffer size limited
    if len(prediction_buffer) > BUFFER_SIZE:
        prediction_buffer.pop(0)
    
    # Average predictions over buffer
    if len(prediction_buffer) > 0:
        smoothed = np.mean(prediction_buffer, axis=0)
        return smoothed
    return current_predictions

def load_model():
    """Load the trained model"""
    global model, class_names
    
    model_path = 'models/sign_language_model.h5'
    model_info_path = 'models/model_info.json'
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Load class names
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                class_names = model_info['class_names']
                print(f"Loaded {len(class_names)} classes: {class_names}")
        else:
            # Fallback to default class names
            class_names = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]
            print(f"Using default class names: {class_names}")
    else:
        print(f"ERROR: Model not found at {model_path}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Sign Language Recognition API is running',
        'model_loaded': model is not None
    })

@app.route('/predict-frame', methods=['POST'])
def predict_frame():
    """Predict gesture from video frame - optimized for real-time use"""
    try:
        # Get image from request
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Process frame with model
        if model is not None:
            # Enhanced preprocessing
            img_batch = preprocess_frame(frame)
            
            # Get predictions
            predictions = model.predict(img_batch, verbose=0)[0]
            
            # Apply temporal smoothing
            smoothed_predictions = smooth_predictions(predictions)
            
            # Get top 3 predictions from smoothed results
            top_3_indices = np.argsort(smoothed_predictions)[-3:][::-1]
            top_3_predictions = [
                {
                    'class': class_names[idx],
                    'confidence': float(smoothed_predictions[idx])
                }
                for idx in top_3_indices
            ]
            
            # Get top prediction
            top_class_idx = top_3_indices[0]
            predicted_class = class_names[top_class_idx]
            confidence = float(smoothed_predictions[top_class_idx])
            
            # Only return prediction if confidence is above threshold
            min_confidence = 0.3  # Minimum confidence threshold
            if confidence < min_confidence:
                predicted_class = "-"
                confidence = 0.0
            
            return jsonify({
                'success': True,
                'prediction': predicted_class,
                'confidence': confidence,
                'top_3': top_3_predictions
            })
        else:
            return jsonify({'error': 'Model not loaded'}), 500
            
    except Exception as e:
        print(f"Frame prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def open_browser():
    """Open the web interface in the default browser after a short delay"""
    time.sleep(2)  # Wait for server to start
    interface_path = os.path.abspath('web/index.html')
    interface_url = f'file:///{interface_path.replace(os.sep, "/")}'
    print(f"\n🌐 Opening web interface in browser...")
    print(f"   {interface_url}\n")
    webbrowser.open(interface_url)

if __name__ == '__main__':
    # Load model
    load_model()
    
    # Run app
    host = '0.0.0.0'
    port = 5000
    
    print(f"\n{'='*60}")
    print(f"Starting Sign Language Recognition API")
    print(f"Server: http://{host}:{port}")
    print(f"Model: {'Loaded' if model else 'NOT LOADED'}")
    print(f"Classes: {len(class_names)}")
    print(f"{'='*60}\n")
    print(f"📷 Web interface will open automatically in your browser...")
    print(f"   If it doesn't open, navigate to: file:///d:/sign/web/index.html")
    print(f"\n{'='*60}\n")
    
    # Open browser in a separate thread after server starts
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host=host, port=port, debug=False)
