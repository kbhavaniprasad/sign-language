"""
Flask API for Sign Language Recognition System
"""
import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yaml

from src.recognition.gesture_recognizer import GestureRecognizer
from src.translation.translator import MultilingualTranslator
from src.tts.speech_synthesizer import SpeechSynthesizer
from src.eye_control.eye_tracker import EyeTracker
from src.eye_control.pc_controller import PCController
from src.utils.logger import setup_logger

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup logger
logger = setup_logger(__name__, log_file='logs/api.log')

# Initialize components
gesture_recognizer = None
translator = MultilingualTranslator()
speech_synthesizer = SpeechSynthesizer()
eye_tracker = EyeTracker()
pc_controller = PCController()

# Global state
current_language = config['translation']['default_language']

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Sign Language Recognition API is running'
    })

@app.route('/languages', methods=['GET'])
def get_languages():
    """Get supported languages"""
    languages = translator.get_supported_languages()
    return jsonify({
        'languages': languages,
        'current': current_language
    })

@app.route('/language', methods=['POST'])
def set_language():
    """Set target language"""
    global current_language
    
    data = request.json
    language = data.get('language', 'en')
    
    if translator.is_language_supported(language):
        current_language = language
        logger.info(f"Language set to: {language}")
        return jsonify({
            'success': True,
            'language': current_language
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Language not supported'
        }), 400

@app.route('/predict', methods=['POST'])
def predict_gesture():
    """Predict gesture from image"""
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
        
        # Process frame
        if gesture_recognizer:
            _, gesture, confidence = gesture_recognizer.process_frame(frame)
            
            # Translate if gesture detected
            translated_text = None
            if gesture and current_language != 'en':
                translated_text = translator.translate(gesture, current_language)
            
            return jsonify({
                'success': True,
                'gesture': gesture,
                'confidence': float(confidence) if confidence else 0.0,
                'translated': translated_text,
                'language': current_language
            })
        else:
            return jsonify({'error': 'Model not loaded'}), 500
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

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
        if gesture_recognizer and gesture_recognizer.model:
            # Preprocess frame
            img_resized = cv2.resize(frame, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype('float32') / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Get predictions
            predictions = gesture_recognizer.model.predict(img_batch, verbose=0)[0]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            top_3_predictions = [
                {
                    'class': gesture_recognizer.class_names[idx],
                    'confidence': float(predictions[idx])
                }
                for idx in top_3_indices
            ]
            
            # Get top prediction
            top_class_idx = top_3_indices[0]
            predicted_class = gesture_recognizer.class_names[top_class_idx]
            confidence = float(predictions[top_class_idx])
            
            return jsonify({
                'success': True,
                'prediction': predicted_class,
                'confidence': confidence,
                'top_3': top_3_predictions
            })
        else:
            return jsonify({'error': 'Model not loaded'}), 500
            
    except Exception as e:
        logger.error(f"Frame prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate_text():
    """Translate text to target language"""
    try:
        data = request.json
        text = data.get('text')
        target_lang = data.get('target_language', current_language)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        translated = translator.translate(text, target_lang)
        
        return jsonify({
            'success': True,
            'original': text,
            'translated': translated,
            'language': target_lang
        })
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/speak', methods=['POST'])
def speak_text():
    """Convert text to speech"""
    try:
        data = request.json
        text = data.get('text')
        language = data.get('language', current_language)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate speech
        audio_file = speech_synthesizer.text_to_speech(
            text, 
            language, 
            play=False, 
            save=True,
            filename=f'speech_{hash(text) % 10000}.mp3'
        )
        
        if audio_file and os.path.exists(audio_file):
            return send_file(audio_file, mimetype='audio/mpeg')
        else:
            return jsonify({'error': 'Failed to generate speech'}), 500
            
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/eye-control/status', methods=['GET'])
def eye_control_status():
    """Get eye control status"""
    return jsonify({
        'enabled': pc_controller.is_enabled(),
        'sensitivity': pc_controller.sensitivity,
        'smoothing': pc_controller.smoothing
    })

@app.route('/eye-control/toggle', methods=['POST'])
def toggle_eye_control():
    """Toggle eye control on/off"""
    data = request.json
    enable = data.get('enable', False)
    
    if enable:
        pc_controller.enable()
    else:
        pc_controller.disable()
    
    return jsonify({
        'success': True,
        'enabled': pc_controller.is_enabled()
    })

@app.route('/eye-control/settings', methods=['POST'])
def update_eye_control_settings():
    """Update eye control settings"""
    data = request.json
    
    if 'sensitivity' in data:
        pc_controller.set_sensitivity(float(data['sensitivity']))
    
    if 'smoothing' in data:
        pc_controller.set_smoothing(float(data['smoothing']))
    
    return jsonify({
        'success': True,
        'sensitivity': pc_controller.sensitivity,
        'smoothing': pc_controller.smoothing
    })

def initialize_model():
    """Initialize gesture recognition model"""
    global gesture_recognizer
    
    model_path = config['models']['sign_language_model']
    model_info_path = 'models/model_info.json'
    
    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        gesture_recognizer = GestureRecognizer(
            model_path=model_path,
            model_info_path=model_info_path if os.path.exists(model_info_path) else None,
            buffer_size=config['recognition']['buffer_size'],
            confidence_threshold=config['recognition']['confidence_threshold']
        )
        logger.info("Model loaded successfully")
    else:
        logger.warning(f"Model not found at {model_path}")

if __name__ == '__main__':
    # Initialize model
    initialize_model()
    
    # Run app
    host = config['api']['host']
    port = config['api']['port']
    debug = config['api']['debug']
    
    logger.info(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
