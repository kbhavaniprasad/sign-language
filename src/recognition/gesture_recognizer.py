"""
Real-Time Gesture Recognizer
"""
import cv2
import numpy as np
import json
import os
from typing import Optional, Tuple, List
from tensorflow import keras
from src.data.landmark_extractor import LandmarkExtractor
from src.recognition.gesture_buffer import GestureBuffer
from src.utils.video_utils import VideoProcessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class GestureRecognizer:
    """Real-time sign language gesture recognizer"""
    
    def __init__(self, 
                 model_path: str,
                 model_info_path: str = None,
                 buffer_size: int = 30,
                 confidence_threshold: float = 0.7):
        """
        Initialize gesture recognizer
        
        Args:
            model_path: Path to trained .h5 model
            model_info_path: Path to model info JSON
            buffer_size: Number of frames to buffer
            confidence_threshold: Minimum confidence for prediction
        """
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = keras.models.load_model(model_path)
        
        # Load model info
        if model_info_path and os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                self.class_names = model_info['class_names']
                self.image_size = tuple(model_info['image_size'])
                self.num_classes = model_info['num_classes']
        else:
            logger.warning("Model info not found, using defaults")
            self.class_names = [f"Class_{i}" for i in range(self.model.output_shape[-1])]
            self.image_size = (224, 224)
            self.num_classes = len(self.class_names)
        
        # Initialize components
        self.landmark_extractor = LandmarkExtractor()
        self.gesture_buffer = GestureBuffer(buffer_size, confidence_threshold)
        self.video_processor = VideoProcessor()
        
        # State
        self.current_gesture = None
        self.current_confidence = 0.0
        self.gesture_text = ""
        
        logger.info(f"GestureRecognizer initialized with {self.num_classes} classes")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str], float]:
        """
        Process a single frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (annotated_frame, predicted_text, confidence)
        """
        # Extract landmarks
        landmarks, raw_landmarks = self.landmark_extractor.extract_landmarks(frame)
        
        # Draw landmarks on frame
        if raw_landmarks:
            frame = self.landmark_extractor.draw_landmarks(frame, raw_landmarks)
        
        # Preprocess frame for model
        preprocessed = self.video_processor.preprocess_frame(frame, self.image_size)
        
        # Add to buffer
        self.gesture_buffer.add_frame(preprocessed)
        
        # Make prediction if buffer is ready
        predicted_text = None
        confidence = 0.0
        
        if self.gesture_buffer.is_ready():
            # Get single frame (for simple CNN model)
            input_frame = np.expand_dims(preprocessed, axis=0)
            
            # Predict
            prediction = self.model.predict(input_frame, verbose=0)
            class_id = np.argmax(prediction[0])
            confidence = float(prediction[0][class_id])
            
            # Add to prediction history
            self.gesture_buffer.add_prediction(class_id, confidence)
            
            # Get stable prediction
            stable_pred = self.gesture_buffer.get_stable_prediction()
            
            if stable_pred:
                stable_class, stable_conf = stable_pred
                predicted_text = self.class_names[stable_class]
                confidence = stable_conf
                
                # Update current gesture
                if predicted_text != self.current_gesture:
                    self.current_gesture = predicted_text
                    self.current_confidence = confidence
                    
                    # Add to gesture text (with space)
                    if self.gesture_text:
                        self.gesture_text += " "
                    self.gesture_text += predicted_text
                    
                    logger.info(f"Detected: {predicted_text} (confidence: {confidence:.2f})")
        
        # Draw information on frame
        fps = self.video_processor.calculate_fps()
        frame = self._draw_info(frame, fps, predicted_text, confidence)
        
        return frame, predicted_text, confidence
    
    def _draw_info(self, frame: np.ndarray, fps: float, 
                   gesture: Optional[str], confidence: float) -> np.ndarray:
        """Draw information overlay on frame"""
        h, w = frame.shape[:2]
        
        # FPS
        self.video_processor.draw_text(
            frame, f"FPS: {fps:.1f}", (10, 30),
            font_scale=0.7, color=(0, 255, 0)
        )
        
        # Current gesture
        if gesture:
            self.video_processor.draw_text(
                frame, f"Gesture: {gesture}", (10, 70),
                font_scale=1.0, color=(0, 255, 255), thickness=2
            )
            
            self.video_processor.draw_text(
                frame, f"Confidence: {confidence*100:.1f}%", (10, 110),
                font_scale=0.7, color=(0, 255, 255)
            )
        
        # Gesture text (accumulated)
        if self.gesture_text:
            # Draw background for text
            text_y = h - 60
            cv2.rectangle(frame, (0, text_y - 10), (w, h), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, "Recognized Text:", (10, text_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, self.gesture_text, (10, text_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def get_gesture_text(self) -> str:
        """Get accumulated gesture text"""
        return self.gesture_text
    
    def clear_gesture_text(self):
        """Clear accumulated gesture text"""
        self.gesture_text = ""
        self.gesture_buffer.reset_predictions()
        logger.info("Gesture text cleared")
    
    def run_webcam(self, camera_id: int = 0):
        """
        Run real-time recognition from webcam
        
        Args:
            camera_id: Camera device ID
        """
        logger.info(f"Starting webcam recognition (camera {camera_id})")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return
        
        print("\n" + "="*60)
        print("Real-Time Sign Language Recognition")
        print("="*60)
        print("Controls:")
        print("  - Press 'c' to clear gesture text")
        print("  - Press 'q' to quit")
        print("="*60 + "\n")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Process frame
                annotated_frame, gesture, confidence = self.process_frame(frame)
                
                # Display
                cv2.imshow('Sign Language Recognition', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('c'):
                    self.clear_gesture_text()
                    print("Gesture text cleared")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.landmark_extractor.close()
            logger.info("Webcam recognition stopped")
    
    def close(self):
        """Release resources"""
        self.landmark_extractor.close()
        logger.info("GestureRecognizer closed")
