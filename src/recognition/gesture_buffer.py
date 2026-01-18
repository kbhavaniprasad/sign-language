"""
Gesture Buffer for Sequential Gesture Recognition
"""
import numpy as np
from collections import deque
from typing import Optional, Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class GestureBuffer:
    """Buffer for collecting sequential frames for gesture recognition"""
    
    def __init__(self, buffer_size: int = 30, confidence_threshold: float = 0.7):
        """
        Initialize gesture buffer
        
        Args:
            buffer_size: Number of frames to buffer
            confidence_threshold: Minimum confidence for prediction
        """
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.frame_buffer = deque(maxlen=buffer_size)
        self.prediction_history = deque(maxlen=10)
        
        logger.info(f"GestureBuffer initialized (size={buffer_size})")
    
    def add_frame(self, frame: np.ndarray):
        """
        Add frame to buffer
        
        Args:
            frame: Preprocessed frame
        """
        self.frame_buffer.append(frame)
    
    def is_ready(self) -> bool:
        """Check if buffer has enough frames"""
        return len(self.frame_buffer) >= self.buffer_size
    
    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Get buffered sequence
        
        Returns:
            Numpy array of shape (buffer_size, height, width, channels)
        """
        if not self.is_ready():
            return None
        
        return np.array(list(self.frame_buffer))
    
    def add_prediction(self, class_id: int, confidence: float):
        """
        Add prediction to history
        
        Args:
            class_id: Predicted class ID
            confidence: Prediction confidence
        """
        self.prediction_history.append((class_id, confidence))
    
    def get_stable_prediction(self) -> Optional[Tuple[int, float]]:
        """
        Get stable prediction using temporal smoothing
        
        Returns:
            Tuple of (class_id, confidence) or None
        """
        if len(self.prediction_history) < 3:
            return None
        
        # Get recent predictions
        recent_predictions = list(self.prediction_history)[-5:]
        
        # Count occurrences of each class
        class_counts = {}
        confidence_sum = {}
        
        for class_id, confidence in recent_predictions:
            if confidence < self.confidence_threshold:
                continue
                
            if class_id not in class_counts:
                class_counts[class_id] = 0
                confidence_sum[class_id] = 0.0
            
            class_counts[class_id] += 1
            confidence_sum[class_id] += confidence
        
        if not class_counts:
            return None
        
        # Get most frequent class
        most_common_class = max(class_counts, key=class_counts.get)
        
        # Check if it appears in at least 60% of recent predictions
        if class_counts[most_common_class] / len(recent_predictions) >= 0.6:
            avg_confidence = confidence_sum[most_common_class] / class_counts[most_common_class]
            return most_common_class, avg_confidence
        
        return None
    
    def clear(self):
        """Clear buffer"""
        self.frame_buffer.clear()
        logger.debug("Buffer cleared")
    
    def reset_predictions(self):
        """Reset prediction history"""
        self.prediction_history.clear()
        logger.debug("Prediction history reset")
