"""
MediaPipe Hand Landmark Extractor
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, List, Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class LandmarkExtractor:
    """Extract hand landmarks using MediaPipe"""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 max_num_hands: int = 2):
        """
        Initialize MediaPipe Hands
        
        Args:
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            max_num_hands: Maximum number of hands to detect
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        logger.info("MediaPipe Hands initialized")
    
    def extract_landmarks(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List]]:
        """
        Extract hand landmarks from frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (normalized_landmarks, raw_landmarks)
            normalized_landmarks: numpy array of shape (21, 3) or (42, 3) for 2 hands
            raw_landmarks: MediaPipe landmark objects
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None, None
        
        # Extract landmarks
        all_landmarks = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            
            # Get wrist position for normalization
            wrist = hand_landmarks.landmark[0]
            
            for landmark in hand_landmarks.landmark:
                # Normalize relative to wrist
                landmarks.append([
                    landmark.x - wrist.x,
                    landmark.y - wrist.y,
                    landmark.z - wrist.z
                ])
            
            all_landmarks.extend(landmarks)
        
        # Convert to numpy array
        landmarks_array = np.array(all_landmarks, dtype=np.float32)
        
        # Pad if only one hand detected (to maintain consistent shape)
        if len(results.multi_hand_landmarks) == 1:
            padding = np.zeros((21, 3), dtype=np.float32)
            landmarks_array = np.vstack([landmarks_array, padding])
        
        return landmarks_array, results.multi_hand_landmarks
    
    def draw_landmarks(self, frame: np.ndarray, hand_landmarks_list: List) -> np.ndarray:
        """
        Draw hand landmarks on frame
        
        Args:
            frame: Input frame
            hand_landmarks_list: List of hand landmarks from MediaPipe
            
        Returns:
            Frame with drawn landmarks
        """
        if not hand_landmarks_list:
            return frame
        
        annotated_frame = frame.copy()
        
        for hand_landmarks in hand_landmarks_list:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return annotated_frame
    
    def extract_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract landmarks from static image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Normalized landmarks array
        """
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        landmarks, _ = self.extract_landmarks(frame)
        return landmarks
    
    def close(self):
        """Release resources"""
        self.hands.close()
        logger.info("MediaPipe Hands closed")
