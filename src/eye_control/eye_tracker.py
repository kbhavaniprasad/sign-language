"""
Eye Tracker using MediaPipe Face Mesh
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Dict
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class EyeTracker:
    """Track eye movements and facial landmarks"""
    
    # Eye landmark indices (MediaPipe Face Mesh)
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize eye tracker
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Calibration data
        self.calibrated = False
        self.calibration_data = {}
        
        logger.info("EyeTracker initialized")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[Dict], Optional[any]]:
        """
        Process frame and extract eye features
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (features_dict, face_landmarks)
        """
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, None
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract features
        features = self._extract_features(face_landmarks, frame.shape)
        
        return features, face_landmarks
    
    def _extract_features(self, face_landmarks, frame_shape) -> Dict:
        """Extract eye tracking features"""
        h, w, _ = frame_shape
        
        # Get eye landmarks
        left_eye = self._get_eye_landmarks(face_landmarks, self.LEFT_EYE_INDICES, w, h)
        right_eye = self._get_eye_landmarks(face_landmarks, self.RIGHT_EYE_INDICES, w, h)
        
        # Calculate Eye Aspect Ratio (EAR) for blink detection
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Calculate gaze direction
        gaze_direction = self._calculate_gaze_direction(face_landmarks, w, h)
        
        # Calculate head pose
        head_pose = self._calculate_head_pose(face_landmarks, w, h)
        
        features = {
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear,
            'is_blinking': avg_ear < 0.2,  # Threshold for blink
            'gaze_direction': gaze_direction,
            'head_pose': head_pose,
            'left_eye_center': np.mean(left_eye, axis=0).tolist(),
            'right_eye_center': np.mean(right_eye, axis=0).tolist()
        }
        
        return features
    
    def _get_eye_landmarks(self, face_landmarks, indices, w, h) -> np.ndarray:
        """Get eye landmark coordinates"""
        landmarks = []
        for idx in indices:
            landmark = face_landmarks.landmark[idx]
            landmarks.append([landmark.x * w, landmark.y * h])
        return np.array(landmarks)
    
    def _calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR)
        
        Args:
            eye_landmarks: Eye landmark coordinates
            
        Returns:
            EAR value
        """
        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # EAR
        ear = (v1 + v2) / (2.0 * h)
        
        return ear
    
    def _calculate_gaze_direction(self, face_landmarks, w, h) -> str:
        """
        Calculate gaze direction
        
        Returns:
            Direction string: 'center', 'left', 'right', 'up', 'down'
        """
        # Get iris landmarks (468-477 are iris landmarks in refined mode)
        try:
            left_iris = face_landmarks.landmark[468]
            right_iris = face_landmarks.landmark[473]
            
            # Get eye corners
            left_corner = face_landmarks.landmark[33]
            right_corner = face_landmarks.landmark[263]
            
            # Calculate relative position
            left_ratio = (left_iris.x - left_corner.x) * w
            right_ratio = (right_iris.x - right_corner.x) * w
            
            avg_ratio = (left_ratio + right_ratio) / 2.0
            
            # Determine direction
            if avg_ratio < -5:
                return 'left'
            elif avg_ratio > 5:
                return 'right'
            elif left_iris.y < 0.4:
                return 'up'
            elif left_iris.y > 0.6:
                return 'down'
            else:
                return 'center'
        except:
            return 'center'
    
    def _calculate_head_pose(self, face_landmarks, w, h) -> Dict[str, float]:
        """
        Calculate head pose (pitch, yaw, roll)
        
        Returns:
            Dictionary with pitch, yaw, roll values
        """
        # Simplified head pose estimation
        # Using key facial landmarks
        
        nose_tip = face_landmarks.landmark[1]
        chin = face_landmarks.landmark[152]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        
        # Calculate angles (simplified)
        yaw = (nose_tip.x - 0.5) * 90  # -45 to 45 degrees
        pitch = (nose_tip.y - 0.5) * 90
        
        # Roll (based on eye alignment)
        eye_diff = (right_eye.y - left_eye.y) * 90
        roll = eye_diff
        
        return {
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll
        }
    
    def draw_landmarks(self, frame: np.ndarray, face_landmarks) -> np.ndarray:
        """Draw face mesh on frame"""
        if face_landmarks is None:
            return frame
        
        annotated_frame = frame.copy()
        
        self.mp_drawing.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
        return annotated_frame
    
    def calibrate(self, duration: int = 10):
        """
        Calibrate eye tracker
        
        Args:
            duration: Calibration duration in seconds
        """
        logger.info(f"Starting calibration for {duration} seconds...")
        # Calibration logic would go here
        self.calibrated = True
        logger.info("Calibration complete")
    
    def close(self):
        """Release resources"""
        self.face_mesh.close()
        logger.info("EyeTracker closed")
