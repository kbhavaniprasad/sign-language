"""
Video processing utilities
"""
import cv2
import numpy as np
from typing import Tuple, Optional
import time

class VideoProcessor:
    """Utility class for video processing operations"""
    
    def __init__(self):
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0
        
    def preprocess_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess a video frame for model input
        
        Args:
            frame: Input frame (BGR format)
            target_size: Target size for resizing
            
        Returns:
            Preprocessed frame
        """
        # Resize
        resized = cv2.resize(frame, target_size)
        
        # Convert to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        return normalized
    
    def calculate_fps(self) -> float:
        """
        Calculate current FPS
        
        Returns:
            Current FPS value
        """
        self.fps_frame_count += 1
        elapsed_time = time.time() - self.fps_start_time
        
        if elapsed_time > 1.0:
            self.current_fps = self.fps_frame_count / elapsed_time
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
            
        return self.current_fps
    
    def draw_text(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                  font_scale: float = 1.0, color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
        """
        Draw text on frame with background
        
        Args:
            frame: Input frame
            text: Text to draw
            position: (x, y) position
            font_scale: Font scale
            color: Text color (BGR)
            thickness: Text thickness
            
        Returns:
            Frame with text
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        x, y = position
        cv2.rectangle(frame, 
                     (x, y - text_height - 10), 
                     (x + text_width + 10, y + baseline),
                     (0, 0, 0), 
                     -1)
        
        # Draw text
        cv2.putText(frame, text, (x + 5, y - 5), font, font_scale, color, thickness)
        
        return frame
    
    def draw_landmarks(self, frame: np.ndarray, landmarks, 
                       connections=None, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw landmarks and connections on frame
        
        Args:
            frame: Input frame
            landmarks: Landmark points
            connections: List of landmark connections
            color: Drawing color (BGR)
            
        Returns:
            Frame with landmarks
        """
        h, w, _ = frame.shape
        
        # Draw connections
        if connections:
            for connection in connections:
                start_idx, end_idx = connection
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                
                start_x, start_y = int(start_point.x * w), int(start_point.y * h)
                end_x, end_y = int(end_point.x * w), int(end_point.y * h)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
        
        # Draw landmarks
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, color, -1)
        
        return frame
    
    @staticmethod
    def save_frame(frame: np.ndarray, filepath: str):
        """Save frame to file"""
        cv2.imwrite(filepath, frame)
    
    @staticmethod
    def create_video_writer(filepath: str, fps: int, frame_size: Tuple[int, int]):
        """Create video writer"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(filepath, fourcc, fps, frame_size)
