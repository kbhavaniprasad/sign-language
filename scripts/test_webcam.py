"""
Test Webcam and MediaPipe Integration
"""
import cv2
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.landmark_extractor import LandmarkExtractor
from src.utils.video_utils import VideoProcessor

def test_webcam(camera_id=0):
    """Test webcam with MediaPipe hand detection"""
    print("="*60)
    print("Testing Webcam and MediaPipe Hand Detection")
    print("="*60)
    print(f"\nCamera ID: {camera_id}")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("="*60 + "\n")
    
    # Initialize components
    landmark_extractor = LandmarkExtractor()
    video_processor = VideoProcessor()
    
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("✅ Camera opened successfully")
    print("📹 Showing live feed with hand landmarks...\n")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Failed to read frame")
                break
            
            # Extract landmarks
            landmarks, raw_landmarks = landmark_extractor.extract_landmarks(frame)
            
            # Draw landmarks
            if raw_landmarks:
                frame = landmark_extractor.draw_landmarks(frame, raw_landmarks)
                
                # Show landmark count
                video_processor.draw_text(
                    frame, 
                    f"Hands detected: {len(raw_landmarks)}", 
                    (10, 30),
                    color=(0, 255, 0)
                )
            else:
                video_processor.draw_text(
                    frame, 
                    "No hands detected", 
                    (10, 30),
                    color=(0, 0, 255)
                )
            
            # Calculate and display FPS
            fps = video_processor.calculate_fps()
            video_processor.draw_text(
                frame, 
                f"FPS: {fps:.1f}", 
                (10, 70),
                color=(255, 255, 0)
            )
            
            # Display
            cv2.imshow('Webcam Test - Hand Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmark_extractor.close()
        print("\n✅ Test completed")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test webcam and MediaPipe')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    args = parser.parse_args()
    
    test_webcam(args.camera)
