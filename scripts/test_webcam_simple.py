"""
Simple Webcam Test with Terminal Output
"""
import cv2
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.landmark_extractor import LandmarkExtractor

def test_webcam_simple(camera_id=0):
    """Test webcam with terminal output"""
    print("="*60)
    print("🎥 Simple Webcam Test")
    print("="*60)
    print(f"\nOpening camera {camera_id}...")
    
    # Initialize
    landmark_extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera!")
        print("\nTroubleshooting:")
        print("1. Check if another app is using the camera")
        print("2. Try a different camera ID: python scripts\\test_webcam_simple.py --camera 1")
        return
    
    print("✅ Camera opened successfully!")
    print("\n📹 Camera is working! Processing frames...")
    print("\nWhat's happening:")
    print("- A window should open showing your webcam")
    print("- If you see your hands, landmarks will be drawn")
    print("- Press 'q' in the video window to quit")
    print("\n" + "="*60)
    
    frame_count = 0
    hands_detected_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Extract landmarks
            landmarks, raw_landmarks = landmark_extractor.extract_landmarks(frame)
            
            # Draw landmarks
            if raw_landmarks:
                frame = landmark_extractor.draw_landmarks(frame, raw_landmarks)
                hands_detected_count += 1
                
                # Print to terminal every 30 frames
                if frame_count % 30 == 0:
                    print(f"✋ Frame {frame_count}: {len(raw_landmarks)} hand(s) detected!")
            else:
                if frame_count % 30 == 0:
                    print(f"👁️ Frame {frame_count}: No hands detected")
            
            # Add text to frame
            cv2.putText(frame, "Press 'q' to quit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Hands: {len(raw_landmarks) if raw_landmarks else 0}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display
            cv2.imshow('Webcam Test - Press Q to Quit', frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✅ Quit requested")
                break
            elif key == ord('s'):
                # Save screenshot
                filename = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmark_extractor.close()
        
        print("\n" + "="*60)
        print("📊 Summary:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Frames with hands detected: {hands_detected_count}")
        if frame_count > 0:
            print(f"  Detection rate: {hands_detected_count/frame_count*100:.1f}%")
        print("="*60)
        print("\n✅ Test completed!")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple webcam test')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    args = parser.parse_args()
    
    test_webcam_simple(args.camera)
