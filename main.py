"""
Main Application - Real-Time Sign Language Recognition
"""
import os
import sys
import argparse
import yaml
from src.recognition.gesture_recognizer import GestureRecognizer
from src.translation.translator import MultilingualTranslator
from src.tts.speech_synthesizer import SpeechSynthesizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__, log_file='logs/app.log')

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='Real-Time Sign Language Recognition')
    parser.add_argument('--model', type=str, default='models/sign_language_model.h5',
                       help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--language', type=str, default='en',
                       help='Target language for translation')
    parser.add_argument('--speak', action='store_true',
                       help='Enable text-to-speech')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found: {args.config}")
        config = {}
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        print(f"\n❌ Error: Model file not found at {args.model}")
        print("\nPlease train the model first using the Jupyter notebook:")
        print("  jupyter notebook train_model.ipynb")
        sys.exit(1)
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Gesture recognizer
    recognizer = GestureRecognizer(
        model_path=args.model,
        model_info_path='models/model_info.json'
    )
    
    # Translator
    translator = MultilingualTranslator()
    
    # Speech synthesizer (if enabled)
    tts = None
    if args.speak:
        tts = SpeechSynthesizer()
    
    # Print welcome message
    print("\n" + "="*70)
    print("🤟 REAL-TIME SIGN LANGUAGE RECOGNITION SYSTEM 🤟")
    print("="*70)
    print(f"\n📹 Camera: {args.camera}")
    print(f"🤖 Model: {args.model}")
    print(f"🌍 Language: {args.language}")
    print(f"🔊 Text-to-Speech: {'Enabled' if args.speak else 'Disabled'}")
    print("\n" + "="*70)
    print("\n📋 Controls:")
    print("  - Press 'c' to clear recognized text")
    print("  - Press 's' to speak current text")
    print("  - Press 't' to translate and speak")
    print("  - Press 'q' to quit")
    print("="*70 + "\n")
    
    # Run webcam recognition
    try:
        # Custom webcam loop with translation and TTS
        import cv2
        
        cap = cv2.VideoCapture(args.camera)
        
        if not cap.isOpened():
            logger.error("Failed to open camera")
            print("❌ Error: Could not open camera")
            sys.exit(1)
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process frame
            annotated_frame, gesture, confidence = recognizer.process_frame(frame)
            
            # Display
            cv2.imshow('Sign Language Recognition', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                recognizer.clear_gesture_text()
                print("✅ Text cleared")
            elif key == ord('s'):
                if tts:
                    text = recognizer.get_gesture_text()
                    if text:
                        print(f"🔊 Speaking: {text}")
                        tts.speak(text, 'en')
            elif key == ord('t'):
                if tts:
                    text = recognizer.get_gesture_text()
                    if text:
                        translated = translator.translate(text, args.language)
                        print(f"🌍 Translated: {translated}")
                        print(f"🔊 Speaking in {args.language}...")
                        tts.speak(translated, args.language)
        
        cap.release()
        cv2.destroyAllWindows()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n\n👋 Goodbye!")
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\n❌ Error: {e}")
    
    finally:
        # Cleanup
        recognizer.close()
        if tts:
            tts.close()
        logger.info("Application closed")

if __name__ == '__main__':
    main()
