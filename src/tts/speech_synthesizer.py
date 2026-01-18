"""
Text-to-Speech Synthesizer
"""
import os
from typing import Optional
from gtts import gTTS
import pygame
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SpeechSynthesizer:
    """Convert text to speech in multiple languages"""
    
    def __init__(self, output_dir: str = 'temp'):
        """
        Initialize speech synthesizer
        
        Args:
            output_dir: Directory to save audio files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init()
            self.audio_enabled = True
        except Exception as e:
            logger.warning(f"Audio playback not available: {e}")
            self.audio_enabled = False
        
        logger.info("SpeechSynthesizer initialized")
    
    def text_to_speech(self, text: str, language: str = 'en', 
                       play: bool = True, save: bool = False,
                       filename: Optional[str] = None) -> Optional[str]:
        """
        Convert text to speech
        
        Args:
            text: Text to convert
            language: Language code (e.g., 'en', 'es', 'fr')
            play: Whether to play the audio
            save: Whether to save the audio file
            filename: Custom filename (optional)
            
        Returns:
            Path to audio file or None
        """
        if not text or not text.strip():
            logger.warning("Empty text provided")
            return None
        
        try:
            # Generate speech
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Generate filename
            if not filename:
                filename = f"speech_{hash(text) % 10000}.mp3"
            
            filepath = os.path.join(self.output_dir, filename)
            
            # Save audio
            tts.save(filepath)
            logger.info(f"Speech generated: {filepath}")
            
            # Play audio
            if play and self.audio_enabled:
                self.play_audio(filepath)
            
            # Delete file if not saving
            if not save and os.path.exists(filepath):
                # Wait a bit before deleting (if playing)
                if play:
                    import time
                    time.sleep(0.5)
                
                try:
                    os.remove(filepath)
                except:
                    pass  # File might still be in use
            
            return filepath if save else None
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    def play_audio(self, filepath: str):
        """
        Play audio file
        
        Args:
            filepath: Path to audio file
        """
        if not self.audio_enabled:
            logger.warning("Audio playback not available")
            return
        
        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            logger.info(f"Played audio: {filepath}")
            
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
    
    def speak(self, text: str, language: str = 'en'):
        """
        Quick speak function (play without saving)
        
        Args:
            text: Text to speak
            language: Language code
        """
        self.text_to_speech(text, language, play=True, save=False)
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            for file in os.listdir(self.output_dir):
                if file.endswith('.mp3'):
                    os.remove(os.path.join(self.output_dir, file))
            logger.info("Temporary audio files cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def close(self):
        """Release resources"""
        if self.audio_enabled:
            pygame.mixer.quit()
        self.cleanup()
        logger.info("SpeechSynthesizer closed")
