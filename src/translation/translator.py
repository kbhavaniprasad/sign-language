"""
Multilingual Translator using Google Translate
"""
from typing import Optional
from googletrans import Translator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MultilingualTranslator:
    """Translate text to multiple languages"""
    
    def __init__(self):
        """Initialize translator"""
        self.translator = Translator()
        self.translation_cache = {}
        
        # Supported languages
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'hi': 'Hindi',
            'zh-cn': 'Chinese (Simplified)',
            'ja': 'Japanese',
            'ar': 'Arabic',
            'ru': 'Russian',
            'pt': 'Portuguese',
            'it': 'Italian',
            'ko': 'Korean',
            'tr': 'Turkish',
            'nl': 'Dutch',
            'pl': 'Polish',
            'sv': 'Swedish',
            'fi': 'Finnish',
            'da': 'Danish',
            'no': 'Norwegian',
            'cs': 'Czech'
        }
        
        logger.info("MultilingualTranslator initialized")
    
    def translate(self, text: str, target_language: str = 'en', 
                  source_language: str = 'auto') -> Optional[str]:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Target language code (e.g., 'es', 'fr')
            source_language: Source language code ('auto' for auto-detect)
            
        Returns:
            Translated text or None if translation fails
        """
        if not text or not text.strip():
            return text
        
        # Check cache
        cache_key = f"{text}_{source_language}_{target_language}"
        if cache_key in self.translation_cache:
            logger.debug(f"Using cached translation for: {text}")
            return self.translation_cache[cache_key]
        
        try:
            # Translate
            result = self.translator.translate(
                text, 
                src=source_language, 
                dest=target_language
            )
            
            translated_text = result.text
            
            # Cache result
            self.translation_cache[cache_key] = translated_text
            
            logger.info(f"Translated '{text}' to {target_language}: '{translated_text}'")
            
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text  # Return original text on error
    
    def get_supported_languages(self) -> dict:
        """Get dictionary of supported languages"""
        return self.supported_languages
    
    def is_language_supported(self, language_code: str) -> bool:
        """Check if language is supported"""
        return language_code in self.supported_languages
    
    def clear_cache(self):
        """Clear translation cache"""
        self.translation_cache.clear()
        logger.info("Translation cache cleared")
