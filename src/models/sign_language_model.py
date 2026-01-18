"""
Sign Language Recognition Model: MobileNetV2 + LSTM
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from typing import Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SignLanguageModel:
    """MobileNetV2 + LSTM model for sign language recognition"""
    
    def __init__(self, 
                 num_classes: int,
                 image_size: Tuple[int, int] = (224, 224),
                 sequence_length: int = 30):
        """
        Initialize model architecture
        
        Args:
            num_classes: Number of gesture classes
            image_size: Input image size
            sequence_length: Number of frames in sequence
        """
        self.num_classes = num_classes
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.model = None
        
        logger.info(f"Initializing model for {num_classes} classes")
    
    def build_model(self, use_sequence: bool = False) -> keras.Model:
        """
        Build the model architecture
        
        Args:
            use_sequence: If True, build LSTM model for sequences
                         If False, build simple CNN for single images
        
        Returns:
            Compiled Keras model
        """
        if use_sequence:
            self.model = self._build_sequence_model()
        else:
            self.model = self._build_simple_model()
        
        return self.model
    
    def _build_simple_model(self) -> keras.Model:
        """
        Build simple CNN model for static gestures
        
        Returns:
            Keras model
        """
        logger.info("Building simple CNN model")
        
        # Input layer
        inputs = layers.Input(shape=(*self.image_size, 3))
        
        # MobileNetV2 backbone (pretrained on ImageNet)
        base_model = MobileNetV2(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Feature extraction
        x = base_model(inputs, training=False)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='sign_language_cnn')
        
        logger.info(f"Model built with {model.count_params():,} parameters")
        
        return model
    
    def _build_sequence_model(self) -> keras.Model:
        """
        Build LSTM model for dynamic gestures (sequences)
        
        Returns:
            Keras model
        """
        logger.info("Building LSTM sequence model")
        
        # Input layer for sequences
        inputs = layers.Input(shape=(self.sequence_length, *self.image_size, 3))
        
        # MobileNetV2 backbone (shared across time steps)
        base_model = MobileNetV2(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        # TimeDistributed wrapper for CNN
        x = layers.TimeDistributed(base_model)(inputs)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
        
        # LSTM layers
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Bidirectional(layers.LSTM(128))(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='sign_language_lstm')
        
        logger.info(f"Model built with {model.count_params():,} parameters")
        
        return model
    
    def compile_model(self, 
                     learning_rate: float = 0.001,
                     optimizer: str = 'adam'):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate
            optimizer: Optimizer name
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # Optimizer
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer
        
        # Compile
        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Model compiled with {optimizer} optimizer (lr={learning_rate})")
    
    def get_model(self) -> keras.Model:
        """Get the model"""
        return self.model
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
        else:
            logger.warning("Model not built yet")
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if self.model:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        else:
            logger.warning("No model to save")
    
    @staticmethod
    def load_model(filepath: str) -> keras.Model:
        """Load model from file"""
        model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
