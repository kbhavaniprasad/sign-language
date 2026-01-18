"""
Data loader for sign language dataset - Memory Efficient Version
"""
import os
import shutil
import numpy as np
from typing import Tuple, List
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SignLanguageDataLoader:
    """Load and preprocess sign language dataset using memory-efficient generators"""
    
    def __init__(self, 
                 dataset_path: str,
                 image_size: Tuple[int, int] = (224, 224),
                 use_landmarks: bool = False):
        """
        Initialize data loader
        
        Args:
            dataset_path: Path to dataset directory
            image_size: Target image size
            use_landmarks: Whether to extract hand landmarks (not used in this version)
        """
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.use_landmarks = use_landmarks
        
        self.class_names = []
        self.num_classes = 0
        
        # Get class information from train directory
        train_path = os.path.join(dataset_path, 'train')
        if os.path.exists(train_path):
            self.class_names = sorted([d for d in os.listdir(train_path) 
                                      if os.path.isdir(os.path.join(train_path, d))])
            self.num_classes = len(self.class_names)
        
        logger.info(f"DataLoader initialized with dataset path: {dataset_path}")
        logger.info(f"Found {self.num_classes} classes: {self.class_names}")
    
    def create_validation_split(self, validation_split: float = 0.2):
        """
        Create validation directory by moving files from train to val
        
        Args:
            validation_split: Fraction of training data to use for validation
        """
        train_path = os.path.join(self.dataset_path, 'train')
        val_path = os.path.join(self.dataset_path, 'val')
        
        # Check if validation directory already exists
        if os.path.exists(val_path):
            logger.info("Validation directory already exists, skipping split")
            return
        
        logger.info(f"Creating validation split ({validation_split * 100}%)")
        os.makedirs(val_path, exist_ok=True)
        
        # For each class, move validation_split% of images to val directory
        for class_name in self.class_names:
            train_class_path = os.path.join(train_path, class_name)
            val_class_path = os.path.join(val_path, class_name)
            os.makedirs(val_class_path, exist_ok=True)
            
            # Get all images in this class
            images = [f for f in os.listdir(train_class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Calculate number of validation images
            num_val = int(len(images) * validation_split)
            
            # Randomly select validation images
            np.random.seed(42)
            val_images = np.random.choice(images, size=num_val, replace=False)
            
            # Move images to validation directory
            for img in val_images:
                src = os.path.join(train_class_path, img)
                dst = os.path.join(val_class_path, img)
                shutil.move(src, dst)
            
            logger.info(f"Class '{class_name}': moved {num_val} images to validation")
    
    def create_data_generators(self, 
                               validation_split: float = 0.2,
                               batch_size: int = 32,
                               augment: bool = True):
        """
        Create memory-efficient data generators using flow_from_directory
        
        Args:
            validation_split: Fraction of training data to use for validation
            batch_size: Batch size
            augment: Whether to apply data augmentation
            
        Returns:
            Tuple of (train_generator, val_generator, test_generator)
        """
        # Create validation split if needed
        val_path = os.path.join(self.dataset_path, 'val')
        if not os.path.exists(val_path):
            self.create_validation_split(validation_split)
        
        train_path = os.path.join(self.dataset_path, 'train')
        test_path = os.path.join(self.dataset_path, 'test')
        
        logger.info("Creating memory-efficient data generators")
        
        # Data augmentation for training
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # No augmentation for validation and test (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators from directories
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='sparse',  # For integer labels
            shuffle=True,
            seed=42
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_path,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=False
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=False
        )
        
        logger.info(f"Train samples: {train_generator.n}")
        logger.info(f"Validation samples: {val_generator.n}")
        logger.info(f"Test samples: {test_generator.n}")
        logger.info(f"Classes: {train_generator.class_indices}")
        
        return train_generator, val_generator, test_generator
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.class_names
    
    def get_num_classes(self) -> int:
        """Get number of classes"""
        return self.num_classes
