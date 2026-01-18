"""
Hybrid Sign Language Recognition Model Training Script

This script trains a model on both:
- Static signs: 36 classes (0-9, A-Z) from images
- Dynamic gestures: 8 classes from video frames

Total: 44 classes
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("HYBRID SIGN LANGUAGE RECOGNITION MODEL TRAINING")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print("=" * 80)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""
    
    # Paths
    STATIC_DATASET_PATH = 'isl_dataset'
    DYNAMIC_DATASET_PATH = 'processed_dynamic_dataset'
    MODEL_SAVE_PATH = 'models/hybrid_sign_language_model.h5'
    MODEL_INFO_PATH = 'models/hybrid_model_info.json'
    LOGS_DIR = 'logs'
    
    # Model parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # Dataset info
    STATIC_CLASSES = 36  # 0-9, A-Z
    DYNAMIC_CLASSES = 8  # loud, quiet, happy, sad, Beautiful, Ugly, Deaf, Blind
    TOTAL_CLASSES = STATIC_CLASSES + DYNAMIC_CLASSES  # 44
    
    # Training parameters
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5


config = Config()

# Create directories
os.makedirs(config.LOGS_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)


# ============================================================================
# LOGGING SETUP
# ============================================================================

class TrainingLogger:
    """Comprehensive training logger"""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'training_log_{timestamp}.txt'
        self.history_file = self.log_dir / f'training_history_{timestamp}.json'
        
        self.logs = []
        
    def log(self, message, print_console=True):
        """Log message to file and optionally console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        if print_console:
            print(message)
        
        self.logs.append(log_entry)
        
        # Write to file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def log_section(self, title):
        """Log a section header"""
        separator = "=" * 80
        self.log(f"\n{separator}")
        self.log(title)
        self.log(separator)
    
    def save_history(self, history_dict):
        """Save training history to JSON"""
        with open(self.history_file, 'w') as f:
            json.dump(history_dict, indent=2, fp=f)
        self.log(f"Training history saved to: {self.history_file}")


logger = TrainingLogger()


# ============================================================================
# DATA LOADING
# ============================================================================

def check_dataset_exists(dataset_path, dataset_name):
    """Check if dataset exists"""
    if not os.path.exists(dataset_path):
        logger.log(f"ERROR: {dataset_name} not found at: {dataset_path}")
        logger.log(f"Please ensure the dataset is available.")
        return False
    return True


def load_static_dataset():
    """Load static sign dataset (36 classes)"""
    logger.log_section("LOADING STATIC DATASET")
    
    if not check_dataset_exists(config.STATIC_DATASET_PATH, "Static dataset"):
        return None, None, None, []
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load datasets
    train_path = os.path.join(config.STATIC_DATASET_PATH, 'train')
    val_path = os.path.join(config.STATIC_DATASET_PATH, 'val')
    test_path = os.path.join(config.STATIC_DATASET_PATH, 'test')
    
    train_gen = train_datagen.flow_from_directory(
        train_path,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_gen = val_test_datagen.flow_from_directory(
        val_path,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_gen = val_test_datagen.flow_from_directory(
        test_path,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    static_classes = sorted(train_gen.class_indices.keys())
    
    logger.log(f"Static dataset loaded from: {config.STATIC_DATASET_PATH}")
    logger.log(f"  Classes: {len(static_classes)}")
    logger.log(f"  Train samples: {train_gen.n}")
    logger.log(f"  Validation samples: {val_gen.n}")
    logger.log(f"  Test samples: {test_gen.n}")
    logger.log(f"  Class names: {static_classes}")
    
    return train_gen, val_gen, test_gen, static_classes


def load_dynamic_dataset():
    """Load dynamic gesture dataset (8 classes from videos)"""
    logger.log_section("LOADING DYNAMIC DATASET")
    
    if not check_dataset_exists(config.DYNAMIC_DATASET_PATH, "Dynamic dataset"):
        logger.log("WARNING: Dynamic dataset not found. Run extract_video_frames.py first.")
        logger.log("Proceeding with static dataset only...")
        return None, None, None, []
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load datasets
    train_path = os.path.join(config.DYNAMIC_DATASET_PATH, 'train')
    val_path = os.path.join(config.DYNAMIC_DATASET_PATH, 'val')
    test_path = os.path.join(config.DYNAMIC_DATASET_PATH, 'test')
    
    train_gen = train_datagen.flow_from_directory(
        train_path,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_gen = val_test_datagen.flow_from_directory(
        val_path,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_gen = val_test_datagen.flow_from_directory(
        test_path,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    dynamic_classes = sorted(train_gen.class_indices.keys())
    
    logger.log(f"Dynamic dataset loaded from: {config.DYNAMIC_DATASET_PATH}")
    logger.log(f"  Classes: {len(dynamic_classes)}")
    logger.log(f"  Train samples: {train_gen.n}")
    logger.log(f"  Validation samples: {val_gen.n}")
    logger.log(f"  Test samples: {test_gen.n}")
    logger.log(f"  Class names: {dynamic_classes}")
    
    return train_gen, val_gen, test_gen, dynamic_classes


class HybridDataGenerator(keras.utils.Sequence):
    """Combined data generator for static and dynamic datasets"""
    
    def __init__(self, static_gen, dynamic_gen, static_classes, dynamic_classes):
        self.static_gen = static_gen
        self.dynamic_gen = dynamic_gen
        self.static_classes = static_classes
        self.dynamic_classes = dynamic_classes
        
        # Calculate total samples and batches
        self.static_samples = static_gen.n if static_gen else 0
        self.dynamic_samples = dynamic_gen.n if dynamic_gen else 0
        self.total_samples = self.static_samples + self.dynamic_samples
        
        # Number of batches
        self.static_batches = len(static_gen) if static_gen else 0
        self.dynamic_batches = len(dynamic_gen) if dynamic_gen else 0
        self.total_batches = self.static_batches + self.dynamic_batches
        
        self.n = self.total_samples
        
    def __len__(self):
        return self.total_batches
    
    def __getitem__(self, idx):
        # Alternate between static and dynamic batches
        if self.static_gen and idx < self.static_batches:
            # Get static batch
            images, labels = self.static_gen[idx % len(self.static_gen)]
            # Labels are already one-hot for classes 0-35
            return images, labels
        elif self.dynamic_gen:
            # Get dynamic batch
            dynamic_idx = idx - self.static_batches
            images, labels = self.dynamic_gen[dynamic_idx % len(self.dynamic_gen)]
            
            # Remap labels: dynamic classes are 36-43
            # Convert one-hot back to class indices
            class_indices = np.argmax(labels, axis=1)
            # Add offset for dynamic classes
            class_indices += config.STATIC_CLASSES
            # Convert back to one-hot with 44 classes
            new_labels = keras.utils.to_categorical(class_indices, num_classes=config.TOTAL_CLASSES)
            
            return images, new_labels
        else:
            # Fallback to static
            images, labels = self.static_gen[idx % len(self.static_gen)]
            return images, labels
    
    def on_epoch_end(self):
        if self.static_gen:
            self.static_gen.on_epoch_end()
        if self.dynamic_gen:
            self.dynamic_gen.on_epoch_end()


# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_hybrid_model(num_classes=44, input_shape=(224, 224, 3)):
    """Build hybrid model for static and dynamic sign recognition"""
    logger.log_section("BUILDING HYBRID MODEL")
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Log model summary
    logger.log(f"Model architecture:")
    logger.log(f"  Base: MobileNetV2 (ImageNet weights)")
    logger.log(f"  Input shape: {input_shape}")
    logger.log(f"  Output classes: {num_classes}")
    
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    logger.log(f"  Total parameters: {total_params:,}")
    logger.log(f"  Trainable parameters: {trainable_params:,}")
    logger.log(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_gen, val_gen, class_names):
    """Train the hybrid model"""
    logger.log_section("TRAINING MODEL")
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        config.MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stop, reduce_lr]
    
    logger.log(f"Training configuration:")
    logger.log(f"  Epochs: {config.EPOCHS}")
    logger.log(f"  Batch size: {config.BATCH_SIZE}")
    logger.log(f"  Learning rate: {config.LEARNING_RATE}")
    logger.log(f"  Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    logger.log(f"  Reduce LR patience: {config.REDUCE_LR_PATIENCE}")
    logger.log("")
    logger.log("Starting training...")
    logger.log("")
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.log("")
    logger.log("Training complete!")
    
    # Save model info
    model_info = {
        'num_classes': config.TOTAL_CLASSES,
        'static_classes': config.STATIC_CLASSES,
        'dynamic_classes': config.DYNAMIC_CLASSES,
        'class_names': class_names,
        'image_size': config.IMAGE_SIZE,
        'model_path': config.MODEL_SAVE_PATH,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'epochs_trained': len(history.history['loss']),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy']))
    }
    
    with open(config.MODEL_INFO_PATH, 'w') as f:
        json.dump(model_info, indent=2, fp=f)
    
    logger.log(f"Model saved to: {config.MODEL_SAVE_PATH}")
    logger.log(f"Model info saved to: {config.MODEL_INFO_PATH}")
    
    return history


# ============================================================================
# EVALUATION
# ============================================================================

def plot_training_history(history):
    """Plot training history"""
    logger.log_section("PLOTTING TRAINING HISTORY")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(config.LOGS_DIR, 'hybrid_training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.log(f"Training history plot saved to: {save_path}")


def evaluate_model(model, test_gen, class_names):
    """Evaluate model on test set"""
    logger.log_section("EVALUATING MODEL")
    
    # Get predictions
    logger.log("Generating predictions on test set...")
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = test_gen.classes if hasattr(test_gen, 'classes') else []
    
    if len(true_classes) == 0:
        # For hybrid generator, need to reconstruct true labels
        logger.log("Reconstructing true labels from generator...")
        true_classes = []
        for i in range(len(test_gen)):
            _, labels = test_gen[i]
            true_classes.extend(np.argmax(labels, axis=1))
        true_classes = np.array(true_classes)
    
    # Calculate accuracy
    test_accuracy = np.mean(predicted_classes == true_classes)
    logger.log(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Classification report
    logger.log("\nClassification Report:")
    report = classification_report(true_classes, predicted_classes, 
                                   target_names=class_names, 
                                   zero_division=0)
    logger.log(report)
    
    # Confusion matrix
    logger.log("Generating confusion matrix...")
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Hybrid Model (44 Classes)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(config.LOGS_DIR, 'confusion_matrix_44classes.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.log(f"Confusion matrix saved to: {cm_path}")
    
    # Calculate per-category accuracy
    static_mask = true_classes < config.STATIC_CLASSES
    dynamic_mask = true_classes >= config.STATIC_CLASSES
    
    if np.any(static_mask):
        static_accuracy = np.mean(predicted_classes[static_mask] == true_classes[static_mask])
        logger.log(f"\nStatic Signs Accuracy (classes 0-35): {static_accuracy * 100:.2f}%")
    
    if np.any(dynamic_mask):
        dynamic_accuracy = np.mean(predicted_classes[dynamic_mask] == true_classes[dynamic_mask])
        logger.log(f"Dynamic Gestures Accuracy (classes 36-43): {dynamic_accuracy * 100:.2f}%")
    
    return test_accuracy


def plot_sample_predictions(model, test_gen, class_names, num_samples=16):
    """Plot sample predictions"""
    logger.log_section("PLOTTING SAMPLE PREDICTIONS")
    
    # Get a batch of test images
    images, labels = next(iter(test_gen))
    
    # Get predictions
    predictions = model.predict(images[:num_samples])
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        axes[i].imshow(images[i])
        
        true_label = class_names[np.argmax(labels[i])]
        pred_label = class_names[np.argmax(predictions[i])]
        confidence = np.max(predictions[i])
        
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}',
                         color=color, fontsize=10, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(config.LOGS_DIR, 'sample_predictions_hybrid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.log(f"Sample predictions saved to: {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    logger.log_section("HYBRID SIGN LANGUAGE RECOGNITION - TRAINING PIPELINE")
    logger.log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load datasets
    static_train, static_val, static_test, static_classes = load_static_dataset()
    dynamic_train, dynamic_val, dynamic_test, dynamic_classes = load_dynamic_dataset()
    
    # Check if we have at least one dataset
    if static_train is None and dynamic_train is None:
        logger.log("ERROR: No datasets available. Exiting.")
        return
    
    # Create combined class names
    all_classes = static_classes + dynamic_classes
    logger.log(f"\nTotal classes: {len(all_classes)}")
    logger.log(f"  Static: {static_classes}")
    logger.log(f"  Dynamic: {dynamic_classes}")
    
    # Create hybrid generators
    logger.log_section("CREATING HYBRID DATA GENERATORS")
    
    train_gen = HybridDataGenerator(static_train, dynamic_train, static_classes, dynamic_classes)
    val_gen = HybridDataGenerator(static_val, dynamic_val, static_classes, dynamic_classes)
    test_gen = HybridDataGenerator(static_test, dynamic_test, static_classes, dynamic_classes)
    
    logger.log(f"Hybrid generators created:")
    logger.log(f"  Train samples: {train_gen.n}")
    logger.log(f"  Val samples: {val_gen.n}")
    logger.log(f"  Test samples: {test_gen.n}")
    
    # Build model
    model = build_hybrid_model(num_classes=len(all_classes))
    
    # Train model
    history = train_model(model, train_gen, val_gen, all_classes)
    
    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    logger.save_history(history_dict)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    test_accuracy = evaluate_model(model, test_gen, all_classes)
    
    # Plot sample predictions
    plot_sample_predictions(model, test_gen, all_classes)
    
    # Final summary
    logger.log_section("TRAINING COMPLETE")
    logger.log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"\nFinal Results:")
    logger.log(f"  Total classes: {len(all_classes)}")
    logger.log(f"  Test accuracy: {test_accuracy * 100:.2f}%")
    logger.log(f"  Model saved: {config.MODEL_SAVE_PATH}")
    logger.log(f"  Logs directory: {config.LOGS_DIR}")
    logger.log("\n✓ All done! Check the logs directory for detailed results.")


if __name__ == '__main__':
    main()
