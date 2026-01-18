# Hybrid Model - Notebook Integration Guide

This guide explains how to integrate the hybrid training functionality into your `train_model.ipynb` notebook.

## Quick Start - Use the Standalone Script

The easiest way is to use the standalone Python script:

```bash
# 1. Extract frames from videos
python scripts/extract_video_frames.py

# 2. Train hybrid model
python train_hybrid_model.py
```

This will:
- Extract frames from the `Adjectives/` video dataset
- Train on both static (36 classes) and dynamic (8 classes) datasets
- Save comprehensive logs to `logs/` directory
- Save model to `models/hybrid_sign_language_model.h5`

---

## Integrating into Jupyter Notebook

If you prefer to use the notebook, follow these steps:

### Step 1: Add Video Frame Extraction Cell

Add this cell at the beginning of your notebook:

```python
# Extract frames from dynamic gesture videos
import subprocess
import os

if not os.path.exists('processed_dynamic_dataset'):
    print("Extracting frames from videos...")
    subprocess.run(['python', 'scripts/extract_video_frames.py'], check=True)
    print("✓ Frame extraction complete!")
else:
    print("✓ Dynamic dataset already processed")
```

### Step 2: Update Configuration Cell

Replace the configuration cell with:

```python
# Configuration
STATIC_DATASET_PATH = 'isl_dataset'
DYNAMIC_DATASET_PATH = 'processed_dynamic_dataset'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Dataset info
STATIC_CLASSES = 36  # 0-9, A-Z
DYNAMIC_CLASSES = 8  # loud, quiet, happy, sad, Beautiful, Ugly, Deaf, Blind
TOTAL_CLASSES = STATIC_CLASSES + DYNAMIC_CLASSES  # 44

print(f"Training configuration:")
print(f"  Total classes: {TOTAL_CLASSES}")
print(f"  Static classes: {STATIC_CLASSES}")
print(f"  Dynamic classes: {DYNAMIC_CLASSES}")
```

### Step 3: Add Hybrid Data Generator Class

Add this cell after imports:

```python
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
            return images, labels
        elif self.dynamic_gen:
            # Get dynamic batch
            dynamic_idx = idx - self.static_batches
            images, labels = self.dynamic_gen[dynamic_idx % len(self.dynamic_gen)]
            
            # Remap labels: dynamic classes are 36-43
            class_indices = np.argmax(labels, axis=1)
            class_indices += STATIC_CLASSES
            new_labels = keras.utils.to_categorical(class_indices, num_classes=TOTAL_CLASSES)
            
            return images, new_labels
        else:
            images, labels = self.static_gen[idx % len(self.static_gen)]
            return images, labels
    
    def on_epoch_end(self):
        if self.static_gen:
            self.static_gen.on_epoch_end()
        if self.dynamic_gen:
            self.dynamic_gen.on_epoch_end()
```

### Step 4: Update Data Loading Cell

Replace the data loading cell with:

```python
# Load static dataset
print("Loading static dataset...")
static_train_gen = train_datagen.flow_from_directory(
    'isl_dataset/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

static_val_gen = val_datagen.flow_from_directory(
    'isl_dataset/val',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

static_test_gen = test_datagen.flow_from_directory(
    'isl_dataset/test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

static_classes = sorted(static_train_gen.class_indices.keys())
print(f"Static dataset: {len(static_classes)} classes, {static_train_gen.n} train samples")

# Load dynamic dataset
print("\nLoading dynamic dataset...")
dynamic_train_gen = train_datagen.flow_from_directory(
    'processed_dynamic_dataset/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

dynamic_val_gen = val_datagen.flow_from_directory(
    'processed_dynamic_dataset/val',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

dynamic_test_gen = test_datagen.flow_from_directory(
    'processed_dynamic_dataset/test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

dynamic_classes = sorted(dynamic_train_gen.class_indices.keys())
print(f"Dynamic dataset: {len(dynamic_classes)} classes, {dynamic_train_gen.n} train samples")

# Create combined class names
all_classes = static_classes + dynamic_classes
print(f"\nTotal classes: {len(all_classes)}")
print(f"Classes: {all_classes}")

# Create hybrid generators
train_gen = HybridDataGenerator(static_train_gen, dynamic_train_gen, static_classes, dynamic_classes)
val_gen = HybridDataGenerator(static_val_gen, dynamic_val_gen, static_classes, dynamic_classes)
test_gen = HybridDataGenerator(static_test_gen, dynamic_test_gen, static_classes, dynamic_classes)

print(f"\nHybrid generators created:")
print(f"  Train samples: {train_gen.n}")
print(f"  Val samples: {val_gen.n}")
print(f"  Test samples: {test_gen.n}")
```

### Step 5: Update Model Building Cell

Change the output layer to use `TOTAL_CLASSES`:

```python
# Build model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(TOTAL_CLASSES, activation='softmax')(x)  # Changed to 44 classes

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Model built with {TOTAL_CLASSES} output classes")
model.summary()
```

### Step 6: Update Model Saving

Change the model save path:

```python
# Save model
model.save('models/hybrid_sign_language_model.h5')

# Save model info
model_info = {
    'num_classes': TOTAL_CLASSES,
    'static_classes': STATIC_CLASSES,
    'dynamic_classes': DYNAMIC_CLASSES,
    'class_names': all_classes,
    'image_size': IMAGE_SIZE,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('models/hybrid_model_info.json', 'w') as f:
    json.dump(model_info, indent=2, fp=f)

print("✓ Hybrid model saved!")
```

### Step 7: Add Enhanced Logging

Add this cell after training:

```python
# Save comprehensive training log
import datetime

log_file = f'logs/training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

with open(log_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("HYBRID SIGN LANGUAGE RECOGNITION TRAINING LOG\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Training Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("--- Dataset Information ---\n")
    f.write(f"Static Dataset: {STATIC_DATASET_PATH}\n")
    f.write(f"  Classes: {STATIC_CLASSES} (0-9, A-Z)\n")
    f.write(f"  Train samples: {static_train_gen.n}\n")
    f.write(f"  Val samples: {static_val_gen.n}\n")
    f.write(f"  Test samples: {static_test_gen.n}\n\n")
    
    f.write(f"Dynamic Dataset: {DYNAMIC_DATASET_PATH}\n")
    f.write(f"  Classes: {DYNAMIC_CLASSES} (loud, quiet, happy, sad, Beautiful, Ugly, Deaf, Blind)\n")
    f.write(f"  Train samples: {dynamic_train_gen.n}\n")
    f.write(f"  Val samples: {dynamic_val_gen.n}\n")
    f.write(f"  Test samples: {dynamic_test_gen.n}\n\n")
    
    f.write(f"Total Classes: {TOTAL_CLASSES}\n")
    f.write(f"Total Training Samples: {train_gen.n}\n\n")
    
    f.write("--- Training Configuration ---\n")
    f.write(f"Optimizer: Adam\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Image Size: {IMAGE_SIZE}\n\n")
    
    f.write("--- Training Progress ---\n")
    for epoch in range(len(history.history['loss'])):
        f.write(f"Epoch {epoch+1}/{len(history.history['loss'])}: ")
        f.write(f"loss={history.history['loss'][epoch]:.4f}, ")
        f.write(f"acc={history.history['accuracy'][epoch]:.4f}, ")
        f.write(f"val_loss={history.history['val_loss'][epoch]:.4f}, ")
        f.write(f"val_acc={history.history['val_accuracy'][epoch]:.4f}\n")
    
    f.write("\n--- Final Results ---\n")
    f.write(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}\n")
    f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}\n")
    f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n\n")
    
    f.write("--- Model Saved ---\n")
    f.write(f"Path: models/hybrid_sign_language_model.h5\n")

print(f"✓ Training log saved to: {log_file}")
```

---

## Expected Output

After training, you should see:

### Files Created:
- `processed_dynamic_dataset/` - Extracted video frames
- `models/hybrid_sign_language_model.h5` - Trained model
- `models/hybrid_model_info.json` - Model metadata
- `logs/training_log_YYYYMMDD_HHMMSS.txt` - Detailed training log
- `logs/hybrid_training_history.png` - Training curves
- `logs/confusion_matrix_44classes.png` - Confusion matrix
- `logs/sample_predictions_hybrid.png` - Sample predictions

### Console Output:
```
Training configuration:
  Total classes: 44
  Static classes: 36
  Dynamic classes: 8

Static dataset: 36 classes, 23040 train samples
Dynamic dataset: 8 classes, 1050 train samples

Total classes: 44
Classes: ['0', '1', '2', ..., 'Z', '1. loud', '2. quiet', ...]

Hybrid generators created:
  Train samples: 24090
  Val samples: 6015
  Test samples: 7515
```

---

## Troubleshooting

### Issue: Dynamic dataset not found
**Solution:** Run `python scripts/extract_video_frames.py` first

### Issue: Out of memory
**Solution:** Reduce `BATCH_SIZE` to 16 or 8

### Issue: Training too slow
**Solution:** Ensure GPU is available, or reduce `EPOCHS`

---

## Next Steps

After training:
1. Check `logs/` directory for visualizations
2. Review confusion matrix for class performance
3. Update web interface to use new model
4. Test with real-time recognition
