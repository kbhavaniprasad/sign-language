# Hybrid Model Training - Quick Reference

## Quick Start (3 Steps)

### Windows
```bash
# 1. Activate environment
.\venv\Scripts\activate

# 2. Run automated training
.\train_hybrid.bat
```

### Linux/Mac
```bash
# 1. Activate environment
source venv/bin/activate

# 2. Run automated training
chmod +x train_hybrid.sh
./train_hybrid.sh
```

---

## Manual Steps

### Step 1: Extract Video Frames
```bash
python scripts/extract_video_frames.py
```
**Output**: `processed_dynamic_dataset/` with train/val/test splits

### Step 2: Train Hybrid Model
```bash
python train_hybrid_model.py
```
**Output**: 
- Model: `models/hybrid_sign_language_model.h5`
- Logs: `logs/training_log_*.txt`
- Visualizations: `logs/*.png`

---

## What Gets Trained

### Static Signs (36 classes)
```
0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
A, B, C, D, E, F, G, H, I, J, K, L, M,
N, O, P, Q, R, S, T, U, V, W, X, Y, Z
```

### Dynamic Gestures (8 classes)
```
loud, quiet, happy, sad,
Beautiful, Ugly, Deaf, Blind
```

### Total: 44 Classes

---

## Expected Results

### Training Time
- **GPU**: 4-6 hours
- **CPU**: 12-24 hours

### Accuracy
- **Static Signs**: 95-99%
- **Dynamic Gestures**: 80-90%
- **Overall**: 90-95%

### Files Generated
```
models/
├── hybrid_sign_language_model.h5     (~100 MB)
└── hybrid_model_info.json            (~2 KB)

logs/
├── training_log_YYYYMMDD_HHMMSS.txt  (~50 KB)
├── training_history_*.json           (~10 KB)
├── hybrid_training_history.png       (training curves)
├── confusion_matrix_44classes.png    (44×44 matrix)
└── sample_predictions_hybrid.png     (sample results)

processed_dynamic_dataset/
├── train/     (~728 frames)
├── val/       (~156 frames)
├── test/      (~156 frames)
└── extraction_stats.json
```

---

## Configuration

### Modify Training Parameters
Edit `train_hybrid_model.py`:
```python
class Config:
    BATCH_SIZE = 32        # Reduce if out of memory
    EPOCHS = 50            # Reduce for faster testing
    LEARNING_RATE = 0.001  # Adjust learning rate
```

### Modify Frame Extraction
Edit `scripts/extract_video_frames.py`:
```python
extractor = VideoFrameExtractor(
    frames_per_video=10,   # More frames = more data
    image_size=(224, 224)  # Must match training size
)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `BATCH_SIZE` to 16 or 8 |
| Training too slow | Reduce `EPOCHS` or use GPU |
| Dynamic dataset not found | Run `extract_video_frames.py` first |
| Low dynamic accuracy | Increase `frames_per_video` or collect more videos |

---

## Using the Trained Model

### Option 1: Update Web Interface
Edit `web/api_server.py`:
```python
MODEL_PATH = 'models/hybrid_sign_language_model.h5'
MODEL_INFO_PATH = 'models/hybrid_model_info.json'
```

Then run:
```bash
python web/api_server.py
```

### Option 2: Load in Python
```python
from tensorflow import keras
import json

# Load model
model = keras.models.load_model('models/hybrid_sign_language_model.h5')

# Load class names
with open('models/hybrid_model_info.json', 'r') as f:
    info = json.load(f)
    class_names = info['class_names']

# Make prediction
prediction = model.predict(image)
predicted_class = class_names[prediction.argmax()]
```

---

## Notebook Integration

For Jupyter notebook users, see: [`HYBRID_NOTEBOOK_GUIDE.md`](file:///d:/sign/HYBRID_NOTEBOOK_GUIDE.md)

---

## File Locations

| File | Purpose |
|------|---------|
| `scripts/extract_video_frames.py` | Video frame extraction |
| `train_hybrid_model.py` | Main training script |
| `train_hybrid.bat` | Windows automation |
| `train_hybrid.sh` | Linux/Mac automation |
| `HYBRID_NOTEBOOK_GUIDE.md` | Notebook integration |
| `README.md` | Full documentation |

---

## Command Cheat Sheet

```bash
# Extract frames
python scripts/extract_video_frames.py

# Train model
python train_hybrid_model.py

# Automated (Windows)
.\train_hybrid.bat

# Automated (Linux/Mac)
./train_hybrid.sh

# Test model
python scripts/test_model.py --model models/hybrid_sign_language_model.h5

# Run web interface
python web/api_server.py
```

---

## Next Steps After Training

1. ✅ Check `logs/` for visualizations
2. ✅ Review confusion matrix
3. ✅ Verify accuracy in training log
4. ✅ Update web interface (optional)
5. ✅ Test real-time recognition

---

**Need Help?** See [`walkthrough.md`](file:///C:/Users/k%20bhavaniprasad/.gemini/antigravity/brain/c2e7bcc9-8f70-4274-8816-a15173f00c9e/walkthrough.md) for detailed documentation.
