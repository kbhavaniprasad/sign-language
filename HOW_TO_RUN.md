# 🚀 HOW TO RUN - Step by Step Guide

## ✅ Step 1: Test Your Webcam (No Model Needed!)

This tests if your camera and MediaPipe are working:

```powershell
python scripts\test_webcam.py
```

**What you'll see:**
- Your webcam feed
- Hand landmarks drawn on your hands
- FPS counter
- Press 'q' to quit

---

## ✅ Step 2: Create Sample Dataset

Since you don't have a dataset yet, create a sample one:

```powershell
python scripts\create_sample_dataset.py --classes 10 --images 50
```

**What this does:**
- Creates `dataset/train/` and `dataset/test/` folders
- Generates 10 gesture classes with 50 training images each
- Creates synthetic images for testing

---

## ✅ Step 3: Install Remaining Dependencies

```powershell
# Install TensorFlow and other ML libraries
pip install tensorflow keras scikit-learn pandas matplotlib seaborn

# Install Flask and web libraries
pip install flask flask-cors

# Install translation and TTS
pip install googletrans==4.0.0rc1 gtts pygame

# Install Jupyter for training
pip install jupyter notebook ipykernel

# Install automation libraries
pip install pyautogui pynput pyyaml python-dotenv
```

---

## ✅ Step 4: Train the Model

```powershell
# Start Jupyter Notebook
jupyter notebook train_model.ipynb
```

**In the notebook:**
1. Click "Run All" or press Shift+Enter on each cell
2. Wait for training to complete (~10-30 minutes)
3. Model will be saved as `models/sign_language_model.h5`

---

## ✅ Step 5: Test the Trained Model

```powershell
python scripts\test_model.py
```

**What you'll see:**
- Model accuracy on test set
- Sample predictions with confidence scores

---

## ✅ Step 6: Run Real-Time Recognition

```powershell
python main.py --model models\sign_language_model.h5
```

**Controls:**
- `c` - Clear recognized text
- `s` - Speak current text (if --speak enabled)
- `t` - Translate and speak
- `q` - Quit

---

## 🎯 Quick Commands Reference

### Test Webcam Only
```powershell
python scripts\test_webcam.py
```

### Run with Translation
```powershell
python main.py --model models\sign_language_model.h5 --language es
```

### Run with Speech
```powershell
python main.py --model models\sign_language_model.h5 --speak
```

### Run with Translation AND Speech
```powershell
python main.py --model models\sign_language_model.h5 --language hi --speak
```

### Start API Server
```powershell
python src\api\app.py
```

---

## 🔧 If You Get Errors

### "No module named 'cv2'"
```powershell
pip install opencv-python
```

### "No module named 'mediapipe'"
```powershell
pip install mediapipe
```

### "No module named 'tensorflow'"
```powershell
pip install tensorflow
```

### "Model file not found"
- You need to train the model first using Step 4

### Camera not opening
- Check if another app is using the camera
- Try different camera ID: `--camera 1`

---

## 📝 What Each File Does

- **`scripts\test_webcam.py`** - Tests camera and hand detection (NO MODEL NEEDED)
- **`scripts\create_sample_dataset.py`** - Creates sample training data
- **`train_model.ipynb`** - Trains the deep learning model
- **`scripts\test_model.py`** - Tests the trained model
- **`main.py`** - Main application for real-time recognition
- **`src\api\app.py`** - REST API server

---

## 🎯 Recommended Order

1. ✅ **Test webcam** → Make sure camera works
2. ✅ **Create dataset** → Get training data
3. ✅ **Install dependencies** → Get all libraries
4. ✅ **Train model** → Create the .h5 file
5. ✅ **Test model** → Verify it works
6. ✅ **Run application** → Use it in real-time!

---

## 💡 Pro Tips

- **Start with webcam test** - It doesn't need a trained model
- **Use sample dataset** - Quick way to test the system
- **Train on real data** - Download ASL dataset from Kaggle for better results
- **Check logs/** - If something fails, check the log files

---

**Ready to start? Run this first:**
```powershell
python scripts\test_webcam.py
```
