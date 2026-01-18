# Setup Instructions for Windows PowerShell

## ⚠️ Important: PowerShell Syntax

In PowerShell, you need to use `.\` before batch files:

```powershell
# ❌ WRONG
setup.bat

# ✅ CORRECT
.\setup.bat
```

## 🚀 Quick Setup Steps

### Step 1: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

This will install:
- TensorFlow 2.10.0
- OpenCV
- MediaPipe
- Flask
- gTTS
- PyAutoGUI
- And all other dependencies

**Note**: Installation may take 5-10 minutes depending on your internet speed.

### Step 2: Create Directories

```powershell
# Create necessary directories
New-Item -ItemType Directory -Force -Path models
New-Item -ItemType Directory -Force -Path logs
New-Item -ItemType Directory -Force -Path temp
New-Item -ItemType Directory -Force -Path dataset
```

### Step 3: Create Sample Dataset (Optional)

```powershell
python scripts\create_sample_dataset.py --classes 10 --images 50
```

Or download a real dataset from Kaggle and place it in the `dataset` folder.

### Step 4: Train the Model

```powershell
# Start Jupyter Notebook
jupyter notebook train_model.ipynb
```

Then run all cells in the notebook to train your model.

### Step 5: Test Your Setup

```powershell
# Test webcam and MediaPipe
python scripts\test_webcam.py

# Test the trained model (after training)
python scripts\test_model.py
```

### Step 6: Run the Application

```powershell
# Basic recognition
python main.py --model models\sign_language_model.h5

# With translation and speech
python main.py --model models\sign_language_model.h5 --language es --speak

# Start API server
python src\api\app.py
```

## 🔧 Troubleshooting

### Issue: "pip is not recognized"
**Solution**: Add Python to PATH or use full path:
```powershell
python -m pip install -r requirements.txt
```

### Issue: TensorFlow installation fails
**Solution**: Try installing compatible version:
```powershell
pip install tensorflow==2.10.0 --no-cache-dir
```

### Issue: MediaPipe installation fails
**Solution**: Install Visual C++ Redistributable from Microsoft, then retry.

### Issue: Permission denied
**Solution**: Run PowerShell as Administrator.

## 📦 What Gets Installed

- **TensorFlow 2.10.0** - Deep learning framework
- **Keras 2.10.0** - High-level neural networks API
- **OpenCV 4.8.1** - Computer vision library
- **MediaPipe 0.10.8** - Hand and face tracking
- **Flask 3.0.0** - Web framework for API
- **gTTS 2.4.0** - Text-to-speech
- **PyAutoGUI 0.9.54** - PC automation
- **And more...**

Total download size: ~500MB - 1GB

## ✅ Verify Installation

```powershell
# Check Python version
python --version

# Check if TensorFlow is installed
python -c "import tensorflow as tf; print(tf.__version__)"

# Check if OpenCV is installed
python -c "import cv2; print(cv2.__version__)"

# Check if MediaPipe is installed
python -c "import mediapipe as mp; print('MediaPipe OK')"
```

## 🎯 Next Steps After Installation

1. **Create or download dataset** → Place in `dataset/train/` and `dataset/test/`
2. **Train model** → Run `train_model.ipynb`
3. **Test webcam** → Run `python scripts\test_webcam.py`
4. **Run application** → Run `python main.py --model models\sign_language_model.h5`

---

**Need help?** Check the main README.md or QUICKSTART.md
