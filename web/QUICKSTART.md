# 🚀 Quick Start Guide - Sign Language Recognition Web Interface

## Step-by-Step Instructions

### Step 1: Start the Backend API Server

Open a terminal and run:

```powershell
cd d:\sign
python web\api_server.py
```

**Or simply double-click:** `d:\sign\web\start_server.bat`

**You should see:**
```
============================================================
Starting Sign Language Recognition API
Server: http://0.0.0.0:5000
Model: Loaded
Classes: 36
============================================================

📷 Web interface will open automatically in your browser...

🌐 Opening web interface in browser...
```

✅ **The web interface will automatically open in your default browser!**

✅ **Keep this terminal open!** The server must stay running.

---

### Step 2: Grant Camera Permissions

When the browser opens:

1. **Click "Start Camera"** button
2. **Allow camera permissions** when prompted
3. **Show sign language gestures** (0-9, A-Z) to the camera
4. **Watch predictions appear in real-time!**

---

### Step 3: Start Recognition

## 🎯 What You'll See

### Backend Terminal
```
Loading model from models/sign_language_model.h5...
Model loaded successfully!
Loaded 36 classes: ['0', '1', '2', '3', ..., 'Z']
 * Running on http://127.0.0.1:5000
```

### Web Interface
- **Header**: Shows connection status and FPS
- **Camera Feed**: Live video from your webcam
- **Current Prediction**: Large display of predicted sign
- **Confidence Bar**: Visual confidence indicator
- **Top 3 Predictions**: Alternative predictions
- **Recent Predictions**: History of recent signs

---

## ⚠️ Important Notes

### DO NOT Navigate To:
- ❌ `http://localhost:5000` (This is the API, not the interface!)
- ❌ `http://127.0.0.1:5000` (This is the API, not the interface!)

### DO Navigate To:
- ✅ `file:///d:/sign/web/index.html` (This is the web interface!)

---

## 🔧 Troubleshooting

### "Not Found" Error
**Problem**: You're trying to access `http://localhost:5000` directly  
**Solution**: Open `file:///d:/sign/web/index.html` instead

### "Disconnected" Status
**Problem**: API server is not running  
**Solution**: Start the backend server (Step 1)

### Camera Not Working
**Problem**: Browser doesn't have camera permissions  
**Solution**: Click "Allow" when prompted, or check browser settings

### No Predictions
**Problem**: Model not loaded or API connection failed  
**Solution**: 
1. Check terminal - model should be "Loaded"
2. Check browser console (F12) for errors
3. Ensure connection status shows "Connected"

---

## 🎬 Complete Workflow

```
┌─────────────────────┐
│  Start API Server   │  python web\api_server.py
│  (Terminal)         │  Loads sign_language_model.h5
└──────────┬──────────┘
           │
           │ Model Ready
           ▼
┌─────────────────────┐
│  Open HTML File     │  file:///d:/sign/web/index.html
│  (Browser)          │  
└──────────┬──────────┘
           │
           │ Click "Start Camera"
           ▼
┌─────────────────────┐
│  Camera Captures    │  5 frames per second
│  Hand Signs         │  
└──────────┬──────────┘
           │
           │ Send frame (base64)
           ▼
┌─────────────────────┐
│  API Processes      │  Resize → Normalize → Predict
│  Frame              │  
└──────────┬──────────┘
           │
           │ Return prediction + confidence
           ▼
┌─────────────────────┐
│  Display Results    │  Show prediction, confidence, history
│  in Browser         │  
└─────────────────────┘
```

---

## 🎤 Text-to-Speech (Future Feature)

The current simplified version focuses on real-time prediction. For text-to-speech with multiple languages using gTTS, you would need to:

1. Install additional dependencies: `pip install gtts googletrans==4.0.0rc1`
2. Use the full API: `python src/api/app.py` (instead of `web/api_server.py`)
3. Add translation and speech buttons to the interface

For now, the interface shows predictions in real-time without speech output.

---

**Ready to test? Follow Steps 1-3 above! 🤟**
