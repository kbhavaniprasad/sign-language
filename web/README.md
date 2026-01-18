# Real-Time Sign Language Recognition Web Interface

A modern, real-time web interface for sign language recognition using your trained model.

## 🌟 Features

- **Live Camera Feed**: Real-time video capture from your webcam
- **Instant Predictions**: Sign language recognition at 5 FPS
- **Confidence Scores**: Visual confidence indicators for each prediction
- **Top 3 Predictions**: See alternative predictions with confidence levels
- **Prediction History**: Track recent predictions with timestamps
- **Modern UI**: Premium dark theme with glassmorphism and smooth animations
- **Responsive Design**: Works on desktop and mobile devices

## 📋 Prerequisites

- Python 3.10.2 with virtual environment activated
- Trained model: `models/sign_language_model.h5`
- Modern web browser (Chrome, Firefox, Edge)
- Webcam

## 🚀 Quick Start

### 1. Start the Backend API

Open a terminal and run:

```powershell
cd d:\sign
.\venv\Scripts\activate
python src/api/app.py
```

You should see:
```
INFO - Loading model from models/sign_language_model.h5
INFO - Model loaded successfully
INFO - Starting API server on 0.0.0.0:5000
```

### 2. Open the Web Interface

Open `d:\sign\web\index.html` in your web browser.

**Or** use a local server (recommended):

```powershell
# In a new terminal
cd d:\sign\web
python -m http.server 8000
```

Then open: `http://localhost:8000`

### 3. Start Recognition

1. Click **"Start Camera"** button
2. Grant camera permissions when prompted
3. Show sign language gestures (0-9, A-Z) to the camera
4. Watch real-time predictions appear!

## 🎮 Controls

- **Start Camera**: Begin capturing video and making predictions
- **Stop Camera**: Stop the camera and predictions
- **Clear History**: Clear the prediction history

## 📊 Interface Sections

### Camera Feed
- Live video from your webcam
- Real-time processing at 5 FPS
- Visual feedback when active

### Current Prediction
- Large display of the predicted sign
- Confidence bar showing prediction certainty
- Percentage confidence score

### Top 3 Predictions
- Shows the top 3 most likely predictions
- Confidence scores for each
- Helps understand model uncertainty

### Recent Predictions
- History of recent high-confidence predictions (>70%)
- Timestamps for each prediction
- Automatically limited to last 10 predictions

## 🔧 Configuration

### Adjust Frame Rate

Edit `web/app.js`:
```javascript
const CAPTURE_FPS = 5; // Change to desired FPS (1-10 recommended)
```

### Change API URL

If running the API on a different host/port:
```javascript
const API_URL = 'http://localhost:5000'; // Update as needed
```

### Adjust History Size

```javascript
const MAX_HISTORY = 10; // Change maximum history items
```

## 🌐 API Endpoints Used

### Health Check
```
GET http://localhost:5000/health
```
Checks if the API is running.

### Predict Frame
```
POST http://localhost:5000/predict-frame
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,..."
}
```

**Response:**
```json
{
  "success": true,
  "prediction": "A",
  "confidence": 0.9876,
  "top_3": [
    {"class": "A", "confidence": 0.9876},
    {"class": "B", "confidence": 0.0098},
    {"class": "C", "confidence": 0.0015}
  ]
}
```

## 🎨 Design Features

- **Dark Theme**: Easy on the eyes for extended use
- **Glassmorphism**: Frosted glass effect on cards
- **Gradient Accents**: Purple-to-blue gradients
- **Smooth Animations**: Micro-interactions for better UX
- **Responsive Layout**: Adapts to different screen sizes
- **Custom Fonts**: Inter font family for modern look

## 🐛 Troubleshooting

### Camera Not Working
- **Check permissions**: Ensure browser has camera access
- **Try different browser**: Chrome and Firefox work best
- **Check other apps**: Close other apps using the camera
- **HTTPS required**: Some browsers require HTTPS for camera access

### API Connection Failed
- **Check backend**: Ensure Flask API is running
- **Check URL**: Verify API_URL in `app.js` matches your setup
- **Check CORS**: API has CORS enabled by default
- **Check firewall**: Ensure port 5000 is not blocked

### Low FPS / Lag
- **Reduce FPS**: Lower CAPTURE_FPS in `app.js`
- **Check CPU**: Model inference requires processing power
- **Close other apps**: Free up system resources
- **Use better lighting**: Helps with camera quality

### Predictions Not Accurate
- **Lighting**: Ensure good, even lighting
- **Hand position**: Keep hand clearly visible
- **Distance**: Maintain consistent distance from camera
- **Background**: Use plain background when possible
- **Hold gesture**: Hold each sign for 1-2 seconds

## 📱 Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome  | ✅ Full | Recommended |
| Firefox | ✅ Full | Recommended |
| Edge    | ✅ Full | Works well |
| Safari  | ⚠️ Partial | May require HTTPS |
| Opera   | ✅ Full | Works well |

## 🔒 Privacy & Security

- **Local Processing**: All predictions happen on your local machine
- **No Data Storage**: Video frames are not stored
- **No External Calls**: Only communicates with local API
- **Camera Access**: Only active when you click "Start Camera"

## 📈 Performance Tips

1. **Use good lighting** for better camera quality
2. **Keep hand in frame** and clearly visible
3. **Hold gestures steady** for 1-2 seconds
4. **Use plain background** to reduce noise
5. **Close unnecessary apps** to free resources

## 🎯 Model Information

- **Classes**: 36 (0-9, A-Z)
- **Accuracy**: 98.96%
- **Input Size**: 224x224 pixels
- **Model Type**: MobileNetV2-based CNN
- **Model File**: `models/sign_language_model.h5`

## 📝 Technical Details

### Frame Processing Pipeline

1. **Capture**: Video frame captured from webcam
2. **Encode**: Frame converted to JPEG base64
3. **Send**: Sent to Flask API via POST request
4. **Preprocess**: Resized to 224x224, normalized
5. **Predict**: Model inference
6. **Return**: Top predictions sent back
7. **Display**: UI updated with results

### Performance Metrics

- **Latency**: ~100-200ms per prediction
- **FPS**: 5 frames per second (configurable)
- **Bandwidth**: ~50-100 KB per frame
- **Memory**: Minimal (history limited to 10 items)

---

**Enjoy real-time sign language recognition! 🤟**
