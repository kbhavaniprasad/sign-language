# Real-Time Sign Language to Speech System

A comprehensive system that converts sign language gestures into multilingual speech in real-time, with advanced eye control features for hands-free PC operation.

## 🌟 Features

### Core Features
- **Real-Time Gesture Recognition**: Recognizes sign language gestures using webcam
- **Deep Learning Model**: MobileNetV2 + LSTM architecture for accurate recognition
- **MediaPipe Integration**: Hand landmark detection for robust gesture tracking
- **Multilingual Translation**: Translate recognized text to 20+ languages
- **Text-to-Speech**: Convert recognized gestures to speech in multiple languages
- **Gesture Buffering**: Temporal smoothing for stable predictions

### Advanced Features
- **Eye Control System**: Control your PC using eye movements and blinks
- **Gaze Tracking**: Track eye gaze direction (left, right, up, down, center)
- **Blink Detection**: Detect blinks for click actions
- **Head Pose Estimation**: Track head movements for enhanced control
- **REST API**: Flask-based API for integration with other applications

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow, Keras, PyTorch
- **Computer Vision**: OpenCV, MediaPipe
- **Translation**: Google Translate API (googletrans)
- **Text-to-Speech**: gTTS, Coqui TTS
- **API**: Flask, Flask-CORS
- **PC Control**: PyAutoGUI, Pynput
- **Data Science**: NumPy, Pandas, Scikit-learn

## 📋 Prerequisites

- Python 3.10.2
- Webcam
- Windows OS (for eye control features)
- 4GB+ RAM recommended

## 🚀 Installation

### 1. Clone or Download the Repository

```bash
cd d:\sign
```

### 2. Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Prepare Dataset

Place your sign language dataset in the following structure:

```
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

## 🎓 Training the Model

### Option 1: Using Jupyter Notebook (Recommended)

```powershell
jupyter notebook train_model.ipynb
```

Run all cells in the notebook to:
- Load and preprocess the dataset
- Build the MobileNetV2 + LSTM model
- Train with data augmentation
- Evaluate performance
- Save the model as `models/sign_language_model.h5`

### Option 2: Using Python Script

```powershell
python src/training/train_model.py
```

The trained model will be saved to `models/sign_language_model.h5`.

## 🎯 Usage

### 1. Test Webcam and MediaPipe

```powershell
python scripts/test_webcam.py
```

This will open your webcam and display hand landmarks in real-time.

### 2. Test Trained Model

```powershell
python scripts/test_model.py --model models/sign_language_model.h5
```

### 3. Run Real-Time Recognition

```powershell
python main.py --model models/sign_language_model.h5 --camera 0
```

**Controls:**
- `c` - Clear recognized text
- `s` - Speak current text
- `t` - Translate and speak
- `q` - Quit

### 4. Run with Translation and TTS

```powershell
python main.py --model models/sign_language_model.h5 --language es --speak
```

Supported languages: `en`, `es`, `fr`, `de`, `hi`, `zh-cn`, `ja`, `ar`, `ru`, `pt`, etc.

### 5. Start API Server

```powershell
python src/api/app.py
```

The API will be available at `http://localhost:5000`

## 📡 API Endpoints

### Health Check
```
GET /health
```

### Get Supported Languages
```
GET /languages
```

### Set Target Language
```
POST /language
Body: {"language": "es"}
```

### Predict Gesture
```
POST /predict
Body: {"image": "base64_encoded_image"}
```

### Translate Text
```
POST /translate
Body: {"text": "hello", "target_language": "es"}
```

### Text-to-Speech
```
POST /speak
Body: {"text": "hello", "language": "en"}
```

### Eye Control Status
```
GET /eye-control/status
```

### Toggle Eye Control
```
POST /eye-control/toggle
Body: {"enable": true}
```

## 🎮 Eye Control System

The eye control system allows hands-free PC operation using eye movements and blinks.

### Features:
- **Gaze-based cursor movement**
- **Blink-based clicking**
- **Scroll gestures**
- **Adjustable sensitivity**
- **Calibration support**

### Usage:
1. Enable eye control in the application
2. Perform calibration (optional)
3. Use gaze to move cursor
4. Blink to click
5. Look up/down and dwell to scroll

## 📁 Project Structure

```
d:\sign\
├── config/
│   └── config.yaml          # Configuration file
├── dataset/                 # Dataset directory
│   ├── train/
│   └── test/
├── models/                  # Trained models
│   ├── sign_language_model.h5
│   └── model_info.json
├── src/
│   ├── api/                # Flask API
│   ├── data/               # Data loading and preprocessing
│   ├── eye_control/        # Eye tracking and PC control
│   ├── models/             # Model architectures
│   ├── recognition/        # Gesture recognition
│   ├── training/           # Training scripts
│   ├── translation/        # Translation services
│   ├── tts/                # Text-to-speech
│   └── utils/              # Utilities
├── scripts/                # Test scripts
├── logs/                   # Log files
├── temp/                   # Temporary files
├── train_model.ipynb       # Training notebook
├── main.py                 # Main application
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Model paths
- Training parameters
- Recognition settings
- Translation languages
- TTS settings
- Eye control sensitivity
- API settings

## 🐛 Troubleshooting

### Camera not opening
- Check camera permissions
- Try different camera ID: `--camera 1`
- Ensure no other application is using the camera

### Model not found
- Train the model first using the Jupyter notebook
- Check the model path in config.yaml

### Low FPS
- Reduce image size in config.yaml
- Disable eye control if not needed
- Use GPU if available

### Translation not working
- Check internet connection (googletrans requires internet)
- Try alternative translation service

## 📊 Model Performance

The model achieves:
- **Accuracy**: Varies based on dataset (typically 85-95%)
- **FPS**: 15-30 FPS on standard hardware
- **Latency**: <100ms for real-time recognition

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- **MediaPipe**: Google's MediaPipe for hand and face tracking
- **TensorFlow**: Deep learning framework
- **OpenCV**: Computer vision library
- **Google Translate**: Translation services
- **gTTS**: Text-to-speech synthesis

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Review the troubleshooting section

## 🎯 Future Enhancements

- [ ] Support for more sign languages
- [ ] Mobile app (React Native)
- [ ] Real-time collaboration features
- [ ] Cloud deployment
- [ ] Improved eye control accuracy
- [ ] Custom gesture training
- [ ] Video recording and playback

---

**Made with ❤️ for the hearing-impaired community**
