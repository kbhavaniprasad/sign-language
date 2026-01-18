# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Setup Environment

```powershell
# Run the setup script
.\setup.bat
```

This will:
- Create a virtual environment
- Install all dependencies
- Create necessary directories

### 2. Create Sample Dataset (Optional)

If you don't have a dataset yet:

```powershell
python scripts\create_sample_dataset.py --classes 5 --images 20
```

Or download a real dataset:
- [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

### 3. Train the Model

Open the Jupyter notebook:

```powershell
jupyter notebook train_model.ipynb
```

Run all cells to train the model. This will:
- Load your dataset
- Train the MobileNetV2 + LSTM model
- Save the model as `models/sign_language_model.h5`

Training time: ~10-30 minutes depending on dataset size and hardware.

### 4. Test Your Setup

Test webcam and MediaPipe:

```powershell
python scripts\test_webcam.py
```

You should see your webcam feed with hand landmarks.

### 5. Run Real-Time Recognition

```powershell
python main.py --model models\sign_language_model.h5
```

**Controls:**
- `c` - Clear text
- `s` - Speak text
- `t` - Translate and speak
- `q` - Quit

## 🎯 Common Use Cases

### Use Case 1: Basic Recognition

```powershell
python main.py --model models\sign_language_model.h5 --camera 0
```

### Use Case 2: With Translation

```powershell
python main.py --model models\sign_language_model.h5 --language es --speak
```

### Use Case 3: API Server

```powershell
python src\api\app.py
```

Then access the API at `http://localhost:5000`

## 🔧 Troubleshooting

### Issue: Camera not working
**Solution:** Try different camera ID
```powershell
python main.py --camera 1
```

### Issue: Model not found
**Solution:** Train the model first
```powershell
jupyter notebook train_model.ipynb
```

### Issue: Import errors
**Solution:** Reinstall dependencies
```powershell
pip install -r requirements.txt --force-reinstall
```

### Issue: Low FPS
**Solution:** Reduce image size in `config/config.yaml`

## 📚 Next Steps

1. **Improve Model**: Train with more data
2. **Add Gestures**: Expand your dataset
3. **Customize**: Edit `config/config.yaml`
4. **Deploy**: Use the Flask API for integration

## 🆘 Need Help?

- Check the main [README.md](README.md)
- Review the [API documentation](docs/API.md)
- Open an issue on GitHub

---

**Happy Coding! 🤟**
