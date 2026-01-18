# 🎯 ISL Dataset - Training Guide

## ✅ Your Dataset

- **Classes**: 36 (Indian Sign Language gestures)
- **Training images**: 800 per class = 28,800 total
- **Test images**: 200 per class = 7,200 total
- **Total**: 36,000 images

This is a **professional-quality dataset** that will produce excellent results!

---

## 📊 Dataset Organization

The script is organizing your dataset into:

```
dataset/
├── train/
│   ├── 0/ (800 images)
│   ├── 1/ (800 images)
│   ├── 2/ (800 images)
│   └── ... (36 classes total)
└── test/
    ├── 0/ (200 images)
    ├── 1/ (200 images)
    ├── 2/ (200 images)
    └── ... (36 classes total)
```

---

## 🚀 Training Configuration

I've updated the configuration for your dataset:

- **Classes**: 36 (updated from 10)
- **Epochs**: 100 (increased for better learning)
- **Early stopping patience**: 15 epochs
- **Batch size**: 32
- **Image size**: 224x224

---

## ⏱️ Expected Training Time

With 36,000 images:
- **CPU**: ~2-4 hours
- **GPU**: ~30-60 minutes

The model will auto-save the best version and stop early if no improvement.

---

## 🎓 Start Training

Once organization is complete, run:

```powershell
python -m jupyter notebook train_model.ipynb
```

**In the notebook:**
1. Click **"Cell" → "Run All"**
2. Wait for training to complete
3. Model saves as `models/sign_language_model.h5`

---

## 📈 What to Expect

With this large dataset, you should achieve:
- **Training accuracy**: 95-99%
- **Test accuracy**: 90-95%
- **Real-time FPS**: 15-25 FPS

---

## 💡 Tips for Better Results

1. **Let it train fully** - Don't stop early
2. **Monitor the graphs** - Check for overfitting
3. **Use data augmentation** - Already enabled in the notebook
4. **Fine-tune if needed** - Can adjust learning rate later

---

## 🎯 After Training

```powershell
# Test the model
python scripts\test_model.py

# Run real-time recognition
python main.py --model models\sign_language_model.h5

# With Hindi translation and speech
python main.py --model models\sign_language_model.h5 --language hi --speak
```

---

**Your ISL dataset is being organized now. Training will start soon!** 🚀
