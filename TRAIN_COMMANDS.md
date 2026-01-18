# 🚀 COMMANDS TO TRAIN THE MODEL

## Option 1: Using Jupyter Notebook (Recommended - Visual Interface)

```powershell
python -m jupyter notebook train_model.ipynb
```

**What happens:**
- Opens Jupyter in your browser
- Shows the training notebook
- Click **"Cell" → "Run All"** to start training
- Watch the progress in real-time

---

## Option 2: Run Jupyter and Open Manually

```powershell
# Start Jupyter
python -m jupyter notebook

# Then in the browser, click on "train_model.ipynb"
```

---

## Option 3: Run from Command Line (No Browser)

```powershell
# Convert notebook to Python script and run
python -m jupyter nbconvert --to script train_model.ipynb
python train_model.py
```

---

## ⚡ FASTEST WAY (Copy and Paste This):

```powershell
python -m jupyter notebook train_model.ipynb
```

**Then in the browser that opens:**
1. Click **"Cell"** in the menu bar
2. Click **"Run All"**
3. Wait for training to complete (~2-4 hours)

---

## 🔍 If Jupyter Doesn't Open Automatically

The command will show a URL like:
```
http://localhost:8888/?token=abc123...
```

Copy that URL and paste it into your browser.

---

## 📊 What You'll See

The notebook will show:
- Dataset loading (36 classes, 36,000 images)
- Model architecture summary
- Training progress bar
- Accuracy and loss graphs
- Confusion matrix
- Test predictions

---

## ⏹️ To Stop Training

- Press **"Kernel" → "Interrupt"** in Jupyter
- Or press **Ctrl+C** twice in the terminal

---

## ✅ After Training Completes

The model will be saved as:
```
models/sign_language_model.h5
```

Then you can test it:
```powershell
python scripts\test_model.py
python main.py --model models\sign_language_model.h5
```

---

**Just run this command now:**
```powershell
python -m jupyter notebook train_model.ipynb
```
