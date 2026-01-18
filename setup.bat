@echo off
REM Setup script for Sign Language Recognition System

echo ========================================
echo Sign Language Recognition System Setup
echo ========================================
echo.

REM Check Python version
python --version
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.10.2
    pause
    exit /b 1
)

echo.
echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 4: Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Step 5: Creating necessary directories...
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "temp" mkdir temp

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Place your dataset in the 'dataset' folder
echo    OR run: python scripts\create_sample_dataset.py
echo.
echo 2. Train the model:
echo    jupyter notebook train_model.ipynb
echo.
echo 3. Test webcam:
echo    python scripts\test_webcam.py
echo.
echo 4. Run the application:
echo    python main.py --model models\sign_language_model.h5
echo.
pause
