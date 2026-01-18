@echo off
REM Quick Start Script for Hybrid Model Training
REM This script extracts video frames and trains the hybrid model

echo ========================================
echo Hybrid Sign Language Model Training
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: .\venv\Scripts\activate
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo ========================================
echo Step 1: Extract Frames from Videos
echo ========================================
echo.

REM Check if dynamic dataset already exists
if exist "processed_dynamic_dataset" (
    echo Dynamic dataset already exists.
    set /p REEXTRACT="Re-extract frames? (y/n): "
    if /i "%REEXTRACT%"=="y" (
        echo Extracting frames...
        python scripts\extract_video_frames.py
    ) else (
        echo Skipping frame extraction.
    )
) else (
    echo Extracting frames from videos...
    python scripts\extract_video_frames.py
)

if errorlevel 1 (
    echo ERROR: Frame extraction failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Step 2: Train Hybrid Model
echo ========================================
echo.

python train_hybrid_model.py

if errorlevel 1 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Check the following:
echo   - Model: models\hybrid_sign_language_model.h5
echo   - Logs: logs\ directory
echo   - Visualizations: logs\*.png
echo.
echo To use the model with the web interface:
echo   1. Update web\api_server.py to load hybrid model
echo   2. Run: python web\api_server.py
echo.

pause
