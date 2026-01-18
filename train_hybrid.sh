#!/bin/bash
# Quick Start Script for Hybrid Model Training (Linux/Mac)
# This script extracts video frames and trains the hybrid model

echo "========================================"
echo "Hybrid Sign Language Model Training"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run: python -m venv venv"
    echo "Then: source venv/bin/activate"
    echo "Then: pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "========================================"
echo "Step 1: Extract Frames from Videos"
echo "========================================"
echo ""

# Check if dynamic dataset already exists
if [ -d "processed_dynamic_dataset" ]; then
    echo "Dynamic dataset already exists."
    read -p "Re-extract frames? (y/n): " REEXTRACT
    if [ "$REEXTRACT" = "y" ] || [ "$REEXTRACT" = "Y" ]; then
        echo "Extracting frames..."
        python scripts/extract_video_frames.py
    else
        echo "Skipping frame extraction."
    fi
else
    echo "Extracting frames from videos..."
    python scripts/extract_video_frames.py
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Frame extraction failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Step 2: Train Hybrid Model"
echo "========================================"
echo ""

python train_hybrid_model.py

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "Check the following:"
echo "  - Model: models/hybrid_sign_language_model.h5"
echo "  - Logs: logs/ directory"
echo "  - Visualizations: logs/*.png"
echo ""
echo "To use the model with the web interface:"
echo "  1. Update web/api_server.py to load hybrid model"
echo "  2. Run: python web/api_server.py"
echo ""
