"""
Test Trained Model
"""
import os
import sys
import numpy as np
import cv2
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow import keras
from src.data.data_loader import SignLanguageDataLoader

def test_model(model_path='models/sign_language_model.h5', 
               dataset_path='dataset',
               num_samples=10):
    """Test trained model on sample images"""
    
    print("="*60)
    print("Testing Trained Model")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        print("\nPlease train the model first using:")
        print("  jupyter notebook train_model.ipynb")
        return
    
    # Load model
    print(f"\n📦 Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully")
    
    # Load model info
    model_info_path = 'models/model_info.json'
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
            class_names = model_info['class_names']
            print(f"\n📊 Number of classes: {len(class_names)}")
            print(f"📝 Classes: {', '.join(class_names)}")
    else:
        print("\n⚠️  Model info not found, using default class names")
        class_names = [f"Class_{i}" for i in range(model.output_shape[-1])]
    
    # Load test data
    print(f"\n📂 Loading test data from {dataset_path}...")
    data_loader = SignLanguageDataLoader(dataset_path=dataset_path)
    
    try:
        X_test, y_test, _ = data_loader.load_dataset('test')
        print(f"✅ Loaded {len(X_test)} test images")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Test on random samples
    print(f"\n🧪 Testing on {num_samples} random samples...")
    print("="*60)
    
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    
    correct = 0
    
    for i, idx in enumerate(indices):
        image = X_test[idx]
        true_label = y_test[idx]
        
        # Predict
        pred = model.predict(np.expand_dims(image, axis=0), verbose=0)
        pred_label = np.argmax(pred[0])
        confidence = pred[0][pred_label]
        
        # Check if correct
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        
        # Print result
        status = "✅" if is_correct else "❌"
        print(f"\n{status} Sample {i+1}:")
        print(f"   True: {class_names[true_label]}")
        print(f"   Predicted: {class_names[pred_label]} (confidence: {confidence*100:.2f}%)")
    
    # Print accuracy
    accuracy = (correct / len(indices)) * 100
    print("\n" + "="*60)
    print(f"📊 Test Accuracy: {correct}/{len(indices)} ({accuracy:.2f}%)")
    print("="*60)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained model')
    parser.add_argument('--model', type=str, default='models/sign_language_model.h5',
                       help='Path to model file')
    parser.add_argument('--dataset', type=str, default='dataset',
                       help='Path to dataset directory')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of samples to test')
    
    args = parser.parse_args()
    
    test_model(args.model, args.dataset, args.samples)
