"""
Create Sample Dataset for Testing
This script creates a small sample dataset if you don't have one yet.
"""
import os
import cv2
import numpy as np
from pathlib import Path

def create_sample_dataset(output_dir='dataset', num_classes=5, images_per_class=20):
    """
    Create a sample dataset with synthetic images
    
    Args:
        output_dir: Output directory for dataset
        num_classes: Number of gesture classes
        images_per_class: Number of images per class
    """
    print("="*60)
    print("Creating Sample Dataset")
    print("="*60)
    
    # Class names (you can customize these)
    class_names = [
        'hello', 'thanks', 'yes', 'no', 'please',
        'sorry', 'help', 'stop', 'go', 'wait'
    ][:num_classes]
    
    # Create directories
    for split in ['train', 'test']:
        for class_name in class_names:
            class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    print(f"\n📁 Creating {num_classes} classes with {images_per_class} images each")
    print(f"📝 Classes: {', '.join(class_names)}\n")
    
    # Generate synthetic images
    for split in ['train', 'test']:
        num_images = images_per_class if split == 'train' else max(5, images_per_class // 4)
        
        print(f"Generating {split} set...")
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(output_dir, split, class_name)
            
            for img_idx in range(num_images):
                # Create synthetic image (colored rectangle with noise)
                img = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
                
                # Add colored rectangle (different color for each class)
                color = (
                    (class_idx * 50) % 255,
                    (class_idx * 80) % 255,
                    (class_idx * 120) % 255
                )
                
                cv2.rectangle(img, (50, 50), (174, 174), color, -1)
                
                # Add some random shapes to make it more varied
                for _ in range(5):
                    x, y = np.random.randint(0, 224, 2)
                    radius = np.random.randint(5, 20)
                    cv2.circle(img, (x, y), radius, color, -1)
                
                # Add text
                cv2.putText(img, class_name, (60, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save image
                img_path = os.path.join(class_dir, f'{class_name}_{img_idx:04d}.jpg')
                cv2.imwrite(img_path, img)
            
            print(f"  ✅ {class_name}: {num_images} images")
    
    print("\n" + "="*60)
    print("✅ Sample dataset created successfully!")
    print("="*60)
    print(f"\n📂 Dataset location: {os.path.abspath(output_dir)}")
    print("\n⚠️  NOTE: This is a SYNTHETIC dataset for testing purposes.")
    print("For real sign language recognition, please use an actual dataset.")
    print("\nPopular datasets:")
    print("  - ASL Alphabet: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
    print("  - Sign Language MNIST: https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
    print("  - Indian Sign Language: https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl")
    print("="*60)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create sample dataset')
    parser.add_argument('--output', type=str, default='dataset',
                       help='Output directory')
    parser.add_argument('--classes', type=int, default=5,
                       help='Number of classes')
    parser.add_argument('--images', type=int, default=20,
                       help='Images per class (train set)')
    
    args = parser.parse_args()
    
    create_sample_dataset(args.output, args.classes, args.images)
