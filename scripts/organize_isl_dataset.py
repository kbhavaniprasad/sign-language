"""
Organize ISL Dataset into train/test folders
"""
import os
import shutil
from pathlib import Path

def organize_isl_dataset(source_dir='isl_dataset', target_dir='dataset', train_count=800, test_count=200):
    """
    Organize ISL dataset into train/test structure
    
    Args:
        source_dir: Source directory with class folders
        target_dir: Target directory (will create train/test subdirs)
        train_count: Number of images for training per class
        test_count: Number of images for testing per class
    """
    print("="*60)
    print("Organizing ISL Dataset")
    print("="*60)
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directories
    train_dir = target_path / 'train'
    test_dir = target_path / 'test'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    print(f"\nFound {len(class_dirs)} classes")
    print(f"Train images per class: {train_count}")
    print(f"Test images per class: {test_count}")
    print(f"\nTotal images: {len(class_dirs) * (train_count + test_count)}")
    print("\n" + "="*60)
    
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Get all images in class
        images = sorted([f for f in class_dir.iterdir() 
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if len(images) < (train_count + test_count):
            print(f"  ⚠️  Warning: Only {len(images)} images found (expected {train_count + test_count})")
            actual_train = int(len(images) * 0.8)
            actual_test = len(images) - actual_train
        else:
            actual_train = train_count
            actual_test = test_count
        
        # Create class directories
        train_class_dir = train_dir / class_name
        test_class_dir = test_dir / class_name
        
        train_class_dir.mkdir(exist_ok=True)
        test_class_dir.mkdir(exist_ok=True)
        
        # Copy training images
        for i, img in enumerate(images[:actual_train]):
            target = train_class_dir / img.name
            if not target.exists():
                shutil.copy2(img, target)
        
        # Copy test images
        for i, img in enumerate(images[actual_train:actual_train + actual_test]):
            target = test_class_dir / img.name
            if not target.exists():
                shutil.copy2(img, target)
        
        print(f"  ✅ Train: {actual_train} images, Test: {actual_test} images")
    
    print("\n" + "="*60)
    print("✅ Dataset organization complete!")
    print("="*60)
    print(f"\nDataset structure:")
    print(f"  {target_dir}/")
    print(f"    ├── train/ ({len(class_dirs)} classes)")
    print(f"    └── test/ ({len(class_dirs)} classes)")
    print("\nYou can now train the model using:")
    print("  python -m jupyter notebook train_model.ipynb")
    print("="*60)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize ISL dataset')
    parser.add_argument('--source', type=str, default='isl_dataset',
                       help='Source directory')
    parser.add_argument('--target', type=str, default='dataset',
                       help='Target directory')
    parser.add_argument('--train', type=int, default=800,
                       help='Training images per class')
    parser.add_argument('--test', type=int, default=200,
                       help='Test images per class')
    
    args = parser.parse_args()
    
    organize_isl_dataset(args.source, args.target, args.train, args.test)
