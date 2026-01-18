"""
Reorganize ISL Dataset - Create train/test folders inside isl_dataset
"""
import os
import shutil
from pathlib import Path

def reorganize_isl_dataset(source_dir='isl_dataset', train_count=800, test_count=200):
    """
    Reorganize ISL dataset to have train/test folders inside isl_dataset
    
    Structure will be:
    isl_dataset/
    ├── train/
    │   ├── 0/ (800 images)
    │   ├── 1/ (800 images)
    │   └── ...
    └── test/
        ├── 0/ (200 images)
        ├── 1/ (200 images)
        └── ...
    """
    print("="*60)
    print("Reorganizing ISL Dataset")
    print("="*60)
    
    source_path = Path(source_dir)
    
    # Check current structure
    print(f"\nChecking {source_dir} structure...")
    
    # Get all class directories (should be 0-9, A-Z at root level)
    class_dirs = [d for d in source_path.iterdir() 
                  if d.is_dir() and d.name not in ['train', 'test']]
    
    if not class_dirs:
        print("❌ No class folders found in isl_dataset!")
        print("Expected folders: 0, 1, 2, ..., 9, A, B, C, ..., Z")
        return
    
    print(f"✅ Found {len(class_dirs)} class folders")
    
    # Create train and test directories
    train_dir = source_path / 'train'
    test_dir = source_path / 'test'
    
    # Remove existing train/test if they exist
    if train_dir.exists():
        print("Removing existing train folder...")
        shutil.rmtree(train_dir)
    if test_dir.exists():
        print("Removing existing test folder...")
        shutil.rmtree(test_dir)
    
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    print(f"\nOrganizing images:")
    print(f"  Train: {train_count} images per class")
    print(f"  Test: {test_count} images per class")
    print("\n" + "="*60)
    
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Get all images in class
        images = sorted([f for f in class_dir.iterdir() 
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        
        total_images = len(images)
        print(f"  Found {total_images} images")
        
        if total_images < (train_count + test_count):
            print(f"  ⚠️  Warning: Only {total_images} images (need {train_count + test_count})")
            actual_train = int(total_images * 0.8)
            actual_test = total_images - actual_train
        else:
            actual_train = train_count
            actual_test = test_count
        
        # Create class directories in train and test
        train_class_dir = train_dir / class_name
        test_class_dir = test_dir / class_name
        
        train_class_dir.mkdir(exist_ok=True)
        test_class_dir.mkdir(exist_ok=True)
        
        # Move training images
        print(f"  Moving {actual_train} images to train/{class_name}/")
        for i, img in enumerate(images[:actual_train]):
            target = train_class_dir / img.name
            shutil.move(str(img), str(target))
        
        # Move test images
        print(f"  Moving {actual_test} images to test/{class_name}/")
        for i, img in enumerate(images[actual_train:actual_train + actual_test]):
            target = test_class_dir / img.name
            shutil.move(str(img), str(target))
        
        # Remove empty class directory
        if class_dir.exists() and not list(class_dir.iterdir()):
            class_dir.rmdir()
            print(f"  ✅ Removed empty folder: {class_name}/")
    
    print("\n" + "="*60)
    print("✅ Dataset reorganization complete!")
    print("="*60)
    print(f"\nNew structure:")
    print(f"  {source_dir}/")
    print(f"    ├── train/ ({len(class_dirs)} classes)")
    print(f"    └── test/ ({len(class_dirs)} classes)")
    print("\nYou can now train the model!")
    print("="*60)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Reorganize ISL dataset')
    parser.add_argument('--source', type=str, default='isl_dataset',
                       help='Source directory')
    parser.add_argument('--train', type=int, default=800,
                       help='Training images per class')
    parser.add_argument('--test', type=int, default=200,
                       help='Test images per class')
    
    args = parser.parse_args()
    
    reorganize_isl_dataset(args.source, args.train, args.test)
