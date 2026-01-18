"""
Video Frame Extraction for Dynamic Gesture Dataset

This script extracts frames from video files in the Adjectives folder
and organizes them for training the hybrid sign language recognition model.
"""

import cv2
import os
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import json


class VideoFrameExtractor:
    """Extract frames from videos for dynamic gesture recognition"""
    
    def __init__(self, video_dir='Adjectives', output_dir='processed_dynamic_dataset', 
                 frames_per_video=10, image_size=(224, 224)):
        """
        Initialize the frame extractor
        
        Args:
            video_dir: Directory containing video folders
            output_dir: Directory to save extracted frames
            frames_per_video: Number of frames to extract per video
            image_size: Target size for extracted frames
        """
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.frames_per_video = frames_per_video
        self.image_size = image_size
        
        # Create output directories
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.test_dir = self.output_dir / 'test'
        
    def extract_frames_from_video(self, video_path, output_folder, video_name):
        """
        Extract frames from a single video
        
        Args:
            video_path: Path to video file
            output_folder: Folder to save frames
            video_name: Name identifier for the video
            
        Returns:
            Number of frames extracted
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return 0
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print(f"Warning: Video {video_path} has 0 frames")
            cap.release()
            return 0
        
        # Calculate frame indices to extract (uniformly distributed)
        frame_indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
        
        frames_extracted = 0
        
        for idx, frame_idx in enumerate(frame_indices):
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame
                frame_resized = cv2.resize(frame, self.image_size)
                
                # Save frame
                frame_filename = f"{video_name}_frame_{idx:02d}.jpg"
                frame_path = output_folder / frame_filename
                cv2.imwrite(str(frame_path), frame_resized)
                frames_extracted += 1
        
        cap.release()
        return frames_extracted
    
    def process_dataset(self, train_ratio=0.7, val_ratio=0.15):
        """
        Process all videos and split into train/val/test
        
        Args:
            train_ratio: Proportion of videos for training
            val_ratio: Proportion of videos for validation
        """
        print("=" * 60)
        print("Video Frame Extraction for Dynamic Gestures")
        print("=" * 60)
        
        # Remove existing output directory if it exists
        if self.output_dir.exists():
            print(f"\nRemoving existing output directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.train_dir.mkdir(exist_ok=True)
        self.val_dir.mkdir(exist_ok=True)
        self.test_dir.mkdir(exist_ok=True)
        
        # Get all class folders
        class_folders = [f for f in self.video_dir.iterdir() if f.is_dir()]
        class_folders = sorted(class_folders)
        
        print(f"\nFound {len(class_folders)} gesture classes:")
        for folder in class_folders:
            print(f"  - {folder.name}")
        
        stats = {
            'classes': {},
            'total_videos': 0,
            'total_frames': 0,
            'train_frames': 0,
            'val_frames': 0,
            'test_frames': 0
        }
        
        # Process each class
        for class_folder in class_folders:
            class_name = class_folder.name
            print(f"\n{'=' * 60}")
            print(f"Processing class: {class_name}")
            print(f"{'=' * 60}")
            
            # Create class directories
            train_class_dir = self.train_dir / class_name
            val_class_dir = self.val_dir / class_name
            test_class_dir = self.test_dir / class_name
            
            train_class_dir.mkdir(exist_ok=True)
            val_class_dir.mkdir(exist_ok=True)
            test_class_dir.mkdir(exist_ok=True)
            
            # Get all video files
            video_files = list(class_folder.glob('*.MOV')) + list(class_folder.glob('*.mov'))
            video_files = sorted(video_files)
            
            num_videos = len(video_files)
            print(f"Found {num_videos} videos")
            
            if num_videos == 0:
                continue
            
            # Split videos
            num_train = int(num_videos * train_ratio)
            num_val = int(num_videos * val_ratio)
            
            train_videos = video_files[:num_train]
            val_videos = video_files[num_train:num_train + num_val]
            test_videos = video_files[num_train + num_val:]
            
            print(f"Split: {len(train_videos)} train, {len(val_videos)} val, {len(test_videos)} test")
            
            class_stats = {
                'videos': num_videos,
                'train_videos': len(train_videos),
                'val_videos': len(val_videos),
                'test_videos': len(test_videos),
                'train_frames': 0,
                'val_frames': 0,
                'test_frames': 0
            }
            
            # Process train videos
            print("\nExtracting train frames...")
            for video_path in tqdm(train_videos, desc="Train"):
                video_name = video_path.stem
                frames = self.extract_frames_from_video(video_path, train_class_dir, video_name)
                class_stats['train_frames'] += frames
                stats['train_frames'] += frames
            
            # Process validation videos
            print("Extracting validation frames...")
            for video_path in tqdm(val_videos, desc="Val"):
                video_name = video_path.stem
                frames = self.extract_frames_from_video(video_path, val_class_dir, video_name)
                class_stats['val_frames'] += frames
                stats['val_frames'] += frames
            
            # Process test videos
            print("Extracting test frames...")
            for video_path in tqdm(test_videos, desc="Test"):
                video_name = video_path.stem
                frames = self.extract_frames_from_video(video_path, test_class_dir, video_name)
                class_stats['test_frames'] += frames
                stats['test_frames'] += frames
            
            stats['classes'][class_name] = class_stats
            stats['total_videos'] += num_videos
            stats['total_frames'] += (class_stats['train_frames'] + 
                                     class_stats['val_frames'] + 
                                     class_stats['test_frames'])
            
            print(f"\nClass '{class_name}' complete:")
            print(f"  Train frames: {class_stats['train_frames']}")
            print(f"  Val frames: {class_stats['val_frames']}")
            print(f"  Test frames: {class_stats['test_frames']}")
        
        # Save statistics
        stats_file = self.output_dir / 'extraction_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, indent=2, fp=f)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"\nTotal videos processed: {stats['total_videos']}")
        print(f"Total frames extracted: {stats['total_frames']}")
        print(f"\nDataset split:")
        print(f"  Train frames: {stats['train_frames']}")
        print(f"  Val frames: {stats['val_frames']}")
        print(f"  Test frames: {stats['test_frames']}")
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print(f"Statistics saved to: {stats_file.absolute()}")
        
        return stats


def main():
    """Main execution function"""
    # Initialize extractor
    extractor = VideoFrameExtractor(
        video_dir='Adjectives',
        output_dir='processed_dynamic_dataset',
        frames_per_video=10,
        image_size=(224, 224)
    )
    
    # Process dataset
    stats = extractor.process_dataset(train_ratio=0.7, val_ratio=0.15)
    
    print("\n✓ Frame extraction complete!")
    print("You can now use the 'processed_dynamic_dataset' folder for training.")


if __name__ == '__main__':
    main()
