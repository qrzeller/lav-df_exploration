"""
Test script for SimpleVideoDataset pipeline.

This script tests the basic functionality of the new video inference pipeline
without requiring a trained model or actual video files.
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import os


def create_dummy_video(path: str, duration: float = 2.0, fps: int = 25):
    """Create a dummy video file for testing."""
    import torchvision
    
    n_frames = int(duration * fps)
    # Create random video frames (T, H, W, C)
    video = torch.randint(0, 255, (n_frames, 224, 224, 3), dtype=torch.uint8)
    # Create random audio (2 channels, 16kHz)
    audio = torch.randn(2, int(duration * 16000))
    
    # Save video
    torchvision.io.write_video(
        path,
        video,
        fps=fps,
        audio_array=audio,
        audio_fps=16000,
        audio_codec='aac',
        video_codec='h264'
    )


def test_simple_dataset_single_video():
    """Test loading a single video."""
    print("Test 1: Single video loading")
    print("-" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy video
        video_path = os.path.join(tmpdir, "test_video.mp4")
        print(f"Creating dummy video at {video_path}...")
        create_dummy_video(video_path, duration=2.0)
        
        # Load with SimpleVideoDataset
        from dataset import SimpleVideoDataset
        
        dataset = SimpleVideoDataset(
            video_paths=video_path,
            frame_padding=512,
            fps=25,
            return_file_name=True
        )
        
        # Get first item
        video, audio, n_frames, filename = dataset[0]
        
        print(f"✓ Video loaded successfully")
        print(f"  Video shape: {video.shape}")
        print(f"  Audio shape: {audio.shape}")
        print(f"  Number of frames: {n_frames}")
        print(f"  Filename: {filename}")
        
        # Verify shapes
        assert video.shape == (3, 512, 96, 96), f"Expected video shape (3, 512, 96, 96), got {video.shape}"
        assert audio.shape == (64, 2048), f"Expected audio shape (64, 2048), got {audio.shape}"
        assert isinstance(n_frames, int) or isinstance(n_frames, torch.Tensor)
        
        print("✓ Test passed!\n")


def test_simple_dataset_directory():
    """Test loading videos from a directory."""
    print("Test 2: Directory of videos")
    print("-" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple dummy videos
        n_videos = 3
        print(f"Creating {n_videos} dummy videos...")
        for i in range(n_videos):
            video_path = os.path.join(tmpdir, f"video_{i}.mp4")
            create_dummy_video(video_path, duration=1.5)
        
        # Load directory
        from dataset import SimpleVideoDataset
        
        dataset = SimpleVideoDataset(
            video_paths=tmpdir,
            frame_padding=512,
            return_file_name=True
        )
        
        print(f"✓ Found {len(dataset)} videos in directory")
        assert len(dataset) == n_videos, f"Expected {n_videos} videos, found {len(dataset)}"
        
        # Test iteration
        for idx in range(len(dataset)):
            video, audio, n_frames, filename = dataset[idx]
            print(f"  Video {idx}: {os.path.basename(filename)} - {n_frames} frames")
        
        print("✓ Test passed!\n")


def test_create_dataloader():
    """Test creating a DataLoader."""
    print("Test 3: DataLoader creation")
    print("-" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy videos
        video_paths = []
        for i in range(2):
            video_path = os.path.join(tmpdir, f"video_{i}.mp4")
            create_dummy_video(video_path, duration=1.0)
            video_paths.append(video_path)
        
        # Create dataloader
        from dataset import create_simple_dataloader
        
        dataloader = create_simple_dataloader(
            video_paths=video_paths,
            batch_size=2,
            num_workers=0
        )
        
        print(f"✓ DataLoader created with {len(dataloader.dataset)} videos")
        
        # Test batch
        for batch in dataloader:
            video, audio, n_frames, filenames = batch
            print(f"  Batch shape - Video: {video.shape}, Audio: {audio.shape}")
            assert video.shape[0] == 2, "Batch size should be 2"
            assert len(filenames) == 2, "Should have 2 filenames"
            break
        
        print("✓ Test passed!\n")


def test_dataloader_iteration():
    """Test full iteration through dataloader."""
    print("Test 4: DataLoader iteration")
    print("-" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy videos
        n_videos = 5
        for i in range(n_videos):
            create_dummy_video(os.path.join(tmpdir, f"video_{i}.mp4"), duration=1.0)
        
        from dataset import create_simple_dataloader
        
        dataloader = create_simple_dataloader(
            video_paths=tmpdir,
            batch_size=2,
            num_workers=0
        )
        
        processed = 0
        for batch_idx, batch in enumerate(dataloader):
            video, audio, n_frames, filenames = batch
            batch_size = video.shape[0]
            processed += batch_size
            print(f"  Batch {batch_idx + 1}: processed {batch_size} videos")
        
        print(f"✓ Processed {processed}/{n_videos} videos total")
        assert processed == n_videos, f"Expected to process {n_videos} videos, got {processed}"
        print("✓ Test passed!\n")


def test_video_preprocessing():
    """Test video preprocessing steps."""
    print("Test 5: Video preprocessing")
    print("-" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create video with known length
        video_path = os.path.join(tmpdir, "test.mp4")
        duration = 5.0  # 5 seconds
        fps = 25
        create_dummy_video(video_path, duration=duration, fps=fps)
        
        from dataset import SimpleVideoDataset
        
        # Test with different padding sizes
        for frame_padding in [128, 256, 512]:
            dataset = SimpleVideoDataset(
                video_paths=video_path,
                frame_padding=frame_padding,
                fps=fps
            )
            
            video, audio, n_frames, filename = dataset[0]
            
            print(f"  frame_padding={frame_padding}: video shape {video.shape}")
            assert video.shape[1] == frame_padding, f"Expected {frame_padding} frames, got {video.shape[1]}"
        
        print("✓ Test passed!\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running SimpleVideoDataset Tests")
    print("=" * 60)
    print()
    
    try:
        test_simple_dataset_single_video()
        test_simple_dataset_directory()
        test_create_dataloader()
        test_dataloader_iteration()
        test_video_preprocessing()
        
        print("=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
