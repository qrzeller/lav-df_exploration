"""
Minimal example of using SimpleVideoDataset for inference.

This is a bare-bones example showing the simplest way to use the new pipeline.
"""

import torch
from dataset import create_simple_dataloader


def minimal_example():
    """Minimal example: process a single video."""
    
    # 1. Create a dataloader for your video(s)
    # Can be: single file, directory, or list of files
    dataloader = create_simple_dataloader(
        video_paths="path/to/your/video.mp4",  # Change this!
        batch_size=1
    )
    
    # 2. Load your model (example with BATFD+)
    from model.batfd_plus import BatfdPlus
    model = BatfdPlus.load_from_checkpoint("path/to/checkpoint.ckpt")  # Change this!
    model.eval()
    
    # 3. Process videos
    with torch.no_grad():
        for video, audio, n_frames, file_name in dataloader:
            # Run model
            outputs = model(video, audio)
            
            # Extract boundary maps
            fusion_bm, fusion_start, fusion_end, _, _, _, _, _, _ = outputs
            
            # Print results
            print(f"Processed: {file_name[0]}")
            print(f"Video frames: {n_frames[0]}")
            print(f"Boundary map shape: {fusion_bm.shape}")
            print(f"Max boundary score: {fusion_bm.max():.4f}")
            print("-" * 50)


def process_directory_example():
    """Process all videos in a directory."""
    
    from dataset import create_simple_dataloader
    from model.batfd_plus import BatfdPlus
    import torch
    
    # Setup
    model = BatfdPlus.load_from_checkpoint("path/to/checkpoint.ckpt")
    model.eval()
    
    # Load all videos from directory
    dataloader = create_simple_dataloader(
        video_paths="path/to/videos/directory/",
        batch_size=1
    )
    
    print(f"Found {len(dataloader.dataset)} videos to process\n")
    
    # Process each video
    results = []
    with torch.no_grad():
        for idx, (video, audio, n_frames, file_name) in enumerate(dataloader, 1):
            # Inference
            fusion_bm, _, _, _, _, _, _, _, _ = model(video, audio)
            
            # Store result
            result = {
                'file': file_name[0],
                'frames': n_frames[0].item(),
                'max_score': fusion_bm.max().item()
            }
            results.append(result)
            
            print(f"[{idx}/{len(dataloader)}] {result['file']}: max_score={result['max_score']:.4f}")
    
    return results


def process_multiple_videos_example():
    """Process specific video files."""
    
    from dataset import SimpleVideoDataset
    from torch.utils.data import DataLoader
    from model.batfd import Batfd
    import torch
    
    # List of specific videos
    videos_to_process = [
        "video1.mp4",
        "video2.mp4", 
        "video3.mp4"
    ]
    
    # Create dataset
    dataset = SimpleVideoDataset(
        video_paths=videos_to_process,
        return_file_name=True
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load model
    model = Batfd.load_from_checkpoint("path/to/checkpoint.ckpt")
    model.eval()
    
    # Process
    with torch.no_grad():
        for video, audio, n_frames, file_name in dataloader:
            fusion_bm, visual_bm, audio_bm = model(video, audio)
            
            print(f"File: {file_name[0]}")
            print(f"  Fusion score: {fusion_bm.max():.4f}")
            print(f"  Visual score: {visual_bm.max():.4f}")
            print(f"  Audio score: {audio_bm.max():.4f}")


def main():
    """Run the examples."""
    
    print("=" * 60)
    print("SimpleVideoDataset Examples")
    print("=" * 60)
    print()
    
    print("Example 1: Minimal single video processing")
    print("-" * 60)
    try:
        minimal_example()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to update the video path and checkpoint path!")
    
    print()
    print("Example 2: Process directory")
    print("-" * 60)
    try:
        results = process_directory_example()
        print(f"\nProcessed {len(results)} videos successfully!")
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    print("Example 3: Process specific videos")
    print("-" * 60)
    try:
        process_multiple_videos_example()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
