"""
Simple inference script for processing new videos without metadata.

This script demonstrates how to use the SimpleVideoDataset for inference
on new video files that don't have associated metadata or labels.

Usage:
    # Single video file
    python simple_inference.py --video path/to/video.mp4 --model path/to/checkpoint.ckpt
    
    # Directory of videos
    python simple_inference.py --video path/to/videos/ --model path/to/checkpoint.ckpt
    
    # List of specific videos
    python simple_inference.py --videos video1.mp4 video2.mp4 video3.mp4 --model path/to/checkpoint.ckpt
"""

import argparse
import os
from pathlib import Path
from typing import List

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from dataset import create_simple_dataloader, SimpleVideoDataset
from model.batfd import Batfd
from model.batfd_plus import BatfdPlus


def load_model(checkpoint_path: str, model_type: str = "batfd_plus"):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint file
        model_type: Type of model ('batfd' or 'batfd_plus')
        
    Returns:
        Loaded model ready for inference
    """
    if model_type == "batfd":
        model = Batfd.load_from_checkpoint(checkpoint_path)
    elif model_type == "batfd_plus":
        model = BatfdPlus.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    return model


def run_inference(
    video_paths: str | List[str],
    model_checkpoint: str,
    model_type: str = "batfd_plus",
    output_dir: str = "output/simple_inference",
    batch_size: int = 1,
    num_workers: int = 0,
    device: str = "auto"
):
    """
    Run inference on video files.
    
    Args:
        video_paths: Single video path, directory path, or list of video paths
        model_checkpoint: Path to model checkpoint
        model_type: Type of model ('batfd' or 'batfd_plus')
        output_dir: Directory to save results
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading {model_type} model from {model_checkpoint}...")
    model = load_model(model_checkpoint, model_type)
    
    # Create dataloader
    print(f"Loading videos from {video_paths}...")
    dataloader = create_simple_dataloader(
        video_paths=video_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        return_file_name=True
    )
    
    print(f"Processing {len(dataloader.dataset)} videos...")
    
    # Set device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    model = model.to(device)
    
    # Run inference
    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            video, audio, n_frames, file_names = batch
            
            # Move to device
            video = video.to(device)
            audio = audio.to(device)
            
            # Run model
            if model_type == "batfd":
                fusion_bm, visual_bm, audio_bm = model(video, audio)
                outputs = {
                    'fusion_boundary_map': fusion_bm.cpu(),
                    'visual_boundary_map': visual_bm.cpu(),
                    'audio_boundary_map': audio_bm.cpu()
                }
            elif model_type == "batfd_plus":
                fusion_bm, fusion_start, fusion_end, visual_bm, visual_start, visual_end, audio_bm, audio_start, audio_end = model(video, audio)
                outputs = {
                    'fusion_boundary_map': fusion_bm.cpu(),
                    'fusion_start': fusion_start.cpu() if fusion_start is not None else None,
                    'fusion_end': fusion_end.cpu() if fusion_end is not None else None,
                    'visual_boundary_map': visual_bm.cpu(),
                    'visual_start': visual_start.cpu() if visual_start is not None else None,
                    'visual_end': visual_end.cpu() if visual_end is not None else None,
                    'audio_boundary_map': audio_bm.cpu(),
                    'audio_start': audio_start.cpu() if audio_start is not None else None,
                    'audio_end': audio_end.cpu() if audio_end is not None else None,
                }
            
            # Save results for each video in batch
            for i in range(len(file_names)):
                video_name = os.path.basename(file_names[i])
                result = {
                    'file': file_names[i],
                    'n_frames': n_frames[i].item(),
                    'outputs': {k: v[i] if v is not None else None for k, v in outputs.items()}
                }
                results.append(result)
                
                # Save boundary maps as numpy files
                output_file = os.path.join(output_dir, video_name.replace('.mp4', '.pt'))
                torch.save(result['outputs'], output_file)
                print(f"  [{batch_idx + 1}/{len(dataloader)}] Processed {video_name} -> {output_file}")
    
    print(f"\nInference complete! Results saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference on videos without metadata")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to a single video file or directory containing videos")
    group.add_argument("--videos", nargs="+", type=str, help="List of video file paths")
    
    # Model options
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, default="batfd_plus", 
                       choices=["batfd", "batfd_plus"], help="Model type")
    
    # Processing options
    parser.add_argument("--output-dir", type=str, default="output/simple_inference",
                       help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"], help="Device to use")
    
    args = parser.parse_args()
    
    # Determine video paths
    video_paths = args.video if args.video else args.videos
    
    # Run inference
    run_inference(
        video_paths=video_paths,
        model_checkpoint=args.model,
        model_type=args.model_type,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device
    )


if __name__ == "__main__":
    main()
