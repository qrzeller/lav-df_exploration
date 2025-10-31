"""
Simple video dataset for inference on new videos without metadata.
This dataset takes video files as input and prepares them for model inference.
"""

import os
from pathlib import Path
from typing import Optional, List, Callable, Any, Union

import torch
import torchaudio
from einops import rearrange
from torch import Tensor
from torch.nn import Identity
from torch.utils.data import Dataset, DataLoader

from utils import read_video, padding_video, padding_audio, resize_video


class SimpleVideoDataset(Dataset):
    """
    A simple dataset for processing video files for inference.
    
    Args:
        video_paths: List of video file paths or a directory containing videos
        frame_padding: Number of frames to pad/truncate videos to (default: 512)
        fps: Frames per second (default: 25)
        video_transform: Optional transform to apply to video tensors
        audio_transform: Optional transform to apply to audio tensors
        return_file_name: Whether to return the file name with the data
    """

    def __init__(
        self,
        video_paths: Union[str, List[str]],
        frame_padding: int = 512,
        fps: int = 25,
        video_transform: Callable[[Tensor], Tensor] = Identity(),
        audio_transform: Callable[[Tensor], Tensor] = Identity(),
        return_file_name: bool = True
    ):
        self.frame_padding = frame_padding
        self.audio_padding = int(frame_padding / fps * 16000)
        self.fps = fps
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.return_file_name = return_file_name

        # Handle input: either a directory or list of file paths
        if isinstance(video_paths, str):
            if os.path.isdir(video_paths):
                # Load all video files from directory
                self.video_paths = self._get_video_files_from_dir(video_paths)
            else:
                # Single file path
                self.video_paths = [video_paths]
        else:
            # List of file paths
            self.video_paths = video_paths

        # Validate that all files exist
        for path in self.video_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video file not found: {path}")

        print(f"Loaded {len(self.video_paths)} videos for inference.")

    def _get_video_files_from_dir(self, directory: str) -> List[str]:
        """Get all video files from a directory."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        video_files = []
        
        for file in sorted(os.listdir(directory)):
            if Path(file).suffix.lower() in video_extensions:
                video_files.append(os.path.join(directory, file))
        
        return video_files

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> Union[tuple, List]:
        """
        Get a video sample.
        
        Returns:
            If return_file_name is True: (video, audio, n_frames, file_name)
            Otherwise: (video, audio, n_frames)
        """
        video_path = self.video_paths[index]
        
        # Read video and audio
        video, audio, info = read_video(video_path)
        n_frames = video.shape[0]
        
        # Pad/truncate to target length
        video = padding_video(video, target=self.frame_padding)
        audio = padding_audio(audio, target=self.audio_padding)

        # Apply transforms
        video = self.video_transform(video)
        audio = self.audio_transform(audio)

        # Resize and rearrange video: (T, C, H, W) -> (C, T, H, W)
        video = rearrange(resize_video(video, (96, 96)), "t c h w -> c t h w")
        
        # Convert audio to log mel spectrogram
        audio = self._get_log_mel_spectrogram(audio)

        outputs = [video, audio, n_frames]

        if self.return_file_name:
            outputs.append(video_path)

        return outputs

    @staticmethod
    def _get_log_mel_spectrogram(audio: Tensor) -> Tensor:
        """Convert audio to log mel spectrogram."""
        ms = torchaudio.transforms.MelSpectrogram(n_fft=321, n_mels=64)
        spec = torch.log(ms(audio[:, 0]) + 0.01)
        assert spec.shape == (64, 2048), "Wrong log mel-spectrogram setup in Dataset"
        return spec


def create_simple_dataloader(
    video_paths: Union[str, List[str]],
    batch_size: int = 1,
    num_workers: int = 0,
    frame_padding: int = 512,
    fps: int = 25,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for simple video inference.
    
    Args:
        video_paths: Path to video file, directory, or list of video paths
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        frame_padding: Number of frames to pad/truncate videos to
        fps: Frames per second
        **dataset_kwargs: Additional arguments to pass to SimpleVideoDataset
        
    Returns:
        DataLoader ready for inference
        
    Example:
        >>> dataloader = create_simple_dataloader("path/to/videos/")
        >>> for batch in dataloader:
        >>>     video, audio, n_frames, file_names = batch
        >>>     # Process batch...
    """
    dataset = SimpleVideoDataset(
        video_paths=video_paths,
        frame_padding=frame_padding,
        fps=fps,
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
