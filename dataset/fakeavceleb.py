import os
import csv
import json
from pathlib import Path
from typing import Optional, List, Callable, Any, Union, Tuple

import torch
import torchvision
from torch import Tensor
from tqdm import tqdm

from dataset.lavdf import Lavdf, LavdfDataModule, Metadata, T_LABEL, _default_get_meta_attr


def generate_fakeavceleb_metadata(data_root: str, output_path: str = None, use_test_only: bool = True):
    """
    Generate metadata.min.json for FakeAVCeleb dataset.
    Full videos are treated as single manipulation segments (no temporal localization).
    
    Args:
        data_root: Root directory of FakeAVCeleb dataset
        output_path: Path to save metadata.min.json
        use_test_only: If True, only generate test split (for cross-dataset evaluation)
    """
    if output_path is None:
        output_path = os.path.join(data_root, "metadata.min.json")
    
    meta_csv = os.path.join(data_root, "meta_data.csv")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"meta_data.csv not found at {meta_csv}")
    
    metadata_list = []
    
    # First pass: count total rows for progress bar
    with open(meta_csv, 'r') as f:
        total_rows = sum(1 for _ in csv.DictReader(f))
    
    print(f"Processing {total_rows} entries from meta_data.csv...")
    
    with open(meta_csv, 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(tqdm(reader, total=total_rows, desc="Generating metadata")):
            source = row['source']
            method = row['method']
            category_type = row['type']  # e.g., RealVideo-RealAudio, FakeVideo-FakeAudio
            race = row['race']
            gender = row['gender']
            filename = row['path']
            
            # Construct full path
            video_path = os.path.join(category_type, race, gender, source, filename)
            full_path = os.path.join(data_root, video_path)
            
            # Check if file exists
            if not os.path.exists(full_path):
                continue
            
            # Get video metadata without loading full video (faster)
            try:
                # Use torchvision's video reader to get metadata only
                video_info = torchvision.io.read_video(full_path, pts_unit="sec", end_pts=0.1)
                _, _, info = video_info
                
                # Estimate total frames and duration from metadata
                if 'video_fps' in info and info['video_fps'] > 0:
                    fps = info['video_fps']
                else:
                    fps = 25.0  # default assumption
                
                # Get actual video properties using probe
                try:
                    from torchvision.io import _probe_video_from_file
                    probe_info = _probe_video_from_file(full_path)
                    if probe_info and 'video' in probe_info:
                        video_meta = probe_info['video']
                        duration = video_meta.get('duration', [0])[0]
                        video_frames = int(duration * fps)
                    else:
                        # Fallback: estimate from file
                        duration = 10.0  # rough estimate
                        video_frames = int(duration * fps)
                except:
                    # Fallback: estimate from file
                    duration = 10.0  # rough estimate
                    video_frames = int(duration * fps)
                
                # Audio properties (rough estimate)
                audio_frames = int(duration * 16000)  # 16kHz sample rate
                audio_channels = 2
                
            except Exception as e:
                # Skip videos that can't be read
                continue
            
            # Determine if fake and what modalities are modified
            is_real = method == 'real'
            
            if is_real:
                modify_video = False
                modify_audio = False
                fake_periods = []
                n_fakes = 0
            else:
                # Full video is fake
                n_fakes = 1
                fake_periods = [[0, duration]]
                
                # Determine which modality is fake from category_type
                modify_video = 'FakeVideo' in category_type
                modify_audio = 'FakeAudio' in category_type
            
            # Split determination
            if use_test_only:
                # For cross-dataset evaluation, put everything in test
                split = "test"
            else:
                # 80/10/10 split based on index
                if idx % 10 < 8:
                    split = "train"
                elif idx % 10 == 8:
                    split = "dev"
                else:
                    split = "test"
            
            meta = {
                "file": video_path,
                "n_fakes": n_fakes,
                "fake_periods": fake_periods,
                "duration": duration,
                "original": source if not is_real else None,
                "modify_video": modify_video,
                "modify_audio": modify_audio,
                "split": split,
                "video_frames": video_frames,
                "audio_channels": audio_channels,
                "audio_frames": audio_frames
            }
            
            metadata_list.append(meta)
    
    # Write metadata
    with open(output_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    print(f"\n✓ Generated {len(metadata_list)} metadata entries")
    print(f"✓ Saved to {output_path}")
    
    # Print split statistics
    train_count = sum(1 for m in metadata_list if m['split'] == 'train')
    dev_count = sum(1 for m in metadata_list if m['split'] == 'dev')
    test_count = sum(1 for m in metadata_list if m['split'] == 'test')
    fake_count = sum(1 for m in metadata_list if m['n_fakes'] > 0)
    real_count = len(metadata_list) - fake_count
    
    print(f"✓ Split - Train: {train_count}, Dev: {dev_count}, Test: {test_count}")
    print(f"✓ Label - Real: {real_count}, Fake: {fake_count}")
    
    return metadata_list


class FakeAVCelebDataModule(LavdfDataModule):
    """
    DataModule for FakeAVCeleb dataset.
    Reuses Lavdf infrastructure but generates full-video manipulation labels.
    """
    
    def __init__(self, root: str = "data", frame_padding=512, max_duration=40,
        require_match_scores: bool = False, feature_types: Tuple[Optional[str], Optional[str]] = (None, None),
        batch_size: int = 1, num_workers: int = 0,
        take_train: int = None, take_dev: int = None, take_test: int = None,
        cond: Optional[Callable[[Metadata], bool]] = None,
        get_meta_attr: Callable[[Metadata, Tensor, Tensor, Tensor], List[Any]] = _default_get_meta_attr,
        return_file_name: bool = False
    ):
        # Check if metadata exists, if not generate it
        metadata_path = os.path.join(root, "metadata.min.json")
        if not os.path.exists(metadata_path):
            print(f"Generating FakeAVCeleb metadata at {metadata_path}...")
            generate_fakeavceleb_metadata(root, metadata_path)
        
        super().__init__(
            root=root,
            frame_padding=frame_padding,
            max_duration=max_duration,
            require_match_scores=require_match_scores,
            feature_types=feature_types,
            batch_size=batch_size,
            num_workers=num_workers,
            take_train=take_train,
            take_dev=take_dev,
            take_test=take_test,
            cond=cond,
            get_meta_attr=get_meta_attr,
            return_file_name=return_file_name
        )
