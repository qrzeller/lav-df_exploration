#!/usr/bin/env python3
"""Quick profiler to identify hotspots in dataset IO and model forward.

Usage:
  python profile/profile_quick.py --data_root /path/to/LAV-DF --checkpoint /path/to/ckpt --model_type batfd --n_samples 50

This script does two short profiles and saves them to profile/io.prof and profile/model.prof
and prints the top cumulative functions.
"""
import argparse
import os
import time
import cProfile
import pstats

import torch
import sys
# ensure repo root is on sys.path so local imports work when running the script from / or other cwd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.lavdf import LavdfDataModule
from model import Batfd, BatfdPlus


def profile_dataset(root, n_samples=50, frame_padding=512, max_duration=40):
    print(f"Profiling dataset __getitem__ for {n_samples} samples (single-process)...")
    dm = LavdfDataModule(root=root, frame_padding=frame_padding, max_duration=max_duration,
                         batch_size=1, num_workers=0, return_file_name=True)
    dm.setup()
    ds = dm.test_dataset

    def run():
        for i in range(min(n_samples, len(ds))):
            _ = ds[i]

    prof_path = os.path.join("profile", "io.prof")
    os.makedirs("profile", exist_ok=True)
    cProfile.runctx('run()', globals(), locals(), filename=prof_path)
    print(f"Saved dataset profile to {prof_path}")

    ps = pstats.Stats(prof_path).strip_dirs().sort_stats("cumulative")
    print("Top dataset functions by cumulative time:")
    ps.print_stats(20)


def profile_model(root, checkpoint, model_type="batfd", n_batches=20, batch_size=1, frame_padding=512, max_duration=40):
    print(f"Profiling model forward for {n_batches} batches (no grad)...")
    dm = LavdfDataModule(root=root, frame_padding=frame_padding, max_duration=max_duration,
                         batch_size=batch_size, num_workers=0, return_file_name=True)
    dm.setup()
    ds = dm.test_dataset

    # load model
    if model_type == "batfd_plus":
        model_cls = BatfdPlus
    else:
        model_cls = Batfd

    if checkpoint is None:
        raise RuntimeError("Please provide --checkpoint to load the model checkpoint for forward profiling")

    model = model_cls.load_from_checkpoint(checkpoint)
    model.eval()

    # collect a small in-memory batch list
    batches = []
    idx = 0
    while len(batches) < n_batches:
        if idx >= len(ds):
            break
        out = ds[idx]
        # out format: [video, audio, label, ...] or extended; first two are video and audio
        video = out[0].unsqueeze(0)
        audio = out[1].unsqueeze(0)
        batches.append((video, audio))
        idx += 1

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(device)

    # PRE-MOVE: move all batches to device before measuring to isolate pure model compute time
    batches_on_device = []
    for v, a in batches:
        batches_on_device.append((v.to(device), a.to(device)))

    def run_model():
        with torch.no_grad():
            for v, a in batches_on_device:
                # model forward signature may vary; try calling model(v, a) then fallback to model.forward
                try:
                    _ = model(v, a)
                except TypeError:
                    _ = model.forward(v, a)

    prof_path = os.path.join("profile", "model.prof")
    cProfile.runctx('run_model()', globals(), locals(), filename=prof_path)
    print(f"Saved model profile to {prof_path}")

    ps = pstats.Stats(prof_path).strip_dirs().sort_stats("cumulative")
    print("Top model functions by cumulative time:")
    ps.print_stats(20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", choices=["batfd", "batfd_plus"], default="batfd")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--n_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    print("Starting quick profiling. This runs dataset __getitem__ in-process (num_workers=0) to capture IO/decoding time.")
    profile_dataset(args.data_root, n_samples=args.n_samples)
    profile_model(args.data_root, args.checkpoint, model_type=args.model_type, n_batches=args.n_batches, batch_size=args.batch_size)

    print("Done. Use 'python -m pstats profile/io.prof' or tools like snakeviz to inspect the .prof files.")


if __name__ == '__main__':
    main()
