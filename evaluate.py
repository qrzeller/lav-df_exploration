import argparse
import os

import toml
import torch

# Use Tensor Cores for faster float32 matmul (trades a bit of precision for speed).
# Options: 'medium' or 'high' (higher = more aggressive mixed-precision matmul).
torch.set_float32_matmul_precision('high')

from dataset.lavdf import LavdfDataModule
from dataset.fakeavceleb import FakeAVCelebDataModule
from inference import inference_batfd
from metrics import AP, AR
from model import Batfd, BatfdPlus
from post_process import post_process
from utils import generate_metadata_min, read_json

parser = argparse.ArgumentParser(description="BATFD evaluation")
parser.add_argument("--config", type=str)
parser.add_argument("--data_root", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--modalities", type=str, nargs="+", default=["fusion"])
parser.add_argument("--subset", type=str, nargs="+", default=["full"])
parser.add_argument("--gpus", type=int, default=1)


def visual_subset_condition(meta):
    return not (meta.modify_video is False and meta.modify_audio is True)


def audio_subset_condition(meta):
    return not (meta.modify_video is True and meta.modify_audio is False)


conditions = {
    "full": None,
    "subset_for_visual_only": visual_subset_condition,
    "subset_for_audio_only": audio_subset_condition
}


def evaluate_lavdf(config, args):
    for modal in args.modalities:
        assert modal in ["fusion", "audio", "visual"]

    for subset in args.subset:
        assert subset in ["full", "subset_for_visual_only", "subset_for_audio_only"]

    model_name = config["name"]
    alpha = config["soft_nms"]["alpha"]
    t1 = config["soft_nms"]["t1"]
    t2 = config["soft_nms"]["t2"]

    model_type = config["model_type"]
    v_feature = None
    a_feature = None

    # prepare model
    if config["model_type"] == "batfd_plus":
        model = BatfdPlus.load_from_checkpoint(args.checkpoint)
        require_match_scores = True
        get_meta_attr = BatfdPlus.get_meta_attr
    elif config["model_type"] == "batfd":
        model = Batfd.load_from_checkpoint(args.checkpoint)
        require_match_scores = False
        get_meta_attr = Batfd.get_meta_attr
    else:
        raise ValueError("Invalid model type")

    # prepare dataset
    dm = LavdfDataModule(
        root=args.data_root,
        frame_padding=config["num_frames"],
        require_match_scores=require_match_scores,
        feature_types=(v_feature, a_feature),
        max_duration=config["max_duration"],
        batch_size=args.batch_size, num_workers=args.num_workers,
        get_meta_attr=get_meta_attr,
        return_file_name=True
    )
    dm.setup()

    # inference and save dense proposals as csv file
    inference_batfd(model_name, model, dm, config["max_duration"], model_type, args.modalities, args.gpus)

    # postprocess by soft-nms
    for modality in args.modalities:
        proposal_file_name = f"{model_name}{'' if modality == 'fusion' else '_' + modality[0]}"
        post_process(proposal_file_name, dm.test_dataset.metadata, 25, alpha, t1, t2)

    for modality in args.modalities:
        proposal_file_name = f"{model_name}{'' if modality == 'fusion' else '_' + modality[0]}"
        proposals = read_json(f"output/results/{proposal_file_name}.json")

        for subset_name in args.subset:

            dm_subset = LavdfDataModule(
                root=args.data_root,
                frame_padding=config["num_frames"],
                require_match_scores=require_match_scores,
                max_duration=config["max_duration"],
                batch_size=1, num_workers=3,
                get_meta_attr=get_meta_attr,
                cond=conditions[subset_name]
            )
            dm_subset.setup()

            metadata = dm_subset.test_dataset.metadata
            # evaluate AP
            iou_thresholds = [0.5, 0.75, 0.95]
            print("--------------------------------------------------")
            ap_score = AP(iou_thresholds=iou_thresholds)(metadata, proposals)
            for iou_threshold in iou_thresholds:
                print(f"AP@{iou_threshold} Score for {modality} modality in {subset_name} set: "
                      f"{ap_score[iou_threshold]}")
            print("--------------------------------------------------")

            # evaluate AR
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            n_proposals_list = [100, 50, 20, 10]

            ar_score = AR(n_proposals_list, iou_thresholds=iou_thresholds)(metadata, proposals)

            for n_proposals in n_proposals_list:
                print(f"AR@{n_proposals} Score for {modality} modality in {subset_name} set: "
                      f"{ar_score[n_proposals]}")
        print("--------------------------------------------------")


def evaluate_fakeavceleb(config, args):
    """Evaluate on FakeAVCeleb dataset using LAV-DF trained model."""
    for modal in args.modalities:
        assert modal in ["fusion", "audio", "visual"]

    for subset in args.subset:
        assert subset in ["full", "subset_for_visual_only", "subset_for_audio_only"]

    model_name = config["name"]
    alpha = config["soft_nms"]["alpha"]
    t1 = config["soft_nms"]["t1"]
    t2 = config["soft_nms"]["t2"]

    model_type = config["model_type"]
    v_feature = None
    a_feature = None

    # prepare model
    if config["model_type"] == "batfd_plus":
        model = BatfdPlus.load_from_checkpoint(args.checkpoint)
        require_match_scores = True
        get_meta_attr = BatfdPlus.get_meta_attr
    elif config["model_type"] == "batfd":
        model = Batfd.load_from_checkpoint(args.checkpoint)
        require_match_scores = False
        get_meta_attr = Batfd.get_meta_attr
    else:
        raise ValueError("Invalid model type")

    # prepare dataset with FakeAVCeleb DataModule
    dm = FakeAVCelebDataModule(
        root=args.data_root,
        frame_padding=config["num_frames"],
        require_match_scores=require_match_scores,
        feature_types=(v_feature, a_feature),
        max_duration=config["max_duration"],
        batch_size=args.batch_size, num_workers=args.num_workers,
        get_meta_attr=get_meta_attr,
        return_file_name=True
    )
    dm.setup()

    # inference and save dense proposals as csv file
    inference_batfd(model_name, model, dm, config["max_duration"], model_type, args.modalities, args.gpus)

    # postprocess by soft-nms
    for modality in args.modalities:
        proposal_file_name = f"{model_name}{'' if modality == 'fusion' else '_' + modality[0]}"
        post_process(proposal_file_name, dm.test_dataset.metadata, 25, alpha, t1, t2)

    # Helper function to extract category and race from file path
    def get_category_and_race(file_path):
        """Extract category and race from FakeAVCeleb file path.
        Path format: [type]/[race]/[gender]/[id]/[filename]
        """
        parts = file_path.split('/')
        if len(parts) >= 2:
            category = parts[0]  # e.g., FakeVideo-FakeAudio
            race = parts[1] if len(parts) > 1 else "Unknown"  # e.g., African
            return category, race
        return "Unknown", "Unknown"

    for modality in args.modalities:
        proposal_file_name = f"{model_name}{'' if modality == 'fusion' else '_' + modality[0]}"
        proposals = read_json(f"output/results/{proposal_file_name}.json")

        for subset_name in args.subset:

            dm_subset = FakeAVCelebDataModule(
                root=args.data_root,
                frame_padding=config["num_frames"],
                require_match_scores=require_match_scores,
                max_duration=config["max_duration"],
                batch_size=1, num_workers=3,
                get_meta_attr=get_meta_attr,
                cond=conditions[subset_name]
            )
            dm_subset.setup()

            metadata = dm_subset.test_dataset.metadata
            
            # Overall evaluation
            iou_thresholds = [0.5, 0.75, 0.95]
            print("\n" + "="*80)
            print(f"OVERALL EVALUATION - {modality} modality - {subset_name} set")
            print("="*80)
            ap_score = AP(iou_thresholds=iou_thresholds)(metadata, proposals)
            for iou_threshold in iou_thresholds:
                print(f"AP@{iou_threshold}: {ap_score[iou_threshold]:.4f}")

            iou_thresholds_ar = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            n_proposals_list = [100, 50, 20, 10]
            ar_score = AR(n_proposals_list, iou_thresholds=iou_thresholds_ar)(metadata, proposals)
            for n_proposals in n_proposals_list:
                print(f"AR@{n_proposals}: {ar_score[n_proposals]:.4f}")

            # Group metadata by category
            category_metadata = {}
            for meta in metadata:
                category, _ = get_category_and_race(meta.file)
                if category not in category_metadata:
                    category_metadata[category] = []
                category_metadata[category].append(meta)

            # Evaluate per category
            print("\n" + "="*80)
            print(f"PER-CATEGORY EVALUATION - {modality} modality - {subset_name} set")
            print("="*80)
            for category in sorted(category_metadata.keys()):
                cat_meta = category_metadata[category]
                print(f"\n--- Category: {category} (n={len(cat_meta)}) ---")
                
                ap_score_cat = AP(iou_thresholds=iou_thresholds)(cat_meta, proposals)
                for iou_threshold in iou_thresholds:
                    print(f"  AP@{iou_threshold}: {ap_score_cat[iou_threshold]:.4f}")
                
                ar_score_cat = AR(n_proposals_list, iou_thresholds=iou_thresholds_ar)(cat_meta, proposals)
                for n_proposals in [100, 50]:  # Show fewer for brevity
                    print(f"  AR@{n_proposals}: {ar_score_cat[n_proposals]:.4f}")

            # Group metadata by race
            race_metadata = {}
            for meta in metadata:
                _, race = get_category_and_race(meta.file)
                if race not in race_metadata:
                    race_metadata[race] = []
                race_metadata[race].append(meta)

            # Evaluate per race
            print("\n" + "="*80)
            print(f"PER-RACE EVALUATION - {modality} modality - {subset_name} set")
            print("="*80)
            for race in sorted(race_metadata.keys()):
                race_meta = race_metadata[race]
                print(f"\n--- Race: {race} (n={len(race_meta)}) ---")
                
                ap_score_race = AP(iou_thresholds=iou_thresholds)(race_meta, proposals)
                for iou_threshold in iou_thresholds:
                    print(f"  AP@{iou_threshold}: {ap_score_race[iou_threshold]:.4f}")
                
                ar_score_race = AR(n_proposals_list, iou_thresholds=iou_thresholds_ar)(race_meta, proposals)
                for n_proposals in [100, 50]:  # Show fewer for brevity
                    print(f"  AR@{n_proposals}: {ar_score_race[n_proposals]:.4f}")
            
            print("="*80 + "\n")


if __name__ == '__main__':
    args = parser.parse_args()
    
    if not os.path.exists(os.path.join(args.data_root, "metadata.min.json")):
        generate_metadata_min(args.data_root)

    config = toml.load(args.config)
    torch.backends.cudnn.benchmark = True
    if config["dataset"] == "lavdf":
        evaluate_lavdf(config, args)
    elif config["dataset"] == "fakeavceleb":
        evaluate_fakeavceleb(config, args)
    else:
        raise NotImplementedError
