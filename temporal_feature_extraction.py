"""
Temporal Feature Extraction for Deepfake Detection

This script extracts temporal distribution features from boundary prediction CSVs
and trains a Random Forest classifier for improved deepfake detection.

Usage:
    python temporal_feature_extraction.py --config config/batfd_fakeavceleb.toml
    python temporal_feature_extraction.py --config config/batfd_default.toml --analyze-only
"""

import argparse
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import toml
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from tqdm import tqdm

from dataset.fakeavceleb import FakeAVCelebDataModule
from dataset.lavdf import LavdfDataModule
from model.batfd import Batfd
from utils import read_json


def extract_temporal_features(scores: np.ndarray) -> Dict[str, float]:
    """
    Extract temporal distribution features from boundary scores.
    
    Args:
        scores: Array of boundary scores from CSV file
        
    Returns:
        Dictionary of 17 temporal features
    """
    if len(scores) == 0:
        return None
    
    features = {
        # Basic statistics
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'median_score': np.median(scores),
        'q25_score': np.percentile(scores, 25),
        'q75_score': np.percentile(scores, 75),
        'iqr_score': np.percentile(scores, 75) - np.percentile(scores, 25),
        
        # Distribution shape
        'skew': pd.Series(scores).skew(),
        'kurtosis': pd.Series(scores).kurtosis(),
        'entropy': entropy(np.histogram(scores, bins=20)[0] + 1e-10),
        
        # Variability
        'score_range': np.max(scores) - np.min(scores),
        'coef_variation': np.std(scores) / (np.mean(scores) + 1e-10),
        
        # Score concentration
        'pct_high_scores': np.mean(scores > 0.5),
        'pct_medium_scores': np.mean((scores > 0.1) & (scores <= 0.5)),
        'pct_low_scores': np.mean(scores <= 0.1),
        
        # Temporal patterns
        'first_half_mean': np.mean(scores[:len(scores)//2]) if len(scores) >= 4 else np.mean(scores),
        'second_half_mean': np.mean(scores[len(scores)//2:]) if len(scores) >= 4 else np.mean(scores),
        'temporal_diff': abs(np.mean(scores[:len(scores)//2]) - np.mean(scores[len(scores)//2:])) if len(scores) >= 4 else 0,
        'n_proposals': len(scores),
    }
    
    return features


def get_category_and_race(file_path: str) -> Tuple[str, str, str]:
    """
    Extract category, race, and gender from FakeAVCeleb file path.
    
    Args:
        file_path: Path like 'FaceSwap/African/female/id20_id04_0001.mp4'
        
    Returns:
        Tuple of (category, race, gender)
    """
    parts = file_path.split('/')
    if len(parts) >= 3:
        category = parts[0]
        race = parts[1]
        gender = parts[2]
        return category, race, gender
    return 'Unknown', 'Unknown', 'Unknown'


def load_dataset_features(
    csv_dir: str,
    metadata_list: List,
    dataset_name: str = "dataset"
) -> pd.DataFrame:
    """
    Load temporal features from CSV files for all videos.
    
    Args:
        csv_dir: Directory containing CSV files with boundary predictions
        metadata_list: List of metadata objects with video information
        dataset_name: Name of dataset for progress bar
        
    Returns:
        DataFrame with features and labels
    """
    print(f"\n{'='*80}")
    print(f"EXTRACTING TEMPORAL FEATURES FROM {dataset_name.upper()}")
    print(f"{'='*80}")
    
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files")
    
    # Create lookup dict for faster metadata search
    metadata_dict = {}
    for meta in metadata_list:
        filename = meta.file.split('/')[-1]
        metadata_dict[filename] = meta
    
    feature_list = []
    
    for csv_file in tqdm(csv_files, desc="Processing videos"):
        video_file = csv_file.replace('.csv', '.mp4')
        csv_path = os.path.join(csv_dir, csv_file)
        
        # Find metadata
        meta = metadata_dict.get(video_file)
        if meta is None:
            continue
        
        is_fake = meta.n_fakes > 0
        
        # Read temporal scores
        try:
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                continue
            scores = df['score'].values
        except Exception as e:
            print(f"Warning: Failed to read {csv_file}: {e}")
            continue
        
        # Extract features
        features = extract_temporal_features(scores)
        if features is None:
            continue
        
        features['file'] = video_file
        features['is_fake'] = is_fake
        
        # Extract category, race, gender if FakeAVCeleb
        category, race, gender = get_category_and_race(meta.file)
        features['category'] = category
        features['race'] = race
        features['gender'] = gender
        
        feature_list.append(features)
    
    df_features = pd.DataFrame(feature_list)
    print(f"\nExtracted features for {len(df_features)} videos")
    print(f"  Real videos: {len(df_features[df_features['is_fake'] == False])}")
    print(f"  Fake videos: {len(df_features[df_features['is_fake'] == True])}")
    
    return df_features


def analyze_features(df_features: pd.DataFrame):
    """Analyze and compare features between real and fake videos."""
    from scipy.stats import ks_2samp
    
    real_df = df_features[df_features['is_fake'] == False]
    fake_df = df_features[df_features['is_fake'] == True]
    
    print(f"\n{'='*80}")
    print("FEATURE COMPARISON: REAL vs FAKE")
    print(f"{'='*80}")
    
    feature_cols = [col for col in df_features.columns 
                    if col not in ['file', 'is_fake', 'category', 'race', 'gender']]
    
    print(f"\n{'Feature':>25} | {'Real Mean':>12} | {'Fake Mean':>12} | {'Difference':>12} | {'KS p-value':>12}")
    print('='*90)
    
    for feature in feature_cols:
        real_vals = real_df[feature].values
        fake_vals = fake_df[feature].values
        
        # Remove NaN/Inf
        real_vals = real_vals[np.isfinite(real_vals)]
        fake_vals = fake_vals[np.isfinite(fake_vals)]
        
        if len(real_vals) == 0 or len(fake_vals) == 0:
            continue
        
        real_mean = np.mean(real_vals)
        fake_mean = np.mean(fake_vals)
        diff = fake_mean - real_mean
        
        ks_stat, ks_pval = ks_2samp(real_vals, fake_vals)
        
        print(f'{feature:>25} | {real_mean:>12.6f} | {fake_mean:>12.6f} | {diff:>+12.6f} | {ks_pval:>12.6e}')


def analyze_by_demographics(df_features: pd.DataFrame, clf, feature_cols: List[str]):
    """Analyze classifier performance by race and gender."""
    print(f"\n{'='*80}")
    print("PERFORMANCE BY DEMOGRAPHICS")
    print(f"{'='*80}")
    
    # Prepare features
    X = df_features[feature_cols].fillna(0).values
    y_true = df_features['is_fake'].values
    y_proba = clf.predict_proba(X)[:, 1]
    
    # Overall performance
    overall_auc = roc_auc_score(y_true, y_proba)
    overall_ap = average_precision_score(y_true, y_proba)
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  AUC-ROC: {overall_auc:.4f}")
    print(f"  Average Precision: {overall_ap:.4f}")
    print(f"  Total videos: {len(df_features)}")
    
    # By race
    print(f"\n{'='*80}")
    print("PERFORMANCE BY RACE")
    print(f"{'='*80}")
    
    races = df_features['race'].unique()
    races = [r for r in races if r != 'Unknown']
    
    if len(races) > 0:
        print(f"\n{'Race':>20} | {'N':>6} | {'Real':>6} | {'Fake':>6} | {'AUC':>8} | {'AP':>8}")
        print('='*70)
        
        for race in sorted(races):
            race_df = df_features[df_features['race'] == race]
            if len(race_df) < 10:
                continue
            
            race_y_true = race_df['is_fake'].values
            race_indices = df_features['race'] == race
            race_y_proba = y_proba[race_indices]
            
            n_real = (race_y_true == False).sum()
            n_fake = (race_y_true == True).sum()
            
            if n_real > 0 and n_fake > 0:
                race_auc = roc_auc_score(race_y_true, race_y_proba)
                race_ap = average_precision_score(race_y_true, race_y_proba)
                print(f"{race:>20} | {len(race_df):>6} | {n_real:>6} | {n_fake:>6} | {race_auc:>8.4f} | {race_ap:>8.4f}")
            else:
                print(f"{race:>20} | {len(race_df):>6} | {n_real:>6} | {n_fake:>6} | {'N/A':>8} | {'N/A':>8}")
    
    # By gender
    print(f"\n{'='*80}")
    print("PERFORMANCE BY GENDER")
    print(f"{'='*80}")
    
    genders = df_features['gender'].unique()
    genders = [g for g in genders if g != 'Unknown']
    
    if len(genders) > 0:
        print(f"\n{'Gender':>20} | {'N':>6} | {'Real':>6} | {'Fake':>6} | {'AUC':>8} | {'AP':>8}")
        print('='*70)
        
        for gender in sorted(genders):
            gender_df = df_features[df_features['gender'] == gender]
            if len(gender_df) < 10:
                continue
            
            gender_y_true = gender_df['is_fake'].values
            gender_indices = df_features['gender'] == gender
            gender_y_proba = y_proba[gender_indices]
            
            n_real = (gender_y_true == False).sum()
            n_fake = (gender_y_true == True).sum()
            
            if n_real > 0 and n_fake > 0:
                gender_auc = roc_auc_score(gender_y_true, gender_y_proba)
                gender_ap = average_precision_score(gender_y_true, gender_y_proba)
                print(f"{gender:>20} | {len(gender_df):>6} | {n_real:>6} | {n_fake:>6} | {gender_auc:>8.4f} | {gender_ap:>8.4f}")
            else:
                print(f"{gender:>20} | {len(gender_df):>6} | {n_real:>6} | {n_fake:>6} | {'N/A':>8} | {'N/A':>8}")
    
    # By category
    print(f"\n{'='*80}")
    print("PERFORMANCE BY CATEGORY (Fake Method)")
    print(f"{'='*80}")
    
    categories = df_features['category'].unique()
    categories = [c for c in categories if c != 'Unknown']
    
    if len(categories) > 0:
        print(f"\n{'Category':>20} | {'N':>6} | {'Real':>6} | {'Fake':>6} | {'AUC':>8} | {'AP':>8}")
        print('='*70)
        
        for category in sorted(categories):
            cat_df = df_features[df_features['category'] == category]
            if len(cat_df) < 10:
                continue
            
            cat_y_true = cat_df['is_fake'].values
            cat_indices = df_features['category'] == category
            cat_y_proba = y_proba[cat_indices]
            
            n_real = (cat_y_true == False).sum()
            n_fake = (cat_y_true == True).sum()
            
            if n_real > 0 and n_fake > 0:
                cat_auc = roc_auc_score(cat_y_true, cat_y_proba)
                cat_ap = average_precision_score(cat_y_true, cat_y_proba)
                print(f"{category:>20} | {len(cat_df):>6} | {n_real:>6} | {n_fake:>6} | {cat_auc:>8.4f} | {cat_ap:>8.4f}")
            else:
                print(f"{category:>20} | {len(cat_df):>6} | {n_real:>6} | {n_fake:>6} | {'N/A':>8} | {'N/A':>8}")


def train_classifier(
    df_features: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, Dict]:
    """
    Train Random Forest classifier on temporal features.
    
    Args:
        df_features: DataFrame with features and labels
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Trained classifier and performance metrics
    """
    print(f"\n{'='*80}")
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print(f"{'='*80}")
    
    # Prepare feature matrix
    feature_cols = [
        'mean_score', 'std_score', 'median_score', 'iqr_score',
        'skew', 'kurtosis', 'entropy', 'coef_variation',
        'first_half_mean', 'second_half_mean', 'temporal_diff',
        'pct_high_scores', 'pct_medium_scores', 'pct_low_scores',
        'max_score', 'score_range', 'n_proposals'
    ]
    
    X = df_features[feature_cols].fillna(0).values
    y = df_features['is_fake'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=random_state,
        n_jobs=-1
    )
    
    print("\nTraining Random Forest...")
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Baseline (max_score only)
    baseline_auc = roc_auc_score(y_test, X_test[:, feature_cols.index('max_score')])
    
    # Full model metrics
    auc_score = roc_auc_score(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"\n{'='*80}")
    print("TEST SET PERFORMANCE")
    print(f"{'='*80}")
    print(f"\nBaseline (max_score only):  AUC = {baseline_auc:.4f}")
    print(f"Temporal features model:    AUC = {auc_score:.4f}")
    print(f"Improvement:                     {(auc_score - baseline_auc):.4f} ({(auc_score - baseline_auc)*100:+.1f}%)")
    print(f"\nAverage Precision:          AP  = {ap_score:.4f}")
    print(f"Accuracy:                        {acc:.4f}")
    print(f"Balanced Accuracy:               {bal_acc:.4f}")
    print(f"Matthews Correlation:            {mcc:.4f}")
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:4d}  |  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  |  TP: {tp:4d}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE")
    print(f"{'='*80}")
    print(feature_importance.head(10).to_string(index=False))
    
    # Cross-validation
    print(f"\n{'='*80}")
    print("5-FOLD CROSS-VALIDATION")
    print(f"{'='*80}")
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"Cross-val AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Individual folds: {[f'{s:.4f}' for s in cv_scores]}")
    
    metrics = {
        'baseline_auc': baseline_auc,
        'test_auc': auc_score,
        'test_ap': ap_score,
        'test_accuracy': acc,
        'test_balanced_accuracy': bal_acc,
        'test_mcc': mcc,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'feature_importance': feature_importance,
        'feature_cols': feature_cols,
    }
    
    return clf, metrics


def save_model(clf: RandomForestClassifier, feature_cols: List[str], output_path: str):
    """Save trained classifier and feature configuration."""
    model_data = {
        'classifier': clf,
        'feature_cols': feature_cols,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {output_path}")


def predict_video(csv_path: str, clf: RandomForestClassifier, feature_cols: List[str]) -> Tuple[float, Dict]:
    """
    Predict if a video is fake using temporal features.
    
    Args:
        csv_path: Path to CSV file with boundary predictions
        clf: Trained Random Forest classifier
        feature_cols: List of feature names in correct order
        
    Returns:
        Prediction probability and extracted features
    """
    # Read scores
    df = pd.read_csv(csv_path)
    scores = df['score'].values
    
    # Extract features
    features = extract_temporal_features(scores)
    if features is None:
        return 0.0, {}
    
    # Prepare feature vector
    X = np.array([[features.get(col, 0) for col in feature_cols]])
    
    # Predict
    proba = clf.predict_proba(X)[0, 1]
    
    return proba, features


def main():
    parser = argparse.ArgumentParser(description='Temporal Feature Extraction for Deepfake Detection')
    parser.add_argument('--config', type=str, required=True, help='Path to config TOML file')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze features, do not train')
    parser.add_argument('--output-model', type=str, default='output/temporal_classifier.pkl',
                        help='Path to save trained model')
    args = parser.parse_args()
    
    # Load config
    config = toml.load(args.config)
    model_name = config['name']
    csv_dir = f'output/results/{model_name}'
    
    print(f"{'='*80}")
    print("TEMPORAL FEATURE EXTRACTION & CLASSIFICATION")
    print(f"{'='*80}")
    print(f"Config: {args.config}")
    print(f"Model: {model_name}")
    print(f"CSV directory: {csv_dir}")
    
    # Determine dataset type and load metadata
    if 'fakeavceleb' in model_name.lower() or 'fakeavceleb' in args.config.lower():
        print("\nDetected FakeAVCeleb dataset")
        data_root = 'dataset/multimodal-time-localisation/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2'
        dm = FakeAVCelebDataModule(
            root=data_root,
            frame_padding=config.get('num_frames', 16),
            require_match_scores=False,
            max_duration=config.get('max_duration', 10),
            batch_size=1,
            num_workers=1,
            get_meta_attr=Batfd.get_meta_attr,
            return_file_name=True
        )
        dm.setup()
        metadata = dm.test_dataset.metadata
        dataset_name = "FakeAVCeleb"
    else:
        print("\nDetected LAV-DF dataset")
        data_root = 'dataset/LAV-DF'
        dm = LavdfDataModule(
            root=data_root,
            frame_padding=config.get('num_frames', 16),
            require_match_scores=False,
            max_duration=config.get('max_duration', 10),
            batch_size=1,
            num_workers=1,
            get_meta_attr=Batfd.get_meta_attr,
            return_file_name=True
        )
        dm.setup()
        metadata = dm.test_dataset.metadata
        dataset_name = "LAV-DF"
    
    # Extract features
    df_features = load_dataset_features(csv_dir, metadata, dataset_name)
    
    # Analyze features
    analyze_features(df_features)
    
    if args.analyze_only:
        print("\nAnalysis complete (--analyze-only mode)")
        return
    
    # Train classifier
    clf, metrics = train_classifier(df_features)
    
    # Demographic analysis
    analyze_by_demographics(df_features, clf, metrics['feature_cols'])
    
    # Save model
    save_model(clf, metrics['feature_cols'], args.output_model)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset:              {dataset_name}")
    print(f"Videos processed:     {len(df_features)}")
    print(f"Baseline AUC:         {metrics['baseline_auc']:.4f}")
    print(f"Temporal model AUC:   {metrics['test_auc']:.4f}")
    print(f"Improvement:          {(metrics['test_auc'] - metrics['baseline_auc'])*100:+.1f}%")
    print(f"Model saved:          {args.output_model}")


if __name__ == '__main__':
    main()
