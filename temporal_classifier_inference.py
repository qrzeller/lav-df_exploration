"""
Temporal Classifier Inference Script

Use a trained temporal feature classifier to predict deepfakes from boundary CSV files.

Usage:
    # Single video prediction
    python temporal_classifier_inference.py --model output/temporal_classifier.pkl \
        --csv output/results/batfd_default/000001.csv
    
    # Batch prediction on all CSVs
    python temporal_classifier_inference.py --model output/temporal_classifier.pkl \
        --csv-dir output/results/batfd_default/ --output predictions.csv
"""

import argparse
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from temporal_feature_extraction import extract_temporal_features


def load_model(model_path: str):
    """Load trained temporal classifier."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    clf = model_data['classifier']
    feature_cols = model_data['feature_cols']
    
    print(f"Loaded model from: {model_path}")
    print(f"Feature count: {len(feature_cols)}")
    
    return clf, feature_cols


def predict_single_video(
    csv_path: str,
    clf,
    feature_cols: List[str]
) -> Dict:
    """
    Predict if a video is fake from its boundary CSV.
    
    Args:
        csv_path: Path to CSV with boundary predictions
        clf: Trained classifier
        feature_cols: Feature names in correct order
        
    Returns:
        Dictionary with prediction and features
    """
    # Read scores
    try:
        df = pd.read_csv(csv_path)
        scores = df['score'].values
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None
    
    # Extract features
    features = extract_temporal_features(scores)
    if features is None:
        return None
    
    # Prepare feature vector
    X = np.array([[features.get(col, 0) for col in feature_cols]])
    
    # Predict
    proba = clf.predict_proba(X)[0, 1]
    pred_label = 'FAKE' if proba >= 0.5 else 'REAL'
    
    result = {
        'file': os.path.basename(csv_path).replace('.csv', '.mp4'),
        'probability_fake': proba,
        'prediction': pred_label,
        'confidence': max(proba, 1 - proba),
    }
    
    # Add key features
    result.update({
        'max_score': features['max_score'],
        'std_score': features['std_score'],
        'entropy': features['entropy'],
        'temporal_diff': features['temporal_diff'],
    })
    
    return result


def predict_batch(
    csv_dir: str,
    clf,
    feature_cols: List[str],
    output_path: str = None
) -> pd.DataFrame:
    """
    Predict on all videos in a directory.
    
    Args:
        csv_dir: Directory containing CSV files
        clf: Trained classifier
        feature_cols: Feature names in correct order
        output_path: Path to save predictions (optional)
        
    Returns:
        DataFrame with predictions
    """
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files")
    
    results = []
    for csv_file in tqdm(csv_files, desc="Predicting"):
        csv_path = os.path.join(csv_dir, csv_file)
        result = predict_single_video(csv_path, clf, feature_cols)
        if result is not None:
            results.append(result)
    
    df_results = pd.DataFrame(results)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("PREDICTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total videos: {len(df_results)}")
    print(f"Predicted FAKE: {(df_results['prediction'] == 'FAKE').sum()} ({(df_results['prediction'] == 'FAKE').sum() / len(df_results) * 100:.1f}%)")
    print(f"Predicted REAL: {(df_results['prediction'] == 'REAL').sum()} ({(df_results['prediction'] == 'REAL').sum() / len(df_results) * 100:.1f}%)")
    print(f"\nAverage confidence: {df_results['confidence'].mean():.3f}")
    print(f"Min confidence: {df_results['confidence'].min():.3f}")
    print(f"Max confidence: {df_results['confidence'].max():.3f}")
    
    # Distribution of probabilities
    print(f"\nProbability distribution:")
    print(f"  0.0-0.1: {((df_results['probability_fake'] >= 0.0) & (df_results['probability_fake'] < 0.1)).sum()}")
    print(f"  0.1-0.3: {((df_results['probability_fake'] >= 0.1) & (df_results['probability_fake'] < 0.3)).sum()}")
    print(f"  0.3-0.5: {((df_results['probability_fake'] >= 0.3) & (df_results['probability_fake'] < 0.5)).sum()}")
    print(f"  0.5-0.7: {((df_results['probability_fake'] >= 0.5) & (df_results['probability_fake'] < 0.7)).sum()}")
    print(f"  0.7-0.9: {((df_results['probability_fake'] >= 0.7) & (df_results['probability_fake'] < 0.9)).sum()}")
    print(f"  0.9-1.0: {((df_results['probability_fake'] >= 0.9) & (df_results['probability_fake'] <= 1.0)).sum()}")
    
    if output_path:
        df_results.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
    
    return df_results


def main():
    parser = argparse.ArgumentParser(description='Temporal Classifier Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pkl)')
    parser.add_argument('--csv', type=str, help='Path to single CSV file')
    parser.add_argument('--csv-dir', type=str, help='Directory containing CSV files')
    parser.add_argument('--output', type=str, help='Path to save predictions (for batch mode)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold (default: 0.5)')
    args = parser.parse_args()
    
    if not args.csv and not args.csv_dir:
        parser.error("Must specify either --csv or --csv-dir")
    
    # Load model
    clf, feature_cols = load_model(args.model)
    
    # Single video prediction
    if args.csv:
        print(f"\n{'='*80}")
        print("SINGLE VIDEO PREDICTION")
        print(f"{'='*80}")
        print(f"CSV: {args.csv}")
        
        result = predict_single_video(args.csv, clf, feature_cols)
        
        if result is None:
            print("Error: Could not process video")
            return
        
        print(f"\n{'='*80}")
        print("RESULT")
        print(f"{'='*80}")
        print(f"Video: {result['file']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability (FAKE): {result['probability_fake']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"\nKey Features:")
        print(f"  Max Score: {result['max_score']:.4f}")
        print(f"  Std Score: {result['std_score']:.4f}")
        print(f"  Entropy: {result['entropy']:.4f}")
        print(f"  Temporal Diff: {result['temporal_diff']:.4f}")
    
    # Batch prediction
    elif args.csv_dir:
        print(f"\n{'='*80}")
        print("BATCH PREDICTION")
        print(f"{'='*80}")
        print(f"Directory: {args.csv_dir}")
        
        df_results = predict_batch(args.csv_dir, clf, feature_cols, args.output)
        
        # Show top 10 most confident FAKE predictions
        print(f"\n{'='*80}")
        print("TOP 10 MOST CONFIDENT FAKE PREDICTIONS")
        print(f"{'='*80}")
        fake_preds = df_results[df_results['prediction'] == 'FAKE'].sort_values('probability_fake', ascending=False)
        print(fake_preds[['file', 'probability_fake', 'confidence']].head(10).to_string(index=False))
        
        # Show top 10 most confident REAL predictions
        print(f"\n{'='*80}")
        print("TOP 10 MOST CONFIDENT REAL PREDICTIONS")
        print(f"{'='*80}")
        real_preds = df_results[df_results['prediction'] == 'REAL'].sort_values('probability_fake', ascending=True)
        print(real_preds[['file', 'probability_fake', 'confidence']].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
