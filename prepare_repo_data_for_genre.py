# genre_integration/prepare_data_for_genre.py
"""
Export COMPASS data from recourse_benchmarks for GenRe usage.
Ensures 100% data format consistency.

Usage:
    python prepare_data_for_genre.py

Output:
    - data/compass_from_repo.pkl: Processed data
    - data/compass_metadata.json: Data metadata
"""

import sys
import os
from pathlib import Path
import pickle
import json

# Add recourse_benchmarks to Python path
REPO_PATH = Path(__file__).parent.parent / "recourse_benchmarks"
sys.path.insert(0, str(REPO_PATH))

import pandas as pd
import numpy as np
from data.catalog import loadData

# Define standard feature order from mlmodel_catalog.yaml
STANDARD_FEATURE_ORDER = ['x0_ord_0', 'x0_ord_1', 'x0_ord_2', 'x1', 'x2', 'x3', 'x4']



def prepare_compass_data():
    """
    Prepare COMPASS dataset from recourse_benchmarks.
    
    Returns:
        data_dict: Dictionary containing data and metadata
        metadata: Data metadata dictionary
    """
    print("=" * 60)
    print("Preparing COMPASS Data from Recourse Benchmarks for GenRe")
    print("=" * 60)
    
    # 1. Load data using repo's method
    print("\n[1/5] Loading data...")
    dataset = loadData.loadDataset("compass", return_one_hot=True, load_from_cache=True)
    X_train, X_test, y_train, y_test = dataset.getTrainTestSplit(preprocessing="normalize")
    
    print(f"  Train set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Features: {X_train.columns.tolist()}")

    # Reorder features to match catalog
    if set(X_train.columns) == set(STANDARD_FEATURE_ORDER):
        print(f"  Reordering features to standard order...")
        X_train = X_train[STANDARD_FEATURE_ORDER]
        X_test = X_test[STANDARD_FEATURE_ORDER]
        print(f"  Features (ordered): {X_train.columns.tolist()}")
    else:
        print(f"  ⚠️  WARNING: Feature mismatch detected!")
        print(f"    Expected: {STANDARD_FEATURE_ORDER}")
        print(f"    Got: {X_train.columns.tolist()}")
        missing = set(STANDARD_FEATURE_ORDER) - set(X_train.columns)
        extra = set(X_train.columns) - set(STANDARD_FEATURE_ORDER)
        if missing:
            print(f"    Missing features: {missing}")
        if extra:
            print(f"    Extra features: {extra}")
    
    # 2. Combine into full dataset
    print("\n[2/5] Merging train and test sets...")
    X_full = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_full = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
    
    print(f"  Full dataset: {X_full.shape}")
    print(f"  Class distribution: {y_full.value_counts().to_dict()}")
    
    # 3. Extract metadata
    print("\n[3/5] Extracting metadata...")
    metadata = {
        'dataset_name': 'compass',
        'n_samples': len(X_full),
        'n_features': X_full.shape[1],
        'feature_names': X_full.columns.tolist(),
        'continuous_features': dataset.getRealBasedAttributeNames("kurz").tolist(),
        'categorical_features': dataset.getBinaryAttributeNames("kurz").tolist(),
        'immutable_features': dataset.getNonMutableAttributeNames("kurz").tolist(),
        'target_name': 'y',
        'data_range': {
            'min': float(X_full.min().min()),
            'max': float(X_full.max().max())
        },
        'train_test_split': {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_indices': list(range(len(X_train))),
            'test_indices': list(range(len(X_train), len(X_full)))
        }
    }
    
    print(f"  Continuous features: {metadata['continuous_features']}")
    print(f"  Categorical features: {metadata['categorical_features']}")
    print(f"  Immutable features: {metadata['immutable_features']}")
    
    # 4. Prepare data dictionary
    print("\n[4/5] Preparing data dictionary...")
    data_dict = {
        'X': X_full.values,
        'y': y_full.values,
        'X_df': X_full,  # Keep DataFrame format
        'y_series': y_full,
        'metadata': metadata
    }
    
    # 5. Save data
    print("\n[5/5] Saving data...")
    
    # Create output directory
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    # Save data
    data_path = output_dir / "compass_from_repo.pkl"
    with open(data_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"  ✓ Data saved: {data_path}")
    
    # Save metadata
    metadata_path = output_dir / "compass_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved: {metadata_path}")
    
    # 5.5. Save as CSV for GenRe (GenRe expects CSV format)
    print("\n[6/6] Preparing CSV files for GenRe...")

    # Create GenRe-compatible directory structure
    genre_data_dir = Path(__file__).parent.parent / "genre" / "datasets" / "compas"
    genre_data_dir.mkdir(parents=True, exist_ok=True)

    # Save train set
    train_df = X_train.copy()
    train_df['score'] = y_train  # GenRe expects 'score' column for COMPASS
    train_csv_path = genre_data_dir / "train.csv"
    train_df.to_csv(train_csv_path, index=False)
    print(f"  ✓ Train CSV saved: {train_csv_path}")

    # Save test set
    test_df = X_test.copy()
    test_df['score'] = y_test
    test_csv_path = genre_data_dir / "test.csv"
    test_df.to_csv(test_csv_path, index=False)
    print(f"  ✓ Test CSV saved: {test_csv_path}")

    print(f"\n✓ GenRe can now load data with: --dataset compas-all")


    # 6. Verification
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"\nData file: {data_path}")
    print(f"Metadata file: {metadata_path}")
    print(f"\nData Statistics:")
    print(f"  - Samples: {metadata['n_samples']}")
    print(f"  - Features: {metadata['n_features']}")
    print(f"  - Data range: [{metadata['data_range']['min']:.4f}, {metadata['data_range']['max']:.4f}]")
    print(f"  - Class 0: {sum(y_full == 0)} samples")
    print(f"  - Class 1: {sum(y_full == 1)} samples")
    
    return data_dict, metadata


if __name__ == "__main__":
    try:
        data_dict, metadata = prepare_compass_data()
        print("\n✅ Data preparation successful!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)