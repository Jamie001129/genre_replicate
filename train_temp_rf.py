# genre_integration/train_temp_rf.py
"""
Train a Random Forest using the same data as train_temp_ann.py.
Much simpler model — used only for comparison or debugging.

Usage:
    python train_temp_rf.py

Output:
    - temp_models/temp_rf.pkl: trained RF model
    - temp_models/rf_log.txt: performance log
"""

import sys
import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    print("=" * 60)
    print("Training Random Forest (sklearn)")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading data...")

    data_path = Path(__file__).parent / "data" / "compass_from_repo.pkl"
    if not data_path.exists():
        print(f"❌ Error: {data_path} not found!")
        print("Please run prepare_data_for_genre.py first.")
        sys.exit(1)

    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)

    metadata = data_dict["metadata"]
    X_full = data_dict["X"]
    y_full = data_dict["y"]

    # same split as ANN
    train_size = metadata["train_test_split"]["train_size"]
    X_train = X_full[:train_size]
    y_train = y_full[:train_size]
    X_test = X_full[train_size:]
    y_test = y_full[train_size:]

    print(f"  ✓ Train: {len(X_train)} samples")
    print(f"  ✓ Test:  {len(X_test)} samples")
    print(f"  ✓ Features: {X_train.shape[1]}")

    # Train RF
    print("\n[2/3] Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    print("\n[3/3] Evaluating model...")

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"  ✓ Accuracy:  {acc:.4f}")
    print(f"  ✓ Precision: {prec:.4f}")
    print(f"  ✓ Recall:    {rec:.4f}")
    print(f"  ✓ F1:        {f1:.4f}")

    # Save model
    output_dir = Path(__file__).parent / "temp_models"
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "temp_rf.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "metadata": metadata,
            "final_metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1
            }
        }, f)

    print(f"\n  ✓ Model saved to: {model_path}")

    # Save log
    log_path = output_dir / "rf_log.txt"
    with open(log_path, "w") as f:
        f.write("Random Forest Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Test Accuracy:  {acc:.4f}\n")
        f.write(f"Test Precision: {prec:.4f}\n")
        f.write(f"Test Recall:    {rec:.4f}\n")
        f.write(f"Test F1:        {f1:.4f}\n")
    print(f"  ✓ Log saved: {log_path}")

    print("\nTraining Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
