# genre_integration/generate_pairs.py
"""
Generate training pairs for GenRe Transformer.

This script:
1. Loads trained temp_ann
2. Identifies negative instances needing recourse (D-)
3. Identifies valid positive instances (D+)
4. Generates pairs (x-, x+) using weighted sampling based on distance

Algorithm (from GenRe paper):
- For each x- in D-:
  - Compute L1 distance to all x+ in D+
  - Assign weights: w(x+) = exp(-λ * distance)
  - Sample top-K closest x+ with probability proportional to weights

Usage:
    python generate_pairs.py

Output:
    - temp_models/training_pairs.pkl: Training pairs for Transformer
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from tqdm import tqdm


class ThreeLayerANN(nn.Module):
    """Same architecture as train_temp_ann.py"""
    
    def __init__(self, n_features, n_hidden=10):
        super(ThreeLayerANN, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1)


def load_temp_ann(model_path, device):
    """
    Load trained temporary ANN.
    
    Args:
        model_path: Path to temp_ann.pth
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    n_features = checkpoint['n_features']
    n_hidden = checkpoint['n_hidden']
    
    model = ThreeLayerANN(n_features, n_hidden)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def predict_probabilities(model, X, device, batch_size=1024):
    """
    Predict probabilities for large dataset in batches.
    
    Args:
        model: Trained ANN
        X: Features (numpy array)
        device: Device
        batch_size: Batch size for prediction
        
    Returns:
        Probabilities for positive class (N,)
    """
    model.eval()
    all_probs = []
    
    X_tensor = torch.from_numpy(X).float()
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            outputs = model(batch)
            probs = outputs[:, 1].cpu().numpy()  # Probability of positive class
            all_probs.append(probs)
    
    return np.concatenate(all_probs)


def generate_training_pairs(X_train, y_train, model, device, 
                            lambda_param=5.0, k=100, gamma=0.7):
    """
    Generate training pairs for GenRe Transformer.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model: Trained ANN classifier
        device: Device
        lambda_param: Distance weight parameter (default: 5.0 from paper)
        k: Number of positive examples to sample per negative (default: 100)
        gamma: Validity threshold for positive examples (default: 0.7)
        
    Returns:
        Dictionary containing training pairs and metadata
    """
    print("\n[1/4] Predicting on training data...")
    
    # Get predictions
    pred_probs = predict_probabilities(model, X_train, device)
    
    print(f"  Prediction distribution:")
    print(f"    Mean probability: {pred_probs.mean():.3f}")
    print(f"    Std:  {pred_probs.std():.3f}")
    
    # Identify D- (negative instances needing recourse)
    # D- = {x | h(x) <= 0.5 AND y = 0}
    mask_d_minus = (pred_probs <= 0.5) & (y_train == 0)
    D_minus_X = X_train[mask_d_minus]
    D_minus_y = y_train[mask_d_minus]
    
    # Identify D+ (valid positive instances)
    # D+ = {x | h(x) > gamma AND y = 1}
    mask_d_plus = (pred_probs > gamma) & (y_train == 1)
    D_plus_X = X_train[mask_d_plus]
    D_plus_y = y_train[mask_d_plus]
    
    print(f"\n[2/4] Identified instances:")
    print(f"  D- (need recourse): {len(D_minus_X)} instances")
    print(f"  D+ (valid positive): {len(D_plus_X)} instances")
    
    if len(D_minus_X) == 0:
        raise ValueError("No negative instances found! Check model predictions.")
    
    if len(D_plus_X) == 0:
        raise ValueError("No valid positive instances found! Try lowering gamma.")
    
    # Generate pairs
    print(f"\n[3/4] Generating pairs (λ={lambda_param}, k={k})...")
    
    pairs = []
    
    # Compute distances (using L1 distance as in paper)
    print(f"  Computing distances...")
    distances = cdist(D_minus_X, D_plus_X, metric='cityblock')  # L1 distance
    print(f"  Distance matrix shape: {distances.shape}")
    
    # For each negative instance
    for i in tqdm(range(len(D_minus_X)), desc="  Generating pairs"):
        x_minus = D_minus_X[i]
        
        # Get distances to all positive instances
        dists = distances[i]
        
        # Compute weights: w = exp(-λ * distance)
        weights = np.exp(-lambda_param * dists)
        
        # Normalize to probabilities
        probs = weights / weights.sum()
        
        # Select top-K indices (highest probability = lowest distance)
        top_k_indices = np.argsort(probs)[-k:]
        
        # Create pairs with selected positive instances
        for idx in top_k_indices:
            x_plus = D_plus_X[idx]
            
            pairs.append({
                'x_minus': x_minus,
                'x_plus': x_plus,
                'y_minus': D_minus_y[i],
                'y_plus': D_plus_y[idx],
                'distance': dists[idx],
                'weight': probs[idx]
            })
    
    print(f"  ✓ Generated {len(pairs)} training pairs")
    
    # Statistics
    print(f"\n[4/4] Pair statistics:")
    distances_in_pairs = [p['distance'] for p in pairs]
    print(f"  Distance statistics:")
    print(f"    Mean: {np.mean(distances_in_pairs):.3f}")
    print(f"    Std:  {np.std(distances_in_pairs):.3f}")
    print(f"    Min:  {np.min(distances_in_pairs):.3f}")
    print(f"    Max:  {np.max(distances_in_pairs):.3f}")
    
    # Prepare output
    pairs_data = {
        'pairs': pairs,
        'metadata': {
            'n_pairs': len(pairs),
            'n_d_minus': len(D_minus_X),
            'n_d_plus': len(D_plus_X),
            'lambda': lambda_param,
            'k': k,
            'gamma': gamma,
            'distance_stats': {
                'mean': float(np.mean(distances_in_pairs)),
                'std': float(np.std(distances_in_pairs)),
                'min': float(np.min(distances_in_pairs)),
                'max': float(np.max(distances_in_pairs))
            }
        }
    }
    
    return pairs_data


def main():
    """Main function."""
    
    print("=" * 60)
    print("Generating Training Pairs for GenRe Transformer")
    print("=" * 60)
    
    # Configuration
    LAMBDA_PARAM = 5.0  # Distance weight (from GenRe paper)
    TOP_K = 100  # Number of positive examples per negative
    GAMMA = 0.7  # Validity threshold for D+
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() 
                         else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\nDevice: {device}")
    
    # Paths
    base_dir = Path(__file__).parent
    model_path = base_dir / "temp_models" / "temp_ann.pth"
    data_path = base_dir / "data" / "compass_from_repo.pkl"
    output_path = base_dir / "temp_models" / "training_pairs.pkl"
    
    # Check files exist
    if not model_path.exists():
        print(f"❌ Error: {model_path} not found!")
        print("Please run train_temp_ann.py first.")
        sys.exit(1)
    
    if not data_path.exists():
        print(f"❌ Error: {data_path} not found!")
        print("Please run prepare_data_for_genre.py first.")
        sys.exit(1)
    
    # Load model
    print("\nLoading trained ANN...")
    model, checkpoint = load_temp_ann(model_path, device)
    print(f"  ✓ Model loaded")
    print(f"  ✓ Test accuracy: {checkpoint['final_metrics']['accuracy']:.2%}")
    
    # Load data
    print("\nLoading data...")
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    metadata = data_dict['metadata']
    X_full = data_dict['X']
    y_full = data_dict['y']
    
    # Use only training split for generating pairs
    train_size = metadata['train_test_split']['train_size']
    X_train = X_full[:train_size]
    y_train = y_full[:train_size]
    
    print(f"  ✓ Using training split: {len(X_train)} samples")
    print(f"  ✓ Features: {X_train.shape[1]}")
    
    # Generate pairs
    pairs_data = generate_training_pairs(
        X_train, y_train, model, device,
        lambda_param=LAMBDA_PARAM,
        k=TOP_K,
        gamma=GAMMA
    )
    
    # Save pairs
    print(f"\nSaving training pairs...")
    with open(output_path, 'wb') as f:
        pickle.dump(pairs_data, f)
    
    print(f"  ✓ Saved to: {output_path}")
    print(f"  ✓ File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Summary
    print("\n" + "=" * 60)
    print("Pair Generation Complete!")
    print("=" * 60)
    print(f"\nGenerated {pairs_data['metadata']['n_pairs']:,} training pairs")
    print(f"  D- instances: {pairs_data['metadata']['n_d_minus']}")
    print(f"  D+ instances: {pairs_data['metadata']['n_d_plus']}")
    print(f"  Pairs per D- instance: {TOP_K}")
    print(f"\nParameters:")
    print(f"  λ (lambda): {LAMBDA_PARAM}")
    print(f"  γ (gamma):  {GAMMA}")
    print(f"  Top-K:      {TOP_K}")
    print(f"\nOutput: {output_path}")
    print("\n✅ Ready for next step: train_transformer.py")


if __name__ == "__main__":
    main()