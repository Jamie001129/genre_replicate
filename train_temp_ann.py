# genre_integration/train_temp_ann.py
"""
Train a temporary 3-layer ANN for GenRe training pair generation.

This ANN is ONLY used to:
1. Generate training pairs (x-, x+) for GenRe Transformer
2. NOT used in final integration (repo's ANN will be used)

Usage:
    python train_temp_ann.py

Output:
    - temp_models/temp_ann.pth: Trained ANN weights
    - temp_models/train_log.txt: Training logs
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


class ThreeLayerANN(nn.Module):
    """
    3-layer ANN matching GenRe paper architecture.
    
    Architecture:
        Input → FC(10) → ReLU → FC(10) → ReLU → FC(10) → ReLU → FC(2) → Softmax
    """
    
    def __init__(self, n_features, n_hidden=10):
        """
        Initialize 3-layer ANN.
        
        Args:
            n_features: Number of input features
            n_hidden: Number of neurons per hidden layer (default: 10)
        """
        super(ThreeLayerANN, self).__init__()
        
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, 2)  # Binary classification
        
    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1)
    
    def predict_proba(self, x):
        """
        Predict probabilities (compatible with sklearn interface).
        
        Args:
            x: Input features (numpy array or tensor)
            
        Returns:
            Probabilities for both classes (N x 2)
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            return self.forward(x).numpy()
    
    def predict(self, x):
        """
        Predict class labels.
        
        Args:
            x: Input features
            
        Returns:
            Class predictions (0 or 1)
        """
        proba = self.predict_proba(x)
        return (proba[:, 1] > 0.5).astype(int)


def train_epoch(model, optimizer, train_loader, device):
    """
    Train for one epoch.
    
    Args:
        model: ANN model
        optimizer: Optimizer
        train_loader: DataLoader for training data
        device: Device to train on
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device).long()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)
        
        # Compute loss
        loss = F.cross_entropy(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def eval_epoch(model, test_loader, device):
    """
    Evaluate model on test set.
    
    Args:
        model: ANN model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        
    Returns:
        Dictionary with metrics (loss, accuracy, precision, recall)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).long()
            
            # Forward pass
            outputs = model(batch_X)
            loss = F.cross_entropy(outputs, batch_y)
            
            # Predictions
            _, predicted = torch.max(outputs, 1)
            
            # Accumulate
            total_loss += loss.item()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Compute metrics
    accuracy = correct / total
    
    # Compute precision and recall for positive class
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    true_positives = ((all_preds == 1) & (all_labels == 1)).sum()
    false_positives = ((all_preds == 1) & (all_labels == 0)).sum()
    false_negatives = ((all_preds == 0) & (all_labels == 1)).sum()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'loss': total_loss / len(test_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_training_curves(train_losses, test_metrics, output_dir):
    """
    Plot training curves.
    
    Args:
        train_losses: List of training losses per epoch
        test_metrics: List of test metric dicts per epoch
        output_dir: Directory to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot([m['loss'] for m in test_metrics], label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot([m['accuracy'] for m in test_metrics], label='Accuracy')
    axes[1].plot([m['precision'] for m in test_metrics], label='Precision')
    axes[1].plot([m['recall'] for m in test_metrics], label='Recall')
    axes[1].plot([m['f1'] for m in test_metrics], label='F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Test Metrics')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ann_training_curves.png', dpi=150)
    plt.close()
    
    print(f"  ✓ Training curves saved to {output_dir / 'ann_training_curves.png'}")


def main():
    """Main training function."""
    
    print("=" * 60)
    print("Training Temporary ANN for GenRe")
    print("=" * 60)
    
    # Configuration
    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    SEED = 42
    
    # Set random seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Create output directory
    output_dir = Path(__file__).parent / "temp_models"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
    data_path = Path(__file__).parent / "data" / "compass_from_repo.pkl"
    
    if not data_path.exists():
        print(f"❌ Error: {data_path} not found!")
        print("Please run prepare_data_for_genre.py first.")
        sys.exit(1)
    
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    metadata = data_dict['metadata']
    X_full = data_dict['X']
    y_full = data_dict['y']
    
    # Split train/test
    train_size = metadata['train_test_split']['train_size']
    X_train = X_full[:train_size]
    y_train = y_full[:train_size]
    X_test = X_full[train_size:]
    y_test = y_full[train_size:]
    
    print(f"  ✓ Train: {len(X_train)} samples")
    print(f"  ✓ Test: {len(X_test)} samples")
    print(f"  ✓ Features: {X_train.shape[1]}")
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print("\n[2/5] Initializing model...")
    n_features = X_train.shape[1]
    model = ThreeLayerANN(n_features, n_hidden=10).to(device)
    
    print(f"  ✓ Model: 3-layer ANN")
    print(f"  ✓ Architecture: {n_features} → 10 → 10 → 10 → 2")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\n[3/5] Training for {EPOCHS} epochs...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    train_losses = []
    test_metrics_list = []
    
    best_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, optimizer, train_loader, device)
        train_losses.append(train_loss)
        
        # Evaluate
        test_metrics = eval_epoch(model, test_loader, device)
        test_metrics_list.append(test_metrics)
        
        # Save best model
        if test_metrics['accuracy'] > best_accuracy:
            best_accuracy = test_metrics['accuracy']
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}: "
                  f"train_loss={train_loss:.4f}, "
                  f"test_acc={test_metrics['accuracy']:.4f}, "
                  f"test_f1={test_metrics['f1']:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n[4/5] Final evaluation...")
    final_metrics = eval_epoch(model, test_loader, device)
    
    print(f"  ✓ Test Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  ✓ Test Precision: {final_metrics['precision']:.4f}")
    print(f"  ✓ Test Recall:    {final_metrics['recall']:.4f}")
    print(f"  ✓ Test F1:        {final_metrics['f1']:.4f}")
    
    # Save model
    print("\n[5/5] Saving model...")
    
    model_path = output_dir / "temp_ann.pth"
    torch.save({
        'model_state_dict': best_model_state,
        'n_features': n_features,
        'n_hidden': 10,
        'final_metrics': final_metrics,
        'feature_names': metadata['feature_names']
    }, model_path)
    
    print(f"  ✓ Model saved: {model_path}")
    
    # Save training log
    log_path = output_dir / "ann_train_log.txt"
    with open(log_path, 'w') as f:
        f.write("Training Configuration\n")
        f.write("=" * 40 + "\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Architecture: {n_features} - 10 - 10 - 10 - 2\n")
        f.write("\n")
        f.write("Final Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Test Accuracy:  {final_metrics['accuracy']:.4f}\n")
        f.write(f"Test Precision: {final_metrics['precision']:.4f}\n")
        f.write(f"Test Recall:    {final_metrics['recall']:.4f}\n")
        f.write(f"Test F1:        {final_metrics['f1']:.4f}\n")
    
    print(f"  ✓ Log saved: {log_path}")
    
    # Plot curves
    plot_training_curves(train_losses, test_metrics_list, output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. Model:  {model_path}")
    print(f"  2. Log:    {log_path}")
    print(f"  3. Curves: {output_dir / 'ann_training_curves.png'}")
    print(f"\nFinal Test Accuracy: {final_metrics['accuracy']:.2%}")
    print("\n✅ Ready for next step: generate_pairs.py")


if __name__ == "__main__":
    main()