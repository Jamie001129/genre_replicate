# genre_integration/train_transformer.py
"""
Train GenRe Transformer for counterfactual generation.

This is the core GenRe model that learns the conditional distribution R_θ(x+|x-).

Architecture:
- Encoder: Encodes factual x- into latent representation
- Decoder: Autoregressively generates counterfactual x+
- Binned Output: Discretized output for better generation quality

Training:
- Input: Training pairs (x-, x+) from generate_pairs.py
- Loss: Negative log-likelihood + validity + proximity
- Optimizer: Adam with learning rate scheduling

Usage:
    python train_transformer.py

Output:
    - final_model/genre_transformer.pth: Trained Transformer
    - final_model/training_log.txt: Training logs
    - final_model/training_curves.png: Loss curves
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


# ============================================================================
# Dataset
# ============================================================================

class PairDataset(Dataset):
    """
    Dataset for training pairs.
    """
    
    def __init__(self, pairs):
        """
        Initialize dataset.
        
        Args:
            pairs: List of pair dictionaries from generate_pairs.py
        """
        self.pairs = pairs
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            'x_minus': torch.from_numpy(pair['x_minus']).float(),
            'x_plus': torch.from_numpy(pair['x_plus']).float(),
            'distance': pair['distance'],
            'weight': pair['weight']
        }


# ============================================================================
# Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.
    """
    
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


# ============================================================================
# Binned Output Layer
# ============================================================================

class BinnedOutputLayer(nn.Module):
    """
    Binned output layer for discretized generation.
    
    Instead of directly predicting continuous values, we discretize
    the [0, 1] range into bins and predict bin probabilities.
    This improves generation quality.
    """
    
    def __init__(self, d_model, n_features, n_bins=50):
        """
        Initialize binned output layer.
        
        Args:
            d_model: Hidden dimension
            n_features: Number of features to predict
            n_bins: Number of bins per feature
        """
        super(BinnedOutputLayer, self).__init__()
        
        self.n_features = n_features
        self.n_bins = n_bins
        
        # Separate linear layer for each feature
        self.feature_heads = nn.ModuleList([
            nn.Linear(d_model, n_bins) for _ in range(n_features)
        ])
        
        # Bin centers for converting predictions back to continuous values
        self.register_buffer('bin_centers', 
                           torch.linspace(0, 1, n_bins))
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            List of logits for each feature: [(batch_size, seq_len, n_bins)] * n_features
        """
        outputs = []
        for head in self.feature_heads:
            logits = head(x)  # (batch_size, seq_len, n_bins)
            outputs.append(logits)
        return outputs
    
    def sample(self, x, temperature=1.0):
        """
        Sample from binned distribution.
        
        Args:
            x: (batch_size, seq_len, d_model)
            temperature: Sampling temperature
            
        Returns:
            Sampled continuous values: (batch_size, seq_len, n_features)
        """
        outputs = self.forward(x)
        samples = []
        
        for logits in outputs:
            # Apply temperature
            logits = logits / temperature
            
            # Sample bin
            probs = F.softmax(logits, dim=-1)
            bin_indices = torch.multinomial(
                probs.view(-1, self.n_bins), 
                num_samples=1
            ).view(logits.shape[:-1])
            
            # Convert bin index to continuous value
            values = self.bin_centers[bin_indices]
            samples.append(values)
        
        # Stack: (batch_size, seq_len, n_features)
        return torch.stack(samples, dim=-1)


# ============================================================================
# GenRe Transformer
# ============================================================================

class GenReTransformer(nn.Module):
    """
    GenRe Transformer for counterfactual generation.
    
    Architecture:
    - Encoder: Encodes factual x-
    - Decoder: Autoregressively generates counterfactual x+
    - Binned output: Discretized predictions
    """
    
    def __init__(self, n_features, d_model=32, nhead=4, 
                 num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=128, dropout=0.1, n_bins=50):
        """
        Initialize GenRe Transformer.
        
        Args:
            n_features: Number of input features
            d_model: Hidden dimension (default: 32)
            nhead: Number of attention heads (default: 4)
            num_encoder_layers: Number of encoder layers (default: 4)
            num_decoder_layers: Number of decoder layers (default: 4)
            dim_feedforward: Feedforward dimension (default: 128)
            dropout: Dropout rate (default: 0.1)
            n_bins: Number of bins for output (default: 50)
        """
        super(GenReTransformer, self).__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.n_bins = n_bins
        
        # Input embeddings
        self.src_embedding = nn.Linear(n_features, d_model)
        self.tgt_embedding = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = BinnedOutputLayer(d_model, n_features, n_bins)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, tgt_mask=None):
        """
        Forward pass.
        
        Args:
            src: Source (factual x-): (batch_size, 1, n_features)
            tgt: Target (counterfactual x+): (batch_size, seq_len, n_features)
            tgt_mask: Target mask for autoregressive generation
            
        Returns:
            List of logits for each feature
        """
        # Embed
        src = self.src_embedding(src)  # (batch_size, 1, d_model)
        tgt = self.tgt_embedding(tgt)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        
        # Output layer
        logits = self.output_layer(output)
        
        return logits
    
    def generate(self, src, temperature=1.0, max_len=None):
        """
        Generate counterfactual autoregressively.
        
        Args:
            src: Source (factual): (batch_size, n_features)
            temperature: Sampling temperature
            max_len: Maximum generation length (default: n_features)
            
        Returns:
            Generated counterfactual: (batch_size, n_features)
        """
        if max_len is None:
            max_len = self.n_features
        
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            device = src.device
            
            # Prepare source: (batch_size, 1, n_features)
            src = src.unsqueeze(1)
            
            # Start with zeros (will be filled autoregressively)
            generated = torch.zeros(batch_size, 1, self.n_features).to(device)
            
            # Generate feature by feature
            for t in range(max_len):
                # Forward pass
                logits = self.forward(src, generated)
                
                # Sample next feature values
                next_values = []
                for feat_idx in range(self.n_features):
                    feat_logits = logits[feat_idx][:, -1, :]  # Last position
                    
                    # Apply temperature and sample
                    feat_logits = feat_logits / temperature
                    probs = F.softmax(feat_logits, dim=-1)
                    bin_idx = torch.multinomial(probs, num_samples=1)
                    
                    # Convert to continuous value
                    value = self.output_layer.bin_centers[bin_idx].squeeze(-1)
                    next_values.append(value)
                
                # Stack next features: (batch_size, n_features)
                next_feat = torch.stack(next_values, dim=-1).unsqueeze(1)
                
                # Append to generated sequence
                if t < max_len - 1:
                    generated = torch.cat([generated, next_feat], dim=1)
                else:
                    # Last step: replace with final generation
                    generated = next_feat
            
            return generated.squeeze(1)


# ============================================================================
# Training Functions
# ============================================================================

def compute_loss(model, src, tgt, n_bins):
    """
    Compute training loss.
    
    Args:
        model: GenRe Transformer
        src: Source (x-): (batch_size, n_features)
        tgt: Target (x+): (batch_size, n_features)
        n_bins: Number of bins
        
    Returns:
        Total loss
    """
    batch_size = src.shape[0]
    device = src.device
    n_features = src.shape[1]
    
    # Prepare inputs
    src = src.unsqueeze(1)  # (batch_size, 1, n_features)
    
    # For decoder input, we use teacher forcing
    # Start with zeros and append target features
    decoder_input = torch.cat([
        torch.zeros(batch_size, 1, n_features).to(device),
        tgt.unsqueeze(1)
    ], dim=1)[:, :-1, :]  # Shift right
    
    # Forward pass
    logits = model(src, decoder_input)
    
    # Compute loss for each feature
    total_loss = 0.0
    
    # Convert continuous targets to bin indices
    tgt_expanded = tgt.unsqueeze(1)  # (batch_size, 1, n_features)
    
    for feat_idx in range(n_features):
        feat_logits = logits[feat_idx]  # (batch_size, seq_len, n_bins)
        feat_target = tgt_expanded[:, :, feat_idx]  # (batch_size, 1)
        
        # Find closest bin
        bin_centers = model.output_layer.bin_centers
        distances = torch.abs(feat_target.unsqueeze(-1) - bin_centers)
        target_bins = distances.argmin(dim=-1)  # (batch_size, 1)
        
        # Cross-entropy loss
        loss = F.cross_entropy(
            feat_logits.reshape(-1, n_bins),
            target_bins.reshape(-1)
        )
        total_loss += loss
    
    return total_loss / n_features


def train_epoch(model, train_loader, optimizer, device, n_bins):
    """
    Train for one epoch.
    
    Args:
        model: GenRe Transformer
        train_loader: DataLoader
        optimizer: Optimizer
        device: Device
        n_bins: Number of bins
        
    Returns:
        Average loss
    """
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        src = batch['x_minus'].to(device)
        tgt = batch['x_plus'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        loss = compute_loss(model, src, tgt, n_bins)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, device, n_bins):
    """
    Evaluate model.
    
    Args:
        model: GenRe Transformer
        val_loader: Validation DataLoader
        device: Device
        n_bins: Number of bins
        
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            src = batch['x_minus'].to(device)
            tgt = batch['x_plus'].to(device)
            
            loss = compute_loss(model, src, tgt, n_bins)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def plot_training_curves(train_losses, val_losses, output_dir):
    """Plot training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('GenRe Transformer Training', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()


# ============================================================================
# Main Training
# ============================================================================

def main():
    """Main training function."""
    
    print("=" * 60)
    print("Training GenRe Transformer")
    print("=" * 60)
    
    # Configuration
    EPOCHS = 100
    BATCH_SIZE = 512
    LEARNING_RATE = 0.0001
    VAL_SPLIT = 0.1
    
    # Model hyperparameters (from GenRe paper)
    D_MODEL = 32
    NHEAD = 4
    NUM_ENCODER_LAYERS = 8
    NUM_DECODER_LAYERS = 8
    DIM_FEEDFORWARD = 128
    DROPOUT = 0.1
    N_BINS = 50
    
    SEED = 42
    
    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() 
                         else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\nDevice: {device}")
    
    # Paths
    base_dir = Path(__file__).parent
    pairs_path = base_dir / "temp_models" / "training_pairs.pkl"
    output_dir = base_dir / "final_model"
    output_dir.mkdir(exist_ok=True)
    
    # Load training pairs
    print("\nLoading training pairs...")
    if not pairs_path.exists():
        print(f"❌ Error: {pairs_path} not found!")
        print("Please run generate_pairs.py first.")
        sys.exit(1)
    
    with open(pairs_path, 'rb') as f:
        pairs_data = pickle.load(f)
    
    pairs = pairs_data['pairs']
    n_pairs = len(pairs)
    n_features = len(pairs[0]['x_minus'])
    
    print(f"  ✓ Loaded {n_pairs:,} training pairs")
    print(f"  ✓ Features: {n_features}")
    
    # Split train/val
    n_val = int(n_pairs * VAL_SPLIT)
    n_train = n_pairs - n_val
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]
    
    print(f"  ✓ Train: {n_train:,} pairs")
    print(f"  ✓ Val:   {n_val:,} pairs")
    
    # Create datasets
    train_dataset = PairDataset(train_pairs)
    val_dataset = PairDataset(val_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=0)
    
    # Initialize model
    print("\nInitializing model...")
    model = GenReTransformer(
        n_features=n_features,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        n_bins=N_BINS
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model initialized")
    print(f"  ✓ Parameters: {n_params:,}")
    print(f"  ✓ Architecture:")
    print(f"      - d_model: {D_MODEL}")
    print(f"      - heads: {NHEAD}")
    print(f"      - encoder layers: {NUM_ENCODER_LAYERS}")
    print(f"      - decoder layers: {NUM_DECODER_LAYERS}")
    print(f"      - bins: {N_BINS}")
    
    # Optimizer with learning rate scheduling
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, N_BINS)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = evaluate(model, val_loader, device, N_BINS)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✓ New best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\n⚠️  Early stopping triggered (patience: {max_patience})")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\nFinal Evaluation...")
    final_val_loss = evaluate(model, val_loader, device, N_BINS)
    print(f"  Best val loss: {best_val_loss:.4f}")
    
    # Save model
    print("\nSaving model...")
    model_path = output_dir / "genre_transformer.pth"
    
    torch.save({
        'model_state_dict': best_model_state,
        'n_features': n_features,
        'd_model': D_MODEL,
        'nhead': NHEAD,
        'num_encoder_layers': NUM_ENCODER_LAYERS,
        'num_decoder_layers': NUM_DECODER_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD,
        'dropout': DROPOUT,
        'n_bins': N_BINS,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, model_path)
    
    print(f"  ✓ Model saved: {model_path}")
    
    # Save training log
    log_path = output_dir / "training_log.txt"
    with open(log_path, 'w') as f:
        f.write("GenRe Transformer Training Log\n")
        f.write("=" * 40 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  Epochs: {epoch}\n")
        f.write(f"  Batch size: {BATCH_SIZE}\n")
        f.write(f"  Learning rate: {LEARNING_RATE}\n")
        f.write(f"  Architecture:\n")
        f.write(f"    - d_model: {D_MODEL}\n")
        f.write(f"    - heads: {NHEAD}\n")
        f.write(f"    - encoder layers: {NUM_ENCODER_LAYERS}\n")
        f.write(f"    - decoder layers: {NUM_DECODER_LAYERS}\n")
        f.write(f"    - bins: {N_BINS}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Best val loss: {best_val_loss:.4f}\n")
        f.write(f"  Total parameters: {n_params:,}\n")
    
    print(f"  ✓ Log saved: {log_path}")
    
    # Plot curves
    plot_training_curves(train_losses, val_losses, output_dir)
    print(f"  ✓ Curves saved: {output_dir / 'training_curves.png'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {model_path}")
    print("\n✅ Ready for final step: Integration into recourse_benchmarks")


if __name__ == "__main__":
    main()