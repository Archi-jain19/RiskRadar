"""
DropGuard AI — LSTM Time-Series Dropout Predictor
===================================================
PyTorch LSTM model for sequential student behavior analysis.
Takes multi-semester student data as input sequences and predicts dropout probability.

Architecture:
    Input (seq_len, n_features)
    → LSTM (hidden=64, 2 layers, bidirectional)
    → Dropout (0.3)
    → Dense (128 → 32)
    → Sigmoid → dropout probability
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class StudentSequenceDataset(Dataset):
    """PyTorch Dataset for padded student semester sequences.
    
    Args:
        X: np.ndarray of shape (n_students, max_seq_len, n_features).
        y: np.ndarray of shape (n_students,) with binary labels.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LSTMDropoutPredictor(nn.Module):
    """Bidirectional LSTM for student dropout prediction.
    
    Architecture:
        LSTM (bidirectional, 2 layers) → Attention pooling → FC layers → Sigmoid
        
    Args:
        input_size: Number of features per time step.
        hidden_size: LSTM hidden dimension (default: 64).
        num_layers: Number of LSTM layers (default: 2).
        dropout: Dropout rate (default: 0.3).
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Attention mechanism for sequence pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Tanh()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_features).
            
        Returns:
            Dropout probability tensor of shape (batch, 1).
        """
        # LSTM output: (batch, seq_len, hidden*2)
        lstm_out, _ = self.lstm(x)
        
        # Attention-weighted pooling
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)
        
        # Classification
        return self.classifier(context)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 8,
    device: Optional[str] = None
) -> Tuple[LSTMDropoutPredictor, dict]:
    """Train the LSTM dropout predictor with early stopping.
    
    Args:
        X_train: Training sequences (n_students, seq_len, n_features).
        y_train: Training labels.
        X_val: Validation sequences.
        y_val: Validation labels.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        patience: Early stopping patience.
        device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.
        
    Returns:
        Tuple of (trained model, training history dict).
    """
    # Device selection
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    device = torch.device(device)
    print(f"  LSTM training on: {device}")
    
    # Data loaders
    train_dataset = StudentSequenceDataset(X_train, y_train)
    val_dataset = StudentSequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    n_features = X_train.shape[2]
    model = LSTMDropoutPredictor(input_size=n_features).to(device)
    
    # BCELoss — class balancing handled via SMOTE in the pipeline
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    
    print(f"  Training LSTM: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(y_batch)
        
        train_loss /= len(train_dataset)
        
        # Validate
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch).squeeze()
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * len(y_batch)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)
        
        # Compute AUC
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
        except Exception:
            val_auc = 0.5
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu()
    
    print(f"  ✓ LSTM training complete | Best Val Loss: {best_val_loss:.4f}")
    return model, history


def predict_lstm(model: LSTMDropoutPredictor, X: np.ndarray,
                 batch_size: int = 128) -> np.ndarray:
    """Get dropout probabilities from the LSTM model.
    
    Args:
        model: Trained LSTMDropoutPredictor.
        X: Input sequences (n_students, seq_len, n_features).
        batch_size: Batch size for inference.
        
    Returns:
        np.ndarray of dropout probabilities, shape (n_students,).
    """
    model.eval()
    dataset = StudentSequenceDataset(X, np.zeros(len(X)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            preds = model(X_batch).squeeze()
            all_preds.extend(preds.numpy())
    
    return np.array(all_preds)


def save_lstm_model(model: LSTMDropoutPredictor, path: str) -> None:
    """Save LSTM model state dict to disk."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': model.lstm.input_size,
        'hidden_size': model.lstm.hidden_size,
        'num_layers': model.lstm.num_layers,
    }, path)
    print(f"  ✓ LSTM model saved to {path}")


def load_lstm_model(path: str) -> LSTMDropoutPredictor:
    """Load LSTM model from disk."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = LSTMDropoutPredictor(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✓ LSTM model loaded from {path}")
    return model
