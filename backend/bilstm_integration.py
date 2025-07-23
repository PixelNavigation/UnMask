"""
Enhanced Temporal Bi-LSTM Module
Contains the Enhanced Bi-LSTM class for temporal video processing
"""

import torch
import torch.nn as nn
import numpy as np

class EnhancedTemporalBiLSTM(nn.Module):
    """
    Enhanced Bi-LSTM with rich features, uncertainty modeling, and temporal smoothing
    
    Features:
    - Rich intermediate features (512D)
    - Monte Carlo dropout uncertainty estimation
    - Temporal smoothing with EMA
    - Multi-head attention mechanism
    - NO RETRAINING of existing models required
    """
    def __init__(self, input_size=512, hidden_size=128, num_layers=3, sequence_length=10, dropout_rate=0.4):
        super(EnhancedTemporalBiLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        
        # Feature projection layer for rich embeddings
        self.feature_projection = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Enhanced Bi-LSTM for temporal pattern analysis
        self.temporal_lstm = nn.LSTM(
            input_size=128,  # After feature projection
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # Multi-head attention mechanism
        self.attention_heads = 4
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=self.attention_heads,
            dropout=0.2,
            batch_first=True
        )
        
        # Uncertainty-aware classification with Monte Carlo Dropout
        self.uncertainty_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Temporal smoothing parameters
        self.ema_alpha = 0.3
        self.previous_prediction = None
    
    def monte_carlo_predict(self, x, num_samples=10):
        """
        Monte Carlo Dropout for uncertainty estimation
        
        Args:
            x: Input tensor (batch, sequence_length, features)
            num_samples: Number of forward passes for uncertainty estimation
            
        Returns:
            mean_pred: Mean prediction across samples
            uncertainty: Standard deviation (uncertainty) across samples
        """
        self.train()  # Enable dropout during inference
        predictions = []
        
        for _ in range(num_samples):
            pred, _ = self.forward(x)
            predictions.append(pred)
        
        self.eval()  # Back to eval mode
        predictions = torch.stack(predictions)
        
        # Calculate mean and uncertainty
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)
        
        return mean_pred, uncertainty
    
    def temporal_smooth(self, current_pred):
        """
        Exponential Moving Average for temporal smoothing
        
        Args:
            current_pred: Current prediction tensor
            
        Returns:
            smoothed: Temporally smoothed prediction
        """
        if self.previous_prediction is None:
            self.previous_prediction = current_pred
            return current_pred
        
        # EMA smoothing
        smoothed = self.ema_alpha * current_pred + (1 - self.ema_alpha) * self.previous_prediction
        self.previous_prediction = smoothed
        return smoothed
    
    def reset_temporal_state(self):
        """Reset temporal smoothing state for new video session"""
        self.previous_prediction = None
    
    def forward(self, x):
        """
        Forward pass through Enhanced Temporal Bi-LSTM
        
        Args:
            x: Input tensor (batch, sequence_length, rich_features)
            
        Returns:
            prediction: Final prediction (batch, 1)
            attention_weights: Attention weights for interpretability
        """
        # x shape: (batch, sequence_length, rich_features)
        
        # Project rich features to manageable size
        projected_features = self.feature_projection(x)
        
        # Bi-LSTM processing
        lstm_out, _ = self.temporal_lstm(projected_features)
        
        # Multi-head attention
        attended_features, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Use mean of attended features
        final_features = torch.mean(attended_features, dim=1)
        
        # Classification with uncertainty
        prediction = self.uncertainty_classifier(final_features)
        
        return prediction, attention_weights

# Test function for the module
if __name__ == "__main__":
    print("🧪 Testing Enhanced Temporal Bi-LSTM Module")
    print("=" * 50)
    
    # Create model instance
    model = EnhancedTemporalBiLSTM(
        input_size=512,
        hidden_size=128,
        sequence_length=10
    )
    
    # Test with random data
    test_sequence = torch.randn(1, 10, 512)  # (batch, time_steps, features)
    
    # Standard forward pass
    prediction, attention = model(test_sequence)
    print(f"✅ Forward pass: prediction shape {prediction.shape}")
    print(f"✅ Attention shape: {attention.shape}")
    
    # Monte Carlo uncertainty
    mean_pred, uncertainty = model.monte_carlo_predict(test_sequence, num_samples=5)
    print(f"✅ Monte Carlo: mean={mean_pred.item():.4f}, uncertainty={uncertainty.item():.4f}")
    
    # Temporal smoothing test
    smoothed = model.temporal_smooth(prediction)
    print(f"✅ Temporal smoothing: {prediction.item():.4f} -> {smoothed.item():.4f}")
    
    print("✅ Enhanced Temporal Bi-LSTM module ready for import!")
    print("💡 Usage: from bilstm_integration import EnhancedTemporalBiLSTM")
