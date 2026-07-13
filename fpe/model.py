import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, Tuple

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) to allow non-linear gating."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., dim)
        gate_input = self.fc(x)
        val, gate = gate_input.chunk(2, dim=-1)
        return val * torch.sigmoid(gate)

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) for routing information selectively."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1, context_dim: int = None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_fc = None
            
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Residual projections
        if input_dim != output_dim:
            self.res_proj = nn.Linear(input_dim, output_dim)
        else:
            self.res_proj = None
            
        if hidden_dim != output_dim:
            self.out_proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.out_proj = None

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        residual = self.res_proj(x) if self.res_proj is not None else x
        
        h = self.fc1(x)
        if context is not None and self.context_fc is not None:
            h = h + self.context_fc(context)
            
        h = self.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.glu(h)
        
        h = self.layer_norm(h)
        if self.out_proj is not None:
            h = self.out_proj(h)
            
        return h + residual

class TemporalFusionTransformer(nn.Module):
    """
    A lightweight, CPU-optimized custom implementation of the Temporal Fusion Transformer (TFT) architecture.
    Provides quantile forecasts (p10, p50, p90) for D wellness dimensions.
    """
    def __init__(
        self, 
        num_hist_features: int = 17, 
        num_future_features: int = 3, 
        num_static_features: int = 10,
        hidden_dim: int = 16, 
        num_heads: int = 2, 
        num_targets: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_targets = num_targets
        
        # 1. Feature Projections
        self.hist_proj = nn.Linear(num_hist_features, hidden_dim)
        self.future_proj = nn.Linear(num_future_features, hidden_dim)
        self.static_proj = nn.Linear(num_static_features, hidden_dim)
        
        # 2. Static Enrichment
        self.static_enrich = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        
        # 3. Temporal Processing (Self-Attention Layer)
        # Using standard multihead attention to model step-to-step relationships
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_layernorm = nn.LayerNorm(hidden_dim)
        
        # 4. Decoder / Output GRN
        self.decoder_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout, context_dim=hidden_dim)
        
        # 5. Quantile Output Heads
        # Generates shape (batch, horizon, num_targets, 3) where the last index corresponds to [p10, p50, p90]
        self.quantile_head = nn.Linear(hidden_dim, num_targets * 3)

    def forward(self, x_hist: torch.Tensor, x_future: torch.Tensor, static_cov: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x_hist: (batch, lookback_len, num_hist_features)
        # x_future: (batch, horizon_len, num_future_features)
        # static_cov: (batch, num_static_features)
        
        batch_size = x_hist.size(0)
        lookback_len = x_hist.size(1)
        horizon_len = x_future.size(1)
        
        # 1. Project inputs
        hist_emb = self.hist_proj(x_hist)      # (batch, lookback_len, hidden_dim)
        future_emb = self.future_proj(x_future)  # (batch, horizon_len, hidden_dim)
        static_emb = self.static_proj(static_cov) # (batch, hidden_dim)
        
        # Enrich static covariates
        static_context = self.static_enrich(static_emb) # (batch, hidden_dim)
        
        # 2. Temporal Fusion Decoder
        # Concatenate history and future tokens for attention
        full_sequence = torch.cat([hist_emb, future_emb], dim=1) # (batch, lookback_len + horizon_len, hidden_dim)
        
        # Multi-head attention pass
        attn_out, _ = self.self_attn(full_sequence, full_sequence, full_sequence)
        full_sequence = self.attn_layernorm(full_sequence + attn_out)
        
        # Squeeze out the decoder part (horizon steps)
        decoder_steps = full_sequence[:, lookback_len:] # (batch, horizon_len, hidden_dim)
        
        # Apply Decoder GRN with static context injection
        # Broadcast static context to fit sequence steps
        static_context_expanded = static_context.unsqueeze(1).expand(-1, horizon_len, -1)
        decoder_out = self.decoder_grn(decoder_steps, static_context_expanded) # (batch, horizon_len, hidden_dim)
        
        # 3. Quantile Projections
        quantiles_raw = self.quantile_head(decoder_out) # (batch, horizon_len, num_targets * 3)
        quantiles_raw = quantiles_raw.view(batch_size, horizon_len, self.num_targets, 3)
        
        # Return dict of p10, p50, p90 tensors each of shape (batch, horizon_len, num_targets)
        p10 = quantiles_raw[..., 0]
        p50 = quantiles_raw[..., 1]
        p90 = quantiles_raw[..., 2]
        
        # Enforce quantile constraints: p10 <= p50 <= p90
        # By sorting along the quantile dimension
        quantiles_sorted, _ = torch.sort(quantiles_raw, dim=-1)
        
        return {
            "p10": quantiles_sorted[..., 0],
            "p50": quantiles_sorted[..., 1],
            "p90": quantiles_sorted[..., 2]
        }


# ==========================================
# Linear Regression Baseline Fallback Model
# ==========================================

class LinearBaselineFallback:
    """
    Fallback baseline forecaster. Fits a standard linear regression model on a 
    student's history and projects 7 days into the future.
    """
    def __init__(self, lookback_days: int = 14, horizon_days: int = 7):
        self.lookback_days = lookback_days
        self.horizon_days = horizon_days
        
    def forecast(self, history_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        history_matrix: (lookback_days, D) where D is the dimension size (10)
        Returns:
            Dict of {"p10": (horizon, D), "p50": (horizon, D), "p90": (horizon, D)}
        """
        lookback, num_dims = history_matrix.shape
        x_train = np.arange(lookback).reshape(-1, 1)
        x_predict = np.arange(lookback, lookback + self.horizon_days).reshape(-1, 1)
        
        p50_preds = np.zeros((self.horizon_days, num_dims))
        
        for d in range(num_dims):
            y_train = history_matrix[:, d]
            model = LinearRegression()
            model.fit(x_train, y_train)
            
            preds = model.predict(x_predict)
            # Clip values to valid range [0.0, 1.0]
            p50_preds[:, d] = np.clip(preds, 0.0, 1.0)
            
        # Add simple confidence bounds based on history variance
        residuals_std = np.std(history_matrix, axis=0)
        
        # 10th and 90th percentiles are estimated using standard normal scaling
        # p10 = p50 - 1.28 * std, p90 = p50 + 1.28 * std
        p10_preds = np.zeros_like(p50_preds)
        p90_preds = np.zeros_like(p50_preds)
        
        for d in range(num_dims):
            std_val = max(0.02, residuals_std[d]) # floor std dev at 0.02
            p10_preds[:, d] = np.clip(p50_preds[:, d] - 1.28 * std_val, 0.0, 1.0)
            p90_preds[:, d] = np.clip(p50_preds[:, d] + 1.28 * std_val, 0.0, 1.0)
            
        # Ensure quantile ordering
        for d in range(num_dims):
            for t in range(self.horizon_days):
                vals = sorted([p10_preds[t, d], p50_preds[t, d], p90_preds[t, d]])
                p10_preds[t, d], p50_preds[t, d], p90_preds[t, d] = vals
                
        return {
            "p10": p10_preds,
            "p50": p50_preds,
            "p90": p90_preds
        }
