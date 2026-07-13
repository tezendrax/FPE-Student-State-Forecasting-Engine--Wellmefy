import os
import sys
import torch
import pytest
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpe.model import TemporalFusionTransformer, GatedResidualNetwork, LinearBaselineFallback
from fpe.pipeline import QuantileLoss

def test_gated_residual_network():
    # Test GRN forward pass and dimensions mapping
    batch, length, in_dim, out_dim = 4, 10, 8, 16
    x = torch.randn(batch, length, in_dim)
    
    grn = GatedResidualNetwork(input_dim=in_dim, hidden_dim=12, output_dim=out_dim)
    out = grn(x)
    
    assert out.shape == (batch, length, out_dim)
    assert not torch.isnan(out).any()

def test_temporal_fusion_transformer_forward():
    # Test TFT forward pass and output structures
    batch_size = 2
    lookback = 14
    horizon = 7
    
    x_hist = torch.randn(batch_size, lookback, 17) # 17 historical features
    x_future = torch.randn(batch_size, horizon, 3)  # 3 future features
    static_cov = torch.randn(batch_size, 10)         # 10 static features
    
    model = TemporalFusionTransformer(
        num_hist_features=17,
        num_future_features=3,
        num_static_features=10,
        hidden_dim=16,
        num_heads=2,
        num_targets=10
    )
    
    preds = model(x_hist, x_future, static_cov)
    
    assert isinstance(preds, dict)
    assert "p10" in preds
    assert "p50" in preds
    assert "p90" in preds
    
    for key in ["p10", "p50", "p90"]:
        assert preds[key].shape == (batch_size, horizon, 10)
        assert not torch.isnan(preds[key]).any()
        
    # Check quantile ordering: p10 <= p50 <= p90
    assert torch.all(preds["p10"] <= preds["p50"])
    assert torch.all(preds["p50"] <= preds["p90"])

def test_quantile_loss():
    # Test pinball loss calculations
    preds = {
        "p10": torch.ones(2, 7, 10) * 0.3,
        "p50": torch.ones(2, 7, 10) * 0.5,
        "p90": torch.ones(2, 7, 10) * 0.8
    }
    target = torch.ones(2, 7, 10) * 0.6
    
    criterion = QuantileLoss()
    loss = criterion(preds, target)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0.0

def test_linear_fallback_forecaster():
    # Test linear trend extrapolation fallback
    lookback = 14
    horizon = 7
    num_dims = 10
    
    # Linear trend: values increasing from 0.1 to 0.5
    history = np.zeros((lookback, num_dims))
    for d in range(num_dims):
        history[:, d] = np.linspace(0.1, 0.5, lookback)
        
    forecaster = LinearBaselineFallback(lookback_days=lookback, horizon_days=horizon)
    results = forecaster.forecast(history)
    
    assert isinstance(results, dict)
    for key in ["p10", "p50", "p90"]:
        assert results[key].shape == (horizon, num_dims)
        assert np.all(results[key] >= 0.0)
        assert np.all(results[key] <= 1.0)
        
    # Check that predictions project the upward trend (greater than 0.5)
    assert np.all(results["p50"][0] > 0.5)
    # Check quantile constraint
    assert np.all(results["p10"] <= results["p50"])
    assert np.all(results["p50"] <= results["p90"])
