"""
Quick test script to verify dual-backbone implementation.
Tests model construction and forward pass with sample data.
"""
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.model.backbone import EMGScalogramBackbone, AngleHistoryBackbone, DualBackbone
from src.model.predictor import DeterministicModel, ProbabilisticModel

def test_emg_backbone():
    """Test EMG scalogram backbone."""
    print("\n=== Testing EMG Scalogram Backbone ===")
    
    # Create sample data: (batch=2, channels=8, time=100, freq=40)
    emg_data = torch.randn(2, 8, 100, 40)
    
    # Test conv2d_lstm
    print("\nTest 1: Conv2D + LSTM backbone")
    backbone = EMGScalogramBackbone(
        n_channels=8,
        n_freq_scales=40,
        hidden_dim=128,
        backbone_type='conv2d_lstm'
    )
    features = backbone(emg_data)
    print(f"  Input shape: {emg_data.shape}")
    print(f"  Output shape: {features.shape}")
    assert features.shape == (2, 128), f"Expected (2, 128), got {features.shape}"
    print("  ✓ Conv2D+LSTM backbone works!")
    
    # Test flatten_lstm
    print("\nTest 2: Flatten + LSTM backbone")
    backbone = EMGScalogramBackbone(
        n_channels=8,
        n_freq_scales=40,
        hidden_dim=128,
        backbone_type='flatten_lstm'
    )
    features = backbone(emg_data)
    print(f"  Input shape: {emg_data.shape}")
    print(f"  Output shape: {features.shape}")
    assert features.shape == (2, 128), f"Expected (2, 128), got {features.shape}"
    print("  ✓ Flatten+LSTM backbone works!")

def test_angle_backbone():
    """Test angle history backbone."""
    print("\n=== Testing Angle History Backbone ===")
    
    # Create sample data: (batch=2, angles=6, time=100)
    angle_data = torch.randn(2, 6, 100)
    
    # Test LSTM
    print("\nTest 1: LSTM backbone")
    backbone = AngleHistoryBackbone(
        n_angles=6,
        hidden_dim=64,
        backbone_type='lstm'
    )
    features = backbone(angle_data)
    print(f"  Input shape: {angle_data.shape}")
    print(f"  Output shape: {features.shape}")
    assert features.shape == (2, 64), f"Expected (2, 64), got {features.shape}"
    print("  ✓ LSTM backbone works!")
    
    # Test TCN
    print("\nTest 2: TCN backbone")
    backbone = AngleHistoryBackbone(
        n_angles=6,
        hidden_dim=64,
        backbone_type='tcn'
    )
    features = backbone(angle_data)
    print(f"  Input shape: {angle_data.shape}")
    print(f"  Output shape: {features.shape}")
    assert features.shape == (2, 64), f"Expected (2, 64), got {features.shape}"
    print("  ✓ TCN backbone works!")

def test_dual_backbone():
    """Test dual-backbone with all feature modes."""
    print("\n=== Testing Dual Backbone ===")
    
    # Create sample data
    emg_data = torch.randn(2, 8, 100, 40)
    angle_data = torch.randn(2, 6, 100)
    
    # Create sub-backbones
    emg_bb = EMGScalogramBackbone(8, 40, 128, 'conv2d_lstm')
    angle_bb = AngleHistoryBackbone(6, 64, 'lstm')
    
    # Test 'both' mode
    print("\nTest 1: Feature mode = 'both'")
    dual_bb = DualBackbone(emg_bb, angle_bb, feature_mode='both', fusion_hidden_dim=128)
    features = dual_bb(emg_data=emg_data, angle_data=angle_data)
    print(f"  EMG shape: {emg_data.shape}")
    print(f"  Angle shape: {angle_data.shape}")
    print(f"  Output shape: {features.shape}")
    assert features.shape == (2, 128), f"Expected (2, 128), got {features.shape}"
    print("  ✓ 'both' mode works!")
    
    # Test 'emg_only' mode
    print("\nTest 2: Feature mode = 'emg_only'")
    dual_bb = DualBackbone(emg_bb, angle_bb, feature_mode='emg_only', fusion_hidden_dim=128)
    features = dual_bb(emg_data=emg_data)
    print(f"  EMG shape: {emg_data.shape}")
    print(f"  Output shape: {features.shape}")
    assert features.shape == (2, 128), f"Expected (2, 128), got {features.shape}"
    print("  ✓ 'emg_only' mode works!")
    
    # Test 'angle_only' mode
    print("\nTest 3: Feature mode = 'angle_only'")
    dual_bb = DualBackbone(emg_bb, angle_bb, feature_mode='angle_only', fusion_hidden_dim=128)
    features = dual_bb(angle_data=angle_data)
    print(f"  Angle shape: {angle_data.shape}")
    print(f"  Output shape: {features.shape}")
    assert features.shape == (2, 128), f"Expected (2, 128), got {features.shape}"
    print("  ✓ 'angle_only' mode works!")

def test_full_model():
    """Test complete predictor models."""
    print("\n=== Testing Full Models ===")
    
    # Create sample data
    emg_data = torch.randn(2, 8, 100, 40)
    angle_data = torch.randn(2, 6, 100)
    
    # Create dual backbone
    emg_bb = EMGScalogramBackbone(8, 40, 128, 'conv2d_lstm')
    angle_bb = AngleHistoryBackbone(6, 64, 'lstm')
    dual_bb = DualBackbone(emg_bb, angle_bb, feature_mode='both', fusion_hidden_dim=128)
    
    # Test deterministic model
    print("\nTest 1: Deterministic model")
    model = DeterministicModel(dual_bb)
    prediction = model(emg_data=emg_data, angle_data=angle_data)
    print(f"  Input: EMG {emg_data.shape}, Angle {angle_data.shape}")
    print(f"  Output shape: {prediction.shape}")
    assert prediction.shape == (2, 1), f"Expected (2, 1), got {prediction.shape}"
    print("  ✓ Deterministic model works!")
    
    # Test probabilistic model
    print("\nTest 2: Probabilistic model")
    model = ProbabilisticModel(dual_bb)
    pred_mean, pred_logvar = model(emg_data=emg_data, angle_data=angle_data)
    print(f"  Input: EMG {emg_data.shape}, Angle {angle_data.shape}")
    print(f"  Mean shape: {pred_mean.shape}")
    print(f"  LogVar shape: {pred_logvar.shape}")
    assert pred_mean.shape == (2, 1), f"Expected (2, 1), got {pred_mean.shape}"
    assert pred_logvar.shape == (2, 1), f"Expected (2, 1), got {pred_logvar.shape}"
    print("  ✓ Probabilistic model works!")

def test_backward_pass():
    """Test that gradients flow correctly."""
    print("\n=== Testing Backward Pass ===")
    
    # Create sample data
    emg_data = torch.randn(2, 8, 100, 40)
    angle_data = torch.randn(2, 6, 100)
    labels = torch.randn(2, 1)
    
    # Create model
    emg_bb = EMGScalogramBackbone(8, 40, 128, 'conv2d_lstm')
    angle_bb = AngleHistoryBackbone(6, 64, 'lstm')
    dual_bb = DualBackbone(emg_bb, angle_bb, feature_mode='both', fusion_hidden_dim=128)
    model = DeterministicModel(dual_bb)
    
    # Forward pass
    prediction = model(emg_data=emg_data, angle_data=angle_data)
    
    # Compute loss
    loss = torch.nn.functional.mse_loss(prediction, labels)
    print(f"  Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads, "No gradients computed!"
    print("  ✓ Gradients computed successfully!")

if __name__ == '__main__':
    print("=" * 60)
    print("Dual-Backbone Implementation Test")
    print("=" * 60)
    
    try:
        test_emg_backbone()
        test_angle_backbone()
        test_dual_backbone()
        test_full_model()
        test_backward_pass()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
